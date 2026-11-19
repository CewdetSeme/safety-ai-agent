"""
RTSP reader — threaded frame source for the safety-AI worker.

Responsibilities:
  1. Open an RTSP stream with FFmpeg (TCP transport, 5 s timeout).
  2. Read frames in a background thread (so reads never block callers).
  3. Push frames into a bounded queue; drop oldest when full (live-video rule).
  4. Auto-reconnect on stream drop with exponential backoff.
  5. Expose a simple `read()` / `stop()` / `is_alive()` API.

Run directly for a smoke test:
    python -m worker.rtsp_reader
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module logger — caller can override level via logging.basicConfig().
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small dataclass: what we hand back from read(). Keeping a timestamp with
# each frame lets downstream stages (tracker, event rules, LLM) reason about
# *when* something happened, not just *what*.
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    image: np.ndarray          # BGR numpy array (H, W, 3), uint8
    timestamp: float           # time.time() when the frame was read
    index: int                 # monotonically increasing frame counter


# ---------------------------------------------------------------------------
# The reader itself.
# ---------------------------------------------------------------------------
class RTSPReader:
    """
    Threaded RTSP frame reader.

    Usage:
        reader = RTSPReader(url)
        reader.start()
        frame = reader.read(timeout=2.0)     # blocks up to 2 s
        ...
        reader.stop()

    Or iterate:
        for frame in reader:
            ...
    """

    def __init__(
        self,
        url: str,
        queue_max: int = 30,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 30.0,
        rtsp_timeout_us: int = 5_000_000,   # 5 s
    ) -> None:
        self.url = url
        self.queue_max = queue_max
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.rtsp_timeout_us = rtsp_timeout_us

        # Bounded queue. We use `queue.Queue` (thread-safe) with maxsize.
        # When it fills, we drop the OLDEST frame so latency stays low.
        self._q: "queue.Queue[Frame]" = queue.Queue(maxsize=queue_max)

        # Threading primitives.
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Internal counters.
        self._frame_index = 0
        self._last_ok_ts = 0.0      # timestamp of the last successful read
        self._open_errors = 0       # consecutive reconnect failures

    # -----------------------------------------------------------------------
    # Public API.
    # -----------------------------------------------------------------------
    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread and self._thread.is_alive():
            return   # already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"RTSPReader[{self.url[:40]}]",
            daemon=True,   # die with the main process
        )
        self._thread.start()
        log.info("RTSPReader started for %s", self.url)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and join it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        log.info("RTSPReader stopped")

    def read(self, timeout: float = 2.0) -> Optional[Frame]:
        """
        Return the next frame, or None if the queue is empty within timeout.

        Returning None is normal — the caller (tracker loop) can use that
        moment to check stop conditions or log health.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """True if the reader thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    def last_frame_age(self) -> float:
        """Seconds since the last successful frame read (useful for health checks)."""
        if self._last_ok_ts == 0.0:
            return float("inf")
        return time.time() - self._last_ok_ts

    # -----------------------------------------------------------------------
    # Iteration support — lets you `for frame in reader:`.
    # -----------------------------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        while not self._stop_event.is_set():
            frame = self.read(timeout=0.5)
            if frame is not None:
                return frame
        raise StopIteration

    # -----------------------------------------------------------------------
    # Internal: open the capture with FFmpeg RTSP hints.
    # -----------------------------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        # These env hints affect FFmpeg's RTSP client: prefer TCP (more
        # reliable than UDP) and add a 5 s socket timeout. `setdefault`
        # leaves any user-provided options intact.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;tcp|stimeout;{self.rtsp_timeout_us}",
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {self.url}")
        return cap

    # -----------------------------------------------------------------------
    # Internal: the reader loop — reconnect forever until stop.
    # -----------------------------------------------------------------------
    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        delay = self.reconnect_delay

        while not self._stop_event.is_set():
            # --- Ensure we have an open capture --------------------------
            if cap is None:
                try:
                    cap = self._open_capture()
                    self._open_errors = 0
                    delay = self.reconnect_delay  # reset backoff
                    log.info("Connected to %s", self.url)
                except Exception as exc:  # noqa: BLE001
                    self._open_errors += 1
                    log.warning(
                        "Open failed (%d consecutive): %s — retrying in %.1fs",
                        self._open_errors, exc, delay,
                    )
                    # Sleep in small chunks so stop() is responsive.
                    if self._stop_event.wait(delay):
                        break
                    delay = min(delay * 2, self.max_reconnect_delay)
                    continue

            # --- Read one frame ----------------------------------------
            try:
                ok, image = cap.read()
            except Exception as exc:  # noqa: BLE001
                log.warning("cap.read raised: %s", exc)
                ok, image = False, None

            if not ok or image is None:
                # Stream dropped — release and reconnect.
                log.warning("Stream read failed; reconnecting…")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                continue

            # --- Success: update health + enqueue ----------------------
            self._last_ok_ts = time.time()
            self._frame_index += 1
            frame = Frame(
                image=image,
                timestamp=self._last_ok_ts,
                index=self._frame_index,
            )

            try:
                self._q.put_nowait(frame)
            except queue.Full:
                # Drop the OLDEST frame to make room for the newest.
                # This bounds latency: the tracker always sees fresh frames.
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    # Extremely unlikely, but never block the reader.
                    pass

        # Clean up on exit.
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to see frames/sec and a preview.
# Usage:
#   python -m worker.rtsp_reader "rtsp://..."
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream"
    )

    reader = RTSPReader(url)
    reader.start()

    print(f"Reading from: {url}")
    print("Ctrl+C to stop.\n")

    count = 0
    t0 = time.time()
    try:
        for frame in reader:
            count += 1
            if count % 30 == 0:
                elapsed = time.time() - t0
                fps = count / elapsed
                print(
                    f"frame={frame.index} "
                    f"shape={frame.image.shape} "
                    f"fps={fps:.1f} "
                    f"queue_alive={reader.is_alive()}"
                )
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        reader.stop()