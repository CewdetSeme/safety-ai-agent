"""
Tracking loop — consumes frames from RTSPReader, runs YOLO track/detect,
and writes one JSON line per detection to disk.

Design summary:
  • Runs on the main thread. RTSPReader already provides the reader thread,
    so no second background thread is needed here.
  • Per-frame try/except: one bad frame cannot kill the worker.
  • Graceful shutdown via SIGINT / SIGTERM → sets a stop Event → loop exits
    → log file flushed → stats returned.
  • JSONL output: one complete JSON object per line, safe under crash
    because each `\n` ends a valid record.
  • Dropped-frame detection: gap in Phase 1's frame.index counts frames
    the consumer missed. Surfaces "am I falling behind?" in logs.
Run directly for a smoke test:
    python -m worker.track_loop "rtsp://..." ./storage/tracks.jsonl
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from dataclasses import dataclass

from typing import Callable, List, Optional


from rtsp_reader import RTSPReader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stats container — same reason Frame exists: downstream reads a single object.
# ---------------
@dataclass
class TrackerStats:
    frames: int = 0
    detections: int = 0 
    dropped_frames: int = 0
    elapsed_s:float = 0.0

    def _as_dict(self)-> dict:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "dropped_frames": self.dropped_frames,
            "elapsed_s": round(self.elapsed_s, 3),
            "fps": round(self.frames / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Helper: serialize one Detection into a stable JSON dict for JSONL.
# Keep field names flat (no nested objects) so `pandas.read_json(lines=True)`
# and simple `grep` work without transformers.
# ----------
def _detection_to_record(frame: Frame, Detection: detection)-> dict:
    return {
        "ts": round(frame.ts, 4),
        "frame": frame.index,              # monotonic frame index
        "track_id": det.track_id,          # None when tracking disabled
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": round(det.confidence, 4),
        "x1": det.bbox[0],
        "y1": det.bbox[1],
        "x2": det.bbox[2],
        "y2": det.bbox[3],
    }

# ---------------------------------------------------------------------------
# Main loop.
# ---------------------
def run_tracker(
    reader: RTSPReader,
    detector: ObjectDetector,
    output_path: str,
    class_filter: Optional[List[str]] = None,
    enable_tracking: bool = True,
    on_detection: Optional[Callable[[Frame, List[Detection]], None]] = None,
    report_every:int = 30,
)-> TrackerStats:
    """Run the tracking loop until the reader stops or a signal is received.

    Args:
        reader: RTSPReader instance to read frames from.
        detector: ObjectDetector instance to run detection/tracking.
        output_path: Path to write JSONL output.
        class_filter: Optional list of class names to filter detections.
        enable_tracking: Whether to enable tracking (track IDs).
        on_detection: Optional callback for each frame's detections.
        report_every: How often (in seconds) to log progress.   
    """
    stats = TrackerStats()
    start_ts = time.time()

    # Stop Event — shared between the signal handler and the main loop.
    # Why Event and not a plain bool? Event.wait() is interruptible.
    stop = threading.Event()

    # Signal handlers: SIGINT (Ctrl+C) and SIGTERM (Docker/systemd stop)
    # both just flip the flag. The loop then exits at the next read() timeout.
    def _handle_signal(signum, _frame):
        log.info("Signal %s received — shutting down", signum)
        stop.set()

    # Restore the previous handlers at the end so importing this module
    # in a larger app doesn't permanently hijack signals.
    prev_int = signal.signal(signal.SIGINT, _handle_signal)
    prev_term = signal.signal(signal.SIGTERM, _handle_signal)
    
    last_frame_index: Optional[int] = None
    error_log_count = 0
    error_log_cap = 3


    # Open the log file in line-buffered text mode. `buffering=1` means
    # every `\n` flushes — critical for crash-safe JSONL and for tail -f.
    log_f = open(output_path, "a", buffering=1, encoding="utf-8")
    log.info("Tracking loop started; writing to %s", output_path)

    try:
        while not stop.is_set():
            # --- Health: exit if the reader thread has died ----------
            if not reader.is_alive():
                log.error("Reader is no longer alive — stopping tracker")
                break
            # --- Read with short timeout so we re-check stop + health --
            frame = reader.read(timeout=0.5)
            if frame is None:
                # No frame available yet; check stop flag and loop again.
                continue
            # --- Track dropped frames via index gap ---------------------
            if last_frame_index is not None and frame.index > last_frame_index + 1:
                stats.dropped_frames += frame.index - last_frame_index - 1
            last_frame_index = frame.index
            # --- Inference: isolate failure per frame --------
            try:
                if enable_tracking:
                    detections = detector.track(frame.image, class_filter)
                else:
                    detections = detector.detect(frame.image, class_filter)
            except Exception as exc:
                # Keep the worker alive; log only the first few to avoid
                # spamming a broken frame every second.
                error_log_count += 1
                if error_log_count <= error_log_cap:
                    log.warning("Inference failed on frame %d: %s",
                                frame.index, exc)
                elif error_log_count == error_log_cap + 1:
                    log.warning("Further inference errors will be suppressed")
                continue

            # --- Emit one JSONL line per detection -------
            for det in detections:
                record = _detection_to_record(frame, det)
                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.detections += 1
                if on_detection is not None:
                    try:
                        on_detection(frame, det)
                    except Exception as exc:   # noqa: BLE001
                        # A misbehaving callback must not kill the loop.
                        log.warning("on_detection callback raised: %s", exc)

            stats.frames += 1

            # --- Periodic progress log ---------------------------------
            if stats.frames % report_every == 0:
                elapsed = max(time.time() - start_ts, 1e-6)
                fps = stats.frames / elapsed
                log.info(
                    "frames=%d detections=%d dropped=%d fps=%.1f",
                    stats.frames, stats.detections,
                    stats.dropped_frames, fps,
                )
            
        # --- Final flush + summary ----
        log_f.flush()
        stats.elapsed_s = max(time.time() - start_ts, 1e-6)
        log.info("Tracking loop stopped: %s", stats.as_dict())
        return stats
    
    finally:
        # Restore previous signal handlers so importing this module
        # doesn't permanently hijack signals.
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        log_f.close()



# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m worker.track_loop <rtsp_url> [output.jsonl]")
        sys.exit(1)

    url = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "storage/tracks.jsonl"

    # Two-phase bootstrap: reader first, detector second, then loop.
    from detector import ObjectDetector

    reader = RTSPReader(url)
    reader.start()

    detector = ObjectDetector("yolov8n.pt", conf_threshold=0.4)
    log.info("Detector loaded. Classes: %d", len(detector.class_names))

    stats = run_tracker(reader, detector, output_path=output)
    reader.stop()
    print(stats.as_dict())