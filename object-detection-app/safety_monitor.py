"""
Safety monitor — RTSP → YOLO → annotated frames → vision LLM → email.

Pipeline:
  1. Consume frames from RTSPReader (Phase 1).
  2. Run YOLO detection + tracking (phase 2).
  3. Draw boxes on the frame (visual signal to the LLM).
  4. Every N seconds, send the annotated JPEG to a vision LLM.
  5. Parse verdict → if unsafe, dedupe → save evidence → send email.

Design notes:
  • One main thread + one RTSP thread. No extra threads needed.
  • LLM is throttled by wall-clock seconds, not per-frame — cost control.
  • Alert dedup uses (track_id, rule_key) with a cooldown window.
  • Everything injectable (llm_client, alert_sink) for testability.

Run directly:
    python -m worker.safety_monitor "rtsp://..." \
        --analyze-every 5 --cooldown 300 \
        --evidence-dir storage/evidence
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import cv2
import numpy as np

from detector import Detection, ObjectDetector
from utils import draw_detections
from rtsp_reader import Frame, RTSPReader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contracts — LLM client and alert sink.
# Both are Protocol-ish (duck-typed). Any object with these methods works.
# ---------------------------------------------------------------------------
class SafetyLLMClient:
    """Interface: analyze(image_b64) -> verdict dict."""
    def analyze(self, image_b64: str) -> dict:
        raise NotImplementedError


class AlertSink:
    """Interface: send(subject, html, evidence_path, context) -> None."""
    def send(self, subject: str, html: str, evidence_path: str, context: dict) -> None:
        raise NotImplementedError


@dataclass
class MonitorStats:
    frames: int = 0
    llm_calls: int = 0
    unsafe_verdicts: int = 0
    alerts_sent: int = 0
    alerts_suppressed_cooldown: int = 0
    llm_errors: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "llm_calls": self.llm_calls,
            "unsafe_verdicts": self.unsafe_verdicts,
            "alerts_sent": self.alerts_sent,
            "alerts_suppressed_cooldown": self.alerts_suppressed_cooldown,
            "llm_errors": self.llm_errors,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Frame watermark: timestamp + camera id burned into the frame.
# Reason: LLM and human reviewers see WHEN/WHERE. Prevents "what time was this?"
# ---------------------------------------------------------------------------
def _watermark(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    # Black strip on top for legibility regardless of scene.
    h, w = out.shape[:2]
    strip_h = max(28, h // 30)
    cv2.rectangle(out, (0, 0), (w, strip_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out, label, (8, strip_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# JPEG base64 encoder for vision APIs.
# ---------------------------------------------------------------------------
def _encode_jpeg_b64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    import base64
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Verdict parser / safety filter.
# ---------------------------------------------------------------------------
def _is_alert_worthy(verdict: dict, min_confidence: float = 0.7) -> bool:
    """
    True if the LLM said 'unsafe' AND at least one person has
    helmet=False with confidence >= min_confidence.
    """
    if not verdict.get("unsafe"):
        return False
    for p in verdict.get("persons", []):
        if p.get("helmet") is False and float(p.get("confidence", 0)) >= min_confidence:
            return True
    return False


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------
def run_safety_monitor(
    reader: RTSPReader,
    detector: ObjectDetector,
    llm_client: SafetyLLMClient,
    alert_sink: AlertSink,
    analyze_every_s: float = 5.0,
    alert_cooldown_s: float = 300.0,
    min_llm_confidence: float = 0.7,
    class_filter: Optional[List[str]] = ["person"],
    evidence_dir: str = "storage/evidence",
    camera_label: str = "camera-1",
    report_every: int = 30,
) -> MonitorStats:
    """
    Run the RTSP → YOLO → LLM → email loop until SIGINT/SIGTERM.
    """
    import os
    os.makedirs(evidence_dir, exist_ok=True)

    stats = MonitorStats()
    start_ts = time.time()
    stop = threading.Event()

    def _handle_signal(signum, _frame):
        log.info("Signal %s received — shutting down", signum)
        stop.set()

    prev_int = signal.signal(signal.SIGINT, _handle_signal)
    prev_term = signal.signal(signal.SIGTERM, _handle_signal)

    # Cooldown tracking: key = (track_id, "helmet") → last alert time
    last_alert: dict[tuple[int, str], float] = {}
    last_analysis_ts = 0.0
    error_log_cap = 3
    error_log_count = 0

    log.info("Safety monitor started; analyze every %.1fs", analyze_every_s)

    try:
        while not stop.is_set():
            if not reader.is_alive():
                log.warning("Reader died — stopping monitor")
                break

            frame = reader.read(timeout=0.5)
            if frame is None:
                continue

            # --- Inference ------------------------------------------------
            try:
                detections = detector.track(frame.image, class_filter)
            except Exception as exc:   # noqa: BLE001
                error_log_count += 1
                if error_log_count <= error_log_cap:
                    log.warning("Inference failed on frame %d: %s",
                                frame.index, exc)
                elif error_log_count == error_log_cap + 1:
                    log.warning("Further inference errors will be suppressed")
                continue

            stats.frames += 1

            # --- Throttle LLM by wall-clock -------------------------------
            now = time.time()
            if now - last_analysis_ts < analyze_every_s:
                continue
            last_analysis_ts = now

            # --- Draw + watermark ---------------------------------------
            annotated = draw_detections(frame.image, detections)
            ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            watermarked = _watermark(annotated, f"{camera_label} · {ts_str} · frame {frame.index}")

            # --- Encode + call LLM --------------------------------------
            try:
                image_b64 = _encode_jpeg_b64(watermarked)
                verdict = llm_client.analyze(image_b64)
                stats.llm_calls += 1
            except Exception as exc:   # noqa: BLE001
                stats.llm_errors += 1
                log.warning("LLM call failed: %s", exc)
                continue

            # --- Decide: alert? -----------------------------------------
            if not _is_alert_worthy(verdict, min_confidence=min_llm_confidence):
                continue
            stats.unsafe_verdicts += 1

            # --- Dedupe per track_id -----------------------------------
            # Find the highest-confidence offending person.
            offending = [
                p for p in verdict.get("persons", [])
                if p.get("helmet") is False
                and float(p.get("confidence", 0)) >= min_llm_confidence
            ]
            offending.sort(key=lambda p: float(p.get("confidence", 0)), reverse=True)
            top = offending[0] if offending else None
            track_id = int(top.get("track_id", -1)) if top else -1

            key = (track_id, "helmet")
            last_ts = last_alert.get(key, 0.0)
            if now - last_ts < alert_cooldown_s:
                stats.alerts_suppressed_cooldown += 1
                continue

            # --- Save evidence ----------------------------------------
            fname = f"{int(now)}_{camera_label}_track{track_id}.jpg"
            path = os.path.join(evidence_dir, fname)
            cv2.imwrite(path, watermarked)

            # --- Compose + send email ---------------------------------
            summary = verdict.get("summary", "Unsafe condition detected")
            subject = f"🚨 Safety alert: {summary[:80]}"
            html = _render_alert_html(
                summary=summary,
                verdict=verdict,
                camera_label=camera_label,
                ts_str=ts_str,
                frame_index=frame.index,
                evidence_filename=fname,
            )
            try:
                alert_sink.send(subject, html, path, context=verdict)
                stats.alerts_sent += 1
                last_alert[key] = now
                log.info("Alert sent for track %d: %s", track_id, summary[:60])
            except Exception as exc:   # noqa: BLE001
                log.warning("Alert send failed: %s", exc)

            # --- Periodic log -----------------------------------------
            if stats.frames % report_every == 0:
                elapsed = max(time.time() - start_ts, 1e-6)
                log.info(
                    "frames=%d llm_calls=%d unsafe=%d alerts=%d fps=%.1f",
                    stats.frames, stats.llm_calls, stats.unsafe_verdicts,
                    stats.alerts_sent, stats.frames / elapsed,
                )

        stats.elapsed_s = max(time.time() - start_ts, 1e-6)
        log.info("Monitor stopped: %s", stats.as_dict())
        return stats

    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


# ---------------------------------------------------------------------------
# Minimal HTML template for the email body.
# ---------------------------------------------------------------------------
def _render_alert_html(
    summary: str,
    verdict: dict,
    camera_label: str,
    ts_str: str,
    frame_index: int,
    evidence_filename: str,
) -> str:
    persons = verdict.get("persons", [])
    rows = "".join(
        f"<tr>"
        f"<td>{p.get('track_id', '?')}</td>"
        f"<td>{'✅' if p.get('helmet') is True else '❌' if p.get('helmet') is False else '❔'}</td>"
        f"<td>{p.get('confidence', 0):.2f}</td>"
        f"<td>{p.get('notes', '')}</td>"
        f"</tr>"
        for p in persons
    )
    return f"""
    <h2>🚨 Safety alert: {summary}</h2>
    <p><b>Camera:</b> {camera_label}<br>
       <b>Time:</b> {ts_str}<br>
       <b>Frame:</b> {frame_index}</p>
    <img src="cid:evidence" width="640" alt="Evidence frame"><br><br>
    <table border="1" cellpadding="4" cellspacing="0">
      <tr><th>Track</th><th>Helmet</th><th>Confidence</th><th>Notes</th></tr>
      {rows}
    </table>
    <p><small>Evidence file: {evidence_filename}</small></p>
    """