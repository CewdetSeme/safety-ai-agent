"""Streamlit UI for YOLOv8 object detection + tracking — with RTSP support."""

from __future__ import annotations

import os
import tempfile
import time
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import Detection, ObjectDetector
from utils import (
    bgr_to_rgb,
    draw_detections,
    make_writer,
    open_video,
    video_properties,
)

# ---------------------------------------------------------------------------
# Page config — first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
)

MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
TRACKER_OPTIONS = ["bytetrack.yaml", "botsort.yaml"]
IMAGE_EXT = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_EXT = ["mp4", "avi", "mov", "mkv", "webm"]


# ---------------------------------------------------------------------------
# Cached model loader.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(
    model_path: str,
    conf: float,
    iou: float,
    tracker: str,
) -> ObjectDetector:
    return ObjectDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Sidebar.
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")

    # --- Model settings -------------------------------------------------
    st.sidebar.subheader("Model")
    model_path = st.sidebar.selectbox("Weights", MODEL_OPTIONS, index=0)
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

    # --- Source choice (NEW: RTSP added) --------------------------------
    st.sidebar.subheader("Input source")
    source = st.sidebar.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image", "RTSP Stream"],
        index=0,
    )

    # --- Tracking -------------------------------------------------------
    st.sidebar.subheader("Tracking")
    enable_tracking = st.sidebar.checkbox(
        "Enable object tracking (video / RTSP only)",
        value=True,
        help="Assigns persistent IDs across frames. Only used for videos and streams.",
    )
    tracker_type = st.sidebar.selectbox(
        "Tracker algorithm",
        TRACKER_OPTIONS,
        index=0,
        disabled=not enable_tracking,
    )

    # --- Video / RTSP shared controls -----------------------------------
    skip = 1
    max_frames = 0
    save_video = True
    rtsp_url = ""
    rtsp_stop_after = 0

    if source in ("Upload Video", "RTSP Stream"):
        skip = st.sidebar.slider(
            "Process every Nth frame", 1, 10, 1,
            help="Higher = faster but choppier output.",
        )

    if source == "Upload Video":
        max_frames = int(
            st.sidebar.number_input(
                "Max frames to process (0 = all)",
                min_value=0, value=0, step=10,
            )
        )
        save_video = st.sidebar.checkbox(
            "Generate downloadable output video", value=True
        )

    if source == "RTSP Stream":
        # URL input — no password masking here because the URL embeds creds.
        rtsp_url = st.sidebar.text_input(
            "RTSP URL",
            value="rtsp://",
            placeholder="rtsp://user:pass@192.168.1.50:554/stream1",
            help="Example: rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101",
        )
        # Live streams have no end — force a stop condition.
        rtsp_stop_after = int(
            st.sidebar.number_input(
                "Stop after N frames (0 = run until Stop)",
                min_value=0, value=300, step=30,
                help="Prevents runaway loops on a live camera. 0 = run indefinitely.",
            )
        )
        save_video = st.sidebar.checkbox(
            "Generate downloadable output video", value=True
        )

    return {
        "model_path": model_path,
        "conf": conf,
        "iou": iou,
        "source": source,
        "enable_tracking": enable_tracking,
        "tracker_type": tracker_type,
        "skip": skip,
        "max_frames": max_frames,
        "save_video": save_video,
        "rtsp_url": rtsp_url,
        "rtsp_stop_after": rtsp_stop_after,
    }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def uploaded_file_to_bgr(uploaded) -> np.ndarray:
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """
    Open an RTSP (or other network) stream.

    RTSP URLs can hang for a long time if the host is unreachable, so we
    set a connection/read timeout via the FFMPEG_OPTIONS environment hint
    where possible (depends on OpenCV build).
    """
    if not url or url.strip() == "" or url.strip() == "rtsp://":
        raise ValueError("RTSP URL is empty")

    # Optional timeout hint for FFmpeg-based backends (ignored on some builds).
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000",  # 5 s in microseconds
    )

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError(f"Could not open RTSP stream: {url}")
    return cap


# ---------------------------------------------------------------------------
# Core: process any cv2.VideoCapture — used by both file and RTSP modes.
# ---------------------------------------------------------------------------
def _run_capture_loop(
    detector: ObjectDetector,
    cap: cv2.VideoCapture,
    skip: int,
    max_frames: int,
    save_video: bool,
    class_filter: List[str],
    enable_tracking: bool,
    is_live: bool,
    stop_event_holder: dict,
) -> tuple[str | None, list[dict], dict]:
    """
    Shared processing loop for video files and RTSP streams.

    `is_live=True` disables the progress bar (no known total) and shows
    a plain "frames processed" counter.
    `stop_event_holder` is a mutable dict; setting ["stop"]=True stops the loop.
    """
    fps, width, height, total = video_properties(cap)

    # --- Verify actual decoded dimensions -------------------------------
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read any frames from source")
    h_actual, w_actual = first_frame.shape[:2]
    if (w_actual, h_actual) != (width, height):
        width, height = w_actual, h_actual

    # Rewind only for files. RTSP streams don't support seeking.
    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if max_frames and total:
            total = min(total, max_frames)
    else:
        # Live stream: pretend "total" is unknown.
        total = 0

    # --- UI placeholders -------------------------------------------------
    # For live streams we hide the progress bar; the counter conveys progress.
    progress = None
    if not is_live:
        progress = st.progress(0.0)
    status = st.empty()

    # A Stop button becomes visible while the loop runs. Clicking it can't
    # interrupt a running script mid-loop, so we check a session_state flag
    # on the *next* rerun (see note below).
    stop_btn = st.empty()
    if stop_btn.button("⏹ Stop (takes effect next rerun)", key="stop_btn"):
        stop_event_holder["stop"] = True

    # --- Output writer ---------------------------------------------------
    writer = None
    out_path = None
    if save_video and width > 0 and height > 0:
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            writer = make_writer(
                out_path,
                fps / max(skip, 1) if fps > 0 else 10.0,
                (width, height),
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not create output video writer: {exc}")
            writer = None
            out_path = None

    # --- Accumulators ----------------------------------------------------
    aggregated: list[dict] = []
    processed = 0
    read_idx = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # For live streams this usually means the stream dropped.
                break

            if skip > 1 and read_idx % skip != 0:
                read_idx += 1
                continue
            read_idx += 1

            # Detect or track.
            if enable_tracking:
                detections: List[Detection] = detector.track(
                    frame, class_filter or None
                )
            else:
                detections = detector.detect(frame, class_filter or None)

            annotated = draw_detections(frame, detections)

            if writer is not None:
                writer.write(annotated)

            for d in detections:
                row = d.as_dict()
                row["frame"] = processed
                aggregated.append(row)

            processed += 1

            # --- UI update ------------------------------------------
            if progress is not None and total and total > 0:
                progress.progress(min(processed / total, 1.0))
            elapsed = max(time.time() - start, 1e-6)
            cur_fps = processed / elapsed
            status.markdown(
                f"**Frame** {processed}"
                + (f" / {total}" if total else "")
                + f" · **FPS** {cur_fps:.1f} · **Elapsed** {elapsed:.1f}s"
            )

            # --- Stop conditions ------------------------------------
            if max_frames and processed >= max_frames:
                break
            if stop_event_holder.get("stop"):
                break

        if progress is not None:
            progress.progress(1.0)
        status.success(f"✅ Done — processed {processed} frame(s).")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = {"frames": processed, "total_objects": len(aggregated)}
    return out_path, aggregated, summary


# ---------------------------------------------------------------------------
# Video file mode.
# ---------------------------------------------------------------------------
def process_video(
    detector: ObjectDetector,
    video_path: str,
    skip: int,
    max_frames: int,
    save_video: bool,
    class_filter: List[str],
    enable_tracking: bool,
) -> tuple[str | None, list[dict], dict]:
    """Process a local video file."""
    cap = open_video(video_path)
    return _run_capture_loop(
        detector=detector,
        cap=cap,
        skip=skip,
        max_frames=max_frames,
        save_video=save_video,
        class_filter=class_filter,
        enable_tracking=enable_tracking,
        is_live=False,
        stop_event_holder={"stop": False},
    )


# ---------------------------------------------------------------------------
# RTSP stream mode.
# ---------------------------------------------------------------------------
def process_rtsp(
    detector: ObjectDetector,
    rtsp_url: str,
    skip: int,
    stop_after: int,
    save_video: bool,
    class_filter: List[str],
    enable_tracking: bool,
) -> tuple[str | None, list[dict], dict]:
    """Process a live RTSP stream."""
    cap = open_rtsp_stream(rtsp_url)

    # Persistent stop flag across reruns (see "Stop button caveat" below).
    if "rtsp_stop_flag" not in st.session_state:
        st.session_state.rtsp_stop_flag = False
    holder = {"stop": st.session_state.rtsp_stop_flag}

    out_path, rows, summary = _run_capture_loop(
        detector=detector,
        cap=cap,
        skip=skip,
        max_frames=stop_after,
        save_video=save_video,
        class_filter=class_filter,
        enable_tracking=enable_tracking,
        is_live=True,
        stop_event_holder=holder,
    )

    # Persist stop flag for the next rerun, then reset for next session.
    st.session_state.rtsp_stop_flag = holder.get("stop", False)
    return out_path, rows, summary


# ---------------------------------------------------------------------------
# Image mode.
# ---------------------------------------------------------------------------
def process_image(detector: ObjectDetector, class_filter: List[str]) -> None:
    uploaded = st.file_uploader("Choose an image", type=IMAGE_EXT)
    if uploaded is None:
        st.info("⬆️ Upload an image to begin.")
        return

    try:
        frame_bgr = uploaded_file_to_bgr(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read image: {exc}")
        return

    detections = detector.detect(frame_bgr, class_filter or None)
    annotated_bgr = draw_detections(frame_bgr, detections)

    st.subheader("Annotated image")
    st.image(bgr_to_rgb(annotated_bgr), channels="RGB", use_container_width=True)


# ---------------------------------------------------------------------------
# Webcam mode.
# ---------------------------------------------------------------------------
def process_webcam(detector: ObjectDetector, class_filter: List[str]) -> None:
    st.info("📷 Click **Take Photo** below to capture a frame from your webcam.")
    photo = st.camera_input("Webcam", key="webcam_photo")
    if photo is None:
        st.caption("No photo captured yet. Click **Take Photo** above.")
        return

    try:
        frame_bgr = uploaded_file_to_bgr(photo)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read webcam photo: {exc}")
        return

    detections = detector.detect(frame_bgr, class_filter or None)
    annotated_bgr = draw_detections(frame_bgr, detections)

    st.subheader("Annotated photo")
    st.image(bgr_to_rgb(annotated_bgr), channels="RGB", use_container_width=True)


# ---------------------------------------------------------------------------
# Downloads section — shared between video and RTSP modes.
# ---------------------------------------------------------------------------
def _render_downloads(out_path: str | None, rows: list[dict]) -> None:
    st.subheader("Downloads")

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇️ Detections as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="detections.csv",
            mime="text/csv",
        )
        col_b.download_button(
            "⬇️ Detections as JSON",
            df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="detections.json",
            mime="application/json",
        )

    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Annotated video (MP4)",
                f.read(),
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )


# ---------------------------------------------------------------------------
# Video file UI.
# ---------------------------------------------------------------------------
def process_video_ui(
    detector: ObjectDetector,
    cfg: dict,
    class_filter: List[str],
) -> None:
    uploaded = st.file_uploader("Choose a video", type=VIDEO_EXT)
    if uploaded is None:
        st.info("⬆️ Upload a video to begin.")
        return

    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save uploaded video: {exc}")
        return

    st.video(tmp_path)

    if not st.button("▶️ Start detection", type="primary"):
        return

    try:
        out_path, rows, summary = process_video(
            detector=detector,
            video_path=tmp_path,
            skip=int(cfg["skip"]),
            max_frames=int(cfg["max_frames"]),
            save_video=bool(cfg["save_video"]),
            class_filter=class_filter,
            enable_tracking=bool(cfg["enable_tracking"]),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Processing failed: {exc}")
        return

    _render_downloads(out_path, rows)


# ---------------------------------------------------------------------------
# RTSP stream UI (NEW).
# ---------------------------------------------------------------------------
def process_rtsp_ui(
    detector: ObjectDetector,
    cfg: dict,
    class_filter: List[str],
) -> None:
    url = (cfg.get("rtsp_url") or "").strip()

    if not url or url == "rtsp://":
        st.info(
            "📡 Paste an RTSP URL in the sidebar to begin. "
            "Example: `rtsp://user:pass@192.168.1.50:554/stream1`"
        )
        return

    st.caption(f"Stream: `{url}`")

    # Start button — only runs the loop when clicked.
    if not st.button("▶️ Start stream processing", type="primary"):
        # Show any previous results (if any) — persisted across reruns.
        prev = st.session_state.get("rtsp_last_result")
        if prev:
            out_path, rows, summary = prev
            st.success(f"✅ Last run — processed {summary['frames']} frame(s).")
            _render_downloads(out_path, rows)
        return

    # Reset stop flag when starting fresh.
    st.session_state.rtsp_stop_flag = False

    try:
        out_path, rows, summary = process_rtsp(
            detector=detector,
            rtsp_url=url,
            skip=int(cfg["skip"]),
            stop_after=int(cfg["rtsp_stop_after"]),
            save_video=bool(cfg["save_video"]),
            class_filter=class_filter,
            enable_tracking=bool(cfg["enable_tracking"]),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Stream processing failed: {exc}")
        return

    # Persist results for the next rerun.
    st.session_state.rtsp_last_result = (out_path, rows, summary)

    _render_downloads(out_path, rows)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🎯 Object Detection + Tracking")
    st.caption("YOLOv8 + Streamlit — supports Webcam, Video, Image, and RTSP stream.")

    cfg = render_sidebar()

    try:
        detector = load_detector(
            cfg["model_path"],
            cfg["conf"],
            cfg["iou"],
            cfg["tracker_type"],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model '{cfg['model_path']}': {exc}")
        st.stop()

    class_filter: List[str] = st.sidebar.multiselect(
        "Only detect these classes (empty = all)",
        options=sorted(detector.class_names),
        default=[],
        key="class_filter_widget",
    )

    if cfg["source"] == "Webcam":
        process_webcam(detector, class_filter)
    elif cfg["source"] == "Upload Video":
        process_video_ui(detector, cfg, class_filter)
    elif cfg["source"] == "RTSP Stream":
        process_rtsp_ui(detector, cfg, class_filter)
    else:
        process_image(detector, class_filter)


if __name__ == "__main__":
    main()