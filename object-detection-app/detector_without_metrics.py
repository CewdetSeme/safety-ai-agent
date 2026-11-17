"""Streamlit UI for YOLOv8 object detection + tracking — with RTSP support.

This app supports four input modes:
  • Webcam      — single-frame capture via the browser
  • Upload Video — local MP4/AVI/etc. processed frame-by-frame
  • Upload Image — single photo
  • RTSP Stream — live IP camera / NVR stream
And optional object tracking (ByteTrack / BoT-SORT) for video and RTSP modes.
"""

# ---------------------------------------------------------------------------
# Lazy type hints — evaluated only by type checkers, not at runtime.
# ---------------------------------------------------------------------------
from __future__ import annotations

# --- Standard library -------------------------------------------------------
import os            # path helpers: splitext, path.exists, path.getsize
import tempfile      # create unique temp files for uploads and output MP4
import time          # elapsed time and FPS measurement
from typing import List   # type hint for list of class-name strings

# --- Third-party ------------------------------------------------------------
import cv2                         # OpenCV: video capture/encode, drawing
import numpy as np                 # array type for frames (H, W, 3)
import streamlit as st             # the web UI framework
from PIL import Image              # robust image decoding (EXIF, WebP, RGBA, etc.)

# --- Local imports ----------------------------------------------------------
# Detection = dataclass holding one object's class/conf/bbox/track_id
# ObjectDetector = wrapper around YOLOv8 (detect + track methods)
from detector import Detection, ObjectDetector

# Helpers from utils.py:
#   bgr_to_rgb      = BGR→RGB conversion for Streamlit display
#   draw_detections = draw boxes + labels; returns a NEW annotated frame
#   make_writer     = create cv2.VideoWriter for MP4 output
#   open_video      = open cv2.VideoCapture with validation
#   video_properties = read fps/width/height/frame_count from a capture
from utils import (
    bgr_to_rgb,
    draw_detections,
    make_writer,
    open_video,
    video_properties,
)

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",   # browser tab title
    page_icon="🎯",                         # favicon
    layout="wide",                          # use full window width
)

# Weights shown in the model dropdown (nano → large = faster → more accurate).
MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

# Tracker options. ByteTrack = fast, ideal for fixed cameras.
# BoT-SORT = better when the camera itself shakes / moves.
TRACKER_OPTIONS = ["bytetrack.yaml", "botsort.yaml"]

# Accepted extensions for the file uploaders and camera widget.
IMAGE_EXT = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_EXT = ["mp4", "avi", "mov", "mkv", "webm"]


# ---------------------------------------------------------------------------
# Cached model loader — one ObjectDetector instance per (model, conf, iou, tracker).
# Without caching, every widget drag would reload YOLO (1–5 s).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(
    model_path: str,
    conf: float,
    iou: float,
    tracker: str,
) -> ObjectDetector:
    # Ultralytics downloads weights on first call, then caches them on disk.
    return ObjectDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Sidebar — renders all controls and returns their current values as a dict.
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")

    # --- Model settings -------------------------------------------------
    st.sidebar.subheader("Model")
    # Dropdown for weights; index=0 selects yolov8n.pt.
    model_path = st.sidebar.selectbox("Weights", MODEL_OPTIONS, index=0)
    # Confidence threshold: 0–1, default 0.4, step 0.05.
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
    # IoU threshold for NMS: overlap above this is suppressed.
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

    # --- Source choice --------------------------------------------------
    st.sidebar.subheader("Input source")
    # Radio: Webcam (default), Upload Video, Upload Image, RTSP Stream.
    source = st.sidebar.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image", "RTSP Stream"],
        index=0,
    )

    # --- Tracking -------------------------------------------------------
    st.sidebar.subheader("Tracking")
    # Toggle tracking. Only video + RTSP benefit; image/webcam are single-frame.
    enable_tracking = st.sidebar.checkbox(
        "Enable object tracking (video / RTSP only)",
        value=True,
        help="Assigns persistent IDs across frames. Only used for videos and streams.",
    )
    # Tracker choice; disabled when tracking is off.
    tracker_type = st.sidebar.selectbox(
        "Tracker algorithm",
        TRACKER_OPTIONS,
        index=0,
        disabled=not enable_tracking,
    )

    # --- Video / RTSP shared controls ----------------------------------
    # Defaults used when source is Webcam or Upload Image.
    skip = 1
    max_frames = 0
    save_video = True
    rtsp_url = ""
    rtsp_stop_after = 0

    # Frame-skip applies to both video files and RTSP streams.
    if source in ("Upload Video", "RTSP Stream"):
        skip = st.sidebar.slider(
            "Process every Nth frame", 1, 10, 1,
            help="Higher = faster but choppier output.",
        )

    # --- Video-file-only controls --------------------------------------
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

    # --- RTSP-only controls --------------------------------------------
    if source == "RTSP Stream":
        # URL input; credentials are often embedded in the URL itself.
        rtsp_url = st.sidebar.text_input(
            "RTSP URL",
            value="rtsp://",
            placeholder="rtsp://user:pass@192.168.1.50:554/stream1",
            help="Example: rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101",
        )
        # A live stream never ends by itself — force a stop condition.
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

    # Bundle all widget values into one dict for easy passing around.
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
    """Decode bytes from file_uploader / camera_input into a BGR numpy array."""
    # Pillow decodes all common formats and normalizes mode (RGBA→RGB, etc.).
    image = Image.open(uploaded).convert("RGB")
    # RGB → BGR because the OpenCV pipeline is BGR-native.
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """
    Open an RTSP (or other network) stream with a socket timeout so we don't
    hang forever when the camera host is unreachable.
    """
    # Reject empty / placeholder URLs early with a clear message.
    if not url or url.strip() == "" or url.strip() == "rtsp://":
        raise ValueError("RTSP URL is empty")

    # FFmpeg hints: prefer TCP (more reliable than UDP) and 5-s timeout.
    # `setdefault` keeps a user-provided value if they already set it.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000",  # 5 s in microseconds
    )

    # CAP_FFMPEG forces the FFmpeg backend (the one that speaks RTSP well).
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

    `is_live=True` disables the progress bar (live streams have no known total)
    and skips the "rewind" step (RTSP can't seek).
    `stop_event_holder` is a mutable dict; setting ["stop"]=True ends the loop.
    """
    # Read fps / dims / total frames from the container.
    fps, width, height, total = video_properties(cap)

    # --- Verify actual decoded dimensions -------------------------------
    # Container metadata can lie (rotation flags, unusual codecs). Trust the
    # decoded frame instead.
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read any frames from source")
    h_actual, w_actual = first_frame.shape[:2]
    if (w_actual, h_actual) != (width, height):
        width, height = w_actual, h_actual

    # Rewind only for files — RTSP streams don't support seeking.
    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if max_frames and total:
            total = min(total, max_frames)
    else:
        # Live stream: unknown end → disable percent-based progress.
        total = 0

    # --- UI placeholders -------------------------------------------------
    # Progress bar only for files (live streams use just the counter).
    progress = None
    if not is_live:
        progress = st.progress(0.0)
    # Status line: "Frame N / total · FPS · Elapsed".
    status = st.empty()

    # Stop button — visible while the loop runs. Clicking it can't interrupt
    # the current loop; it flips a flag checked on the NEXT rerun.
    stop_btn = st.empty()
    if stop_btn.button("⏹ Stop (takes effect next rerun)", key="stop_btn"):
        stop_event_holder["stop"] = True

    # --- Output writer (optional MP4) -----------------------------------
    writer = None
    out_path = None
    if save_video and width > 0 and height > 0:
        # Create a unique temp file; OpenCV will write into this path.
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            writer = make_writer(
                out_path,
                # Output fps must match how many frames we actually write.
                fps / max(skip, 1) if fps > 0 else 10.0,
                (width, height),
            )
        except Exception as exc:  # noqa: BLE001 — any writer failure is non-fatal
            st.warning(f"Could not create output video writer: {exc}")
            writer = None
            out_path = None

    # --- Accumulators ----------------------------------------------------
    aggregated: list[dict] = []   # one dict per detection across all frames
    processed = 0                 # frames that went through inference
    read_idx = 0                  # raw frames read (for skip math)
    start = time.time()           # wall-clock start for elapsed / FPS

    # Placeholder used for the periodic preview thumbnail (every 10 frames).
    annotated_placeholder = st.empty()

    try:
        while True:
            # Read next frame. For RTSP, `ok=False` usually means stream drop.
            ok, frame = cap.read()
            if not ok:
                break

            # Frame skipping: discard non-multiples of `skip` before inference.
            if skip > 1 and read_idx % skip != 0:
                read_idx += 1
                continue
            read_idx += 1

            # Detect or track. `class_filter or None` turns [] into "no filter".
            if enable_tracking:
                detections: List[Detection] = detector.track(
                    frame, class_filter or None
                )
            else:
                detections = detector.detect(frame, class_filter or None)


            print(detections)
            
            # Draw boxes + labels on a COPY.
            annotated = draw_detections(frame, detections)

            # Update the preview thumbnail only every 10 processed frames —
            # pushing every frame through Streamlit would slow the loop.
            if processed % 10 == 0:
                annotated_placeholder.image(
                    bgr_to_rgb(annotated),
                    channels="RGB",
                )

            # Write annotated frame into the output MP4 (if enabled).
            if writer is not None:
                writer.write(annotated)

            # Store one dict per detection for later CSV / JSON export.
            for d in detections:
                row = d.as_dict()
                row["frame"] = processed
                aggregated.append(row)

            # Bump processed frame count.
            processed += 1

            # --- UI update ------------------------------------------
            # Progress bar (files only). Clamp to 1.0 in case of overcount.
            if progress is not None and total and total > 0:
                progress.progress(min(processed / total, 1.0))
            # Compute elapsed & FPS; guard against divide-by-zero.
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

        # Clean finish — fill progress bar (files) and mark success.
        if progress is not None:
            progress.progress(1.0)
        status.success(f"✅ Done — processed {processed} frame(s).")

    finally:
        # Always release resources, even if an exception was raised mid-loop.
        cap.release()
        if writer is not None:
            writer.release()   # flushes MP4 header — mandatory for valid output

    # Summary dict returned to caller (used for the "last run" message).
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
    """Process a local video file (thin wrapper around the shared loop)."""
    cap = open_video(video_path)   # raises ValueError if unreadable
    return _run_capture_loop(
        detector=detector,
        cap=cap,
        skip=skip,
        max_frames=max_frames,
        save_video=save_video,
        class_filter=class_filter,
        enable_tracking=enable_tracking,
        is_live=False,                        # files → progress bar + rewind
        stop_event_holder={"stop": False},    # no external stop flag
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

    # Persist stop flag in session_state so a Stop click survives reruns.
    if "rtsp_stop_flag" not in st.session_state:
        st.session_state.rtsp_stop_flag = False
    holder = {"stop": st.session_state.rtsp_stop_flag}

    out_path, rows, summary = _run_capture_loop(
        detector=detector,
        cap=cap,
        skip=skip,
        max_frames=stop_after,   # treat "stop_after" as the frame cap
        save_video=save_video,
        class_filter=class_filter,
        enable_tracking=enable_tracking,
        is_live=True,            # no progress bar, no rewind
        stop_event_holder=holder,
    )

    # Persist updated stop flag for the next rerun.
    st.session_state.rtsp_stop_flag = holder.get("stop", False)
    return out_path, rows, summary


# ---------------------------------------------------------------------------
# Image mode — single upload, single detection, display.
# ---------------------------------------------------------------------------
def process_image(detector: ObjectDetector, class_filter: List[str]) -> None:
    uploaded = st.file_uploader("Choose an image", type=IMAGE_EXT)
    # Empty state.
    if uploaded is None:
        st.info("⬆️ Upload an image to begin.")
        return

    # Decode the upload; catch corrupt files.
    try:
        frame_bgr = uploaded_file_to_bgr(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read image: {exc}")
        return

    # Detection-only (no tracking needed for a single frame).
    detections = detector.detect(frame_bgr, class_filter or None)
    annotated_bgr = draw_detections(frame_bgr, detections)

    # Display annotated result.
    st.subheader("Annotated image")
    st.image(bgr_to_rgb(annotated_bgr), channels="RGB", use_container_width=True)


# ---------------------------------------------------------------------------
# Webcam mode — single capture via browser camera (requires HTTPS/localhost).
# ---------------------------------------------------------------------------
def process_webcam(detector: ObjectDetector, class_filter: List[str]) -> None:
    st.info("📷 Click **Take Photo** below to capture a frame from your webcam.")

    # Camera widget returns an UploadedFile-like object per capture.
    photo = st.camera_input("Webcam", key="webcam_photo")
    if photo is None:
        st.caption("No photo captured yet. Click **Take Photo** above.")
        return

    # Decode the captured JPEG.
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

    # CSV / JSON — only if we have detection rows.
    if rows:
        import pandas as pd   # deferred import — pandas is heavy

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

    # MP4 — only if a valid output file exists and is non-empty.
    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Annotated video (MP4)",
                f.read(),                       # read whole file into memory
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

    # OpenCV needs a path — persist the upload to a temp file with the same ext.
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save uploaded video: {exc}")
        return

    # Preview the ORIGINAL (unannotated) uploaded video.
    st.video(tmp_path)

    # Only process on the click-run. `st.button` returns True on that single
    # rerun — False on every other rerun (slider drag, radio change, etc.).
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
# RTSP stream UI.
# ---------------------------------------------------------------------------
def process_rtsp_ui(
    detector: ObjectDetector,
    cfg: dict,
    class_filter: List[str],
) -> None:
    url = (cfg.get("rtsp_url") or "").strip()

    # Empty / placeholder URL → hint and exit.
    if not url or url == "rtsp://":
        st.info(
            "📡 Paste an RTSP URL in the sidebar to begin. "
            "Example: `rtsp://user:pass@192.168.1.50:554/stream1`"
        )
        return

    st.caption(f"Stream: `{url}`")

    # Only process on the click-run.
    if not st.button("▶️ Start stream processing", type="primary"):
        # Show the last run's results (persisted across reruns) if any.
        prev = st.session_state.get("rtsp_last_result")
        if prev:
            out_path, rows, summary = prev
            st.success(f"✅ Last run — processed {summary['frames']} frame(s).")
            _render_downloads(out_path, rows)
        return

    # Reset the stop flag for a fresh run.
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

    # Persist results so download buttons survive future reruns.
    st.session_state.rtsp_last_result = (out_path, rows, summary)

    _render_downloads(out_path, rows)


# ---------------------------------------------------------------------------
# Main — orchestrates sidebar, model, class filter, and source dispatch.
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🎯 Object Detection + Tracking")
    st.caption("YOLOv8 + Streamlit — supports Webcam, Video, Image, and RTSP stream.")

    # Sidebar values.
    cfg = render_sidebar()

    # Load model (cached). Stop on failure with a visible error.
    try:
        detector = load_detector(
            cfg["model_path"],
            cfg["conf"],
            cfg["iou"],
            cfg["tracker_type"],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model '{cfg['model_path']}': {exc}")
        st.stop()   # halt script cleanly

    # Class filter — needs the model first for the 80-class option list.
    class_filter: List[str] = st.sidebar.multiselect(
        "Only detect these classes (empty = all)",
        options=sorted(detector.class_names),
        default=[],
        key="class_filter_widget",   # stable key in session_state
    )

    # Dispatch to the selected mode.
    if cfg["source"] == "Webcam":
        process_webcam(detector, class_filter)
    elif cfg["source"] == "Upload Video":
        process_video_ui(detector, cfg, class_filter)
    elif cfg["source"] == "RTSP Stream":
        process_rtsp_ui(detector, cfg, class_filter)
    else:
        process_image(detector, class_filter)


# ---------------------------------------------------------------------------
# Entry point — runs only when executed as a script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()