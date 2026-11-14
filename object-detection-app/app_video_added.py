"""Streamlit UI for YOLOv8 object detection + tracking."""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
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

    # --- Source choice --------------------------------------------------
    st.sidebar.subheader("Input source")
    source = st.sidebar.radio(
        "Source",
        ["Webcam", "Upload Video", "Upload Image"],
        index=0,
    )

    # --- Tracking -------------------------------------------------------
    st.sidebar.subheader("Tracking")
    enable_tracking = st.sidebar.checkbox(
        "Enable object tracking (video only)",
        value=True,
        help="Assigns persistent IDs across frames. Only used in video mode.",
    )
    tracker_type = st.sidebar.selectbox(
        "Tracker algorithm",
        TRACKER_OPTIONS,
        index=0,
        disabled=not enable_tracking,
        help="ByteTrack: fast, great for fixed cameras. BoT-SORT: better for moving cameras.",
    )

    # --- Video-only controls --------------------------------------------
    skip = 1
    max_frames = 0
    save_video = True

    if source == "Upload Video":
        skip = st.sidebar.slider(
            "Process every Nth frame", 1, 10, 1,
            help="Higher = faster but choppier output.",
        )
        max_frames = int(
            st.sidebar.number_input(
                "Max frames to process (0 = all)",
                min_value=0, value=0, step=10,
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
    }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def uploaded_file_to_bgr(uploaded) -> np.ndarray:
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Video processing.
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
    """
    Run detection/tracking over the video, write annotated MP4, return stats.
    """
    cap = open_video(video_path)
    fps, width, height, total = video_properties(cap)

    # Verify actual decoded dims (rotation / codec safety).
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read any frames from video")
    h_actual, w_actual = first_frame.shape[:2]
    if (w_actual, h_actual) != (width, height):
        width, height = w_actual, h_actual
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if max_frames and total:
        total = min(total, max_frames)

    # Reset tracker state for a fresh video.
    if enable_tracking:
        detector.reset_tracker()

    # UI placeholders.
    progress = st.progress(0.0)
    status = st.empty()
    metrics_ph = st.empty()
    table_ph = st.empty()

    # Writer.
    writer = None
    out_path = None
    if width > 0 and height > 0:
        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            writer = make_writer(out_path, fps / max(skip, 1), (width, height))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not create output video writer: {exc}")
            writer = None
            out_path = None

    # Accumulators.
    aggregated: list[dict] = []
    class_counter: Counter = Counter()
    unique_track_ids: set[int] = set()
    conf_sum = 0.0
    conf_count = 0
    processed = 0
    read_idx = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
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
                class_counter[d.class_name] += 1
                conf_sum += d.confidence
                conf_count += 1
                if d.track_id is not None:
                    unique_track_ids.add(d.track_id)

            processed += 1

            # Cheap UI updates.
            if total and total > 0:
                progress.progress(min(processed / total, 1.0))
            elapsed = max(time.time() - start, 1e-6)
            cur_fps = processed / elapsed
            status.markdown(
                f"**Frame** {processed}"
                + (f" / {total}" if total else "")
                + f" · **FPS** {cur_fps:.1f} · **Elapsed** {elapsed:.1f}s"
            )

            if processed % 5 == 0:
                avg_conf = conf_sum / conf_count if conf_count else 0.0
                with metrics_ph.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Objects (total)", sum(class_counter.values()))
                    c2.metric("Classes seen", len(class_counter))
                    c3.metric("Avg confidence", f"{avg_conf:.2f}")
                if class_counter:
                    table_ph.dataframe(
                        [{"class": k, "count": v}
                         for k, v in class_counter.most_common(10)],
                        use_container_width=True,
                        hide_index=True,
                    )

            if max_frames and processed >= max_frames:
                break

        progress.progress(1.0)
        status.success(f"✅ Done — processed {processed} frame(s).")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = {
        "frames": processed,
        "total_objects": sum(class_counter.values()),
        "unique_objects": len(unique_track_ids) if enable_tracking else None,
        "avg_confidence": (conf_sum / conf_count) if conf_count else 0.0,
        "per_class": dict(class_counter),
    }
    return out_path, aggregated, summary


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

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Annotated image")
        st.image(
            bgr_to_rgb(annotated_bgr),
            channels="RGB",
            use_container_width=True,
        )
    with col_right:
        st.subheader("Summary")
        st.metric("Objects detected", len(detections))
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            st.metric("Avg confidence", f"{avg_conf:.2f}")
            st.subheader("Detections")
            st.dataframe(
                [d.as_dict() for d in detections],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No objects detected. Try lowering the confidence threshold.")


# ---------------------------------------------------------------------------
# Webcam mode.
# ---------------------------------------------------------------------------
def process_webcam(detector: ObjectDetector, class_filter: List[str]) -> None:
    st.info(
        "📷 Click **Take Photo** below to capture a frame from your webcam."
    )
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

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Annotated photo")
        st.image(
            bgr_to_rgb(annotated_bgr),
            channels="RGB",
            use_container_width=True,
        )
    with col_right:
        st.subheader("Summary")
        st.metric("Objects detected", len(detections))
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            st.metric("Avg confidence", f"{avg_conf:.2f}")
            st.subheader("Detections")
            st.dataframe(
                [d.as_dict() for d in detections],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No objects detected. Try lowering the confidence threshold.")


# ---------------------------------------------------------------------------
# Video mode UI.
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

    # Smooth annotated playback.
    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        st.subheader("Annotated video")
        st.video(out_path)

    # Summary.
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", summary["frames"])
    c2.metric("Total detections", summary["total_objects"])
    c3.metric(
        "Unique objects",
        summary["unique_objects"] if summary["unique_objects"] is not None else "—",
    )
    c4.metric("Avg confidence", f"{summary['avg_confidence']:.2f}")

    if summary["per_class"]:
        st.dataframe(
            [
                {"class": k, "count": v}
                for k, v in sorted(
                    summary["per_class"].items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            ],
            use_container_width=True,
            hide_index=True,
        )

    # Downloads.
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
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🎯 Object Detection + Tracking")
    st.caption("YOLOv8 + Streamlit — detect objects in webcam, video, or image.")

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
    else:
        process_image(detector, class_filter)


if __name__ == "__main__":
    main()