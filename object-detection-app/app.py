"""Streamlit UI for YOLOv8 object detection — Step 4: image upload."""

from __future__ import annotations

from typing import List

import cv2

import numpy as np

import streamlit as st
from PIL import Image

from detector import ObjectDetector
from utils import bgr_to_rgb, draw_detections

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
)

MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

# ---------------------------------------------------------------------------
# Cached model loader
# ------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(model_path: str, conf: float, iou: float) -> ObjectDetector:
    """Load (and cache) a YOLO model per (path, conf, iou) triple."""
    return ObjectDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
    )


# ---------------------------------------------------------------------------
# Sidebar
# --------------------
def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")

    st.sidebar.subheader("Model")
    model_path = st.sidebar.selectbox("Weights", MODEL_OPTIONS, index=0)
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

    return {"model_path": model_path, "conf": conf, "iou": iou}


# ---------------------------------------------------------------------------
# Uploaded-file → BGR numpy frame
# ------------------------------
def uploaded_file_to_bgr(uploaded) -> np.ndarray:
    """Read Streamlit's UploadedFile bytes as a BGR numpy image."""
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# Main
# --------------------------
def main() -> None:
    st.title("🎯 Object Detection — Image Mode")
    st.caption("Upload an image to detect objects with YOLOv8.")

    cfg = render_sidebar()

    # --- Load model (cached) ------------
    try:
        detector = load_detector(cfg["model_path"], cfg["conf"], cfg["iou"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model '{cfg['model_path']}': {exc}")
        st.stop()
    
    # --- Class filter (needs model to know available classes) -----
    class_filter: List[str] = st.sidebar.multiselect(
        "Only detect these classes (empty = all)",
        options=sorted(detector.class_names),
        default=[],
        key="class_filter_widget",
    )

    # --- Upload ----------------------------------------------------------
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is None:
        st.info("⬆️ Upload an image to begin.")
        return

    # --- Decode + detect -------------------
    try:
        frame_bgr = uploaded_file_to_bgr(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read image: {exc}")
        return
    detections = detector.detect(frame_bgr, class_filter or None)
    annotated_bgr = draw_detections(frame_bgr, detections)

    # --- Display -----------------
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
        if detections:
            st.dataframe(
                [d.as_dict() for d in detections],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No objects detected. Try lowering the confidence threshold.")

if __name__ == "__main__":
    main()