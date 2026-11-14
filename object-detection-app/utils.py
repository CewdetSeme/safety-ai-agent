"""Drawing, video I/O, and frame-conversion helpers."""

from __future__ import annotations

import colorsys
import hashlib
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from detector import Detection

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def _color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Deterministic, visually distinct BGR color per class name."""
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


_COLOR_CACHE: Dict[str, Tuple[int, int, int]] = {}


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    if class_name not in _COLOR_CACHE:
        _COLOR_CACHE[class_name] = _color_for_class(class_name)
    return _COLOR_CACHE[class_name]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw boxes + labels on a copy of the frame (returns BGR image)."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = color_for_class(det.class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: prepend "#<id> " when tracking is active
        if det.track_id is not None:
            label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        else:
            label = f"{det.class_name} {det.confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            annotated,
            (x1, top),
            (x1 + tw + 6, top + th + baseline + 4),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated



def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB for display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and validate it."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    return cap

def video_properties(cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count) for a capture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total


def make_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 writer. `size` is (width, height)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {path}")
    return writer

