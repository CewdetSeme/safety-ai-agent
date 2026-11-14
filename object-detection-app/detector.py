"""YOLOv8-based object detector with optional ByteTrack/BotSORT tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected object in a frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2) in pixels
    track_id: Optional[int] = None         # None when tracking is disabled

    def as_dict(self) -> dict:
        d = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "x1": self.bbox[0],
            "y1": self.bbox[1],
            "x2": self.bbox[2],
            "y2": self.bbox[3],
        }
        if self.track_id is not None:
            d["track_id"] = self.track_id
        return d


class ObjectDetector:
    """Wraps a YOLOv8 model for detection and tracking."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.model = YOLO(model_path)

    @property
    def class_names(self) -> List[str]:
        names = self.model.names
        if isinstance(names, dict):
            return [names[i] for i in range(len(names))]
        return list(names)

    def _results_to_detections(
        self,
        result,
        class_filter: Sequence[str] | None,
        with_track_ids: bool,
    ) -> List[Detection]:
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        allowed = set(class_filter) if class_filter else None
        detections: List[Detection] = []

        track_ids = None
        if with_track_ids and boxes.id is not None:
            track_ids = boxes.id.int().tolist()

        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            cls_name = names[cls_id]
            if allowed is not None and cls_name not in allowed:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_ids[i] if track_ids else None,
                )
            )
        return detections

    def detect(
        self,
        frame: np.ndarray,
        class_filter: Sequence[str] | None = None,
    ) -> List[Detection]:
        """Detection-only (for images / single-frame webcam)."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return []
        return self._results_to_detections(
            results[0], class_filter, with_track_ids=False
        )

    def track(
        self,
        frame: np.ndarray,
        class_filter: Sequence[str] | None = None,
    ) -> List[Detection]:
        """Detection + tracking (for videos). Must be called sequentially per frame."""
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,               # keeps tracker state between calls
            tracker=self.tracker,
            verbose=False,
        )
        if not results:
            return []
        return self._results_to_detections(
            results[0], class_filter, with_track_ids=True
        )

    def reset_tracker(self) -> None:
        """
        Reset tracker state so a new video starts with fresh IDs.

        Ultralytics stores per-task trackers on `model.predictor.trackers`.
        Clearing it forces a fresh tracker on the next `.track()` call.
        """
        try:
            if self.model.predictor is not None:
                self.model.predictor.trackers = []
        except AttributeError:
            # Older / newer ultralytics versions may not expose this attribute.
            # In that case, tracking simply continues; not fatal.
            pass