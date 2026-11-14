import cv2
from detector import ObjectDetector
from utils import bgr_to_rgb, draw_detections

detector = ObjectDetector("yolov8n.pt", conf_threshold=0.4)
frame = cv2.imread("sample.jpg")

dets = detector.detect(frame)
print(f"Detected {len(dets)} objects")
for d in dets:
    print(f"  {d.class_name:<12} {d.confidence:.2f}  {d.bbox}")

annotated_bgr = draw_detections(frame, dets)
cv2.imwrite("annotated_bgr.png", annotated_bgr)          # verify BGR output
cv2.imwrite("annotated_rgb.png", bgr_to_rgb(annotated_bgr))  # for Streamlit

# Verify original frame is unchanged
assert not (frame is annotated_bgr), "draw_detections should return a copy"
print("\n✅ Saved annotated_bgr.png and annotated_rgb.png")