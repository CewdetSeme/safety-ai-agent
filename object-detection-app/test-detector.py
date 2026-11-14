import cv2
from detector import ObjectDetector

detector = ObjectDetector("yolov8n.pt", conf_threshold=0.4)
print(f"Model loaded. {len(detector.class_names)} classes available.")

# Download a sample image the first time, or point at any local image
import urllib.request
sample = "https://ultralytics.com/images/bus.jpg"

urllib.request.urlretrieve(sample, "sample.jpg")

frame = cv2.imread("sample.jpg")
dets = detector.detect(frame)

print(f"\nFound {len(dets)} objects:")
for d in dets:
    print(f"  {d.class_name:<12} conf={d.confidence:.2f}  bbox={d.bbox}")


#python -m venv .venv
#source .venv/bin/activate source .venv/bin/activate      # Windows: .venv\Scripts\activate
#pip install -r requirements.txt
#streamlit run app.py
