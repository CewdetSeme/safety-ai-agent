
""""
import streamlit as st
import cv2

st.title("RTSP Camera Test")

rtsp_url = st.text_input(
    "RTSP URL",
    "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream"
)

if st.button("Start Camera"):

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        st.error("Could not open RTSP stream")
    else:
        st.success("RTSP stream connected")

        frame_placeholder = st.empty()

        while True:
            success, frame = cap.read()

            if not success:
                st.error("Could not read frame")
                break

            # OpenCV gives BGR, Streamlit expects RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()
"""
"""
import streamlit as st
import cv2
import time

st.title("Video Playback Test")

uploaded = st.file_uploader("Upload video", type=["mp4"])

if uploaded:

    with open("videoplayback.mp4", "wb") as f:
        f.write(uploaded.getbuffer())

    cap = cv2.VideoCapture("videoplayback.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_placeholder = st.empty()
    counter = st.empty()

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            frame,
            channels="RGB",
            output_format="JPEG"
        )

        counter.write(f"Frame: {frame_number}")

        time.sleep(1 / fps)

    cap.release()
"""

# test reader with detections with mock llm and console alert
"""
from mock_llm import MockSafetyLLM
from rtsp_reader import RTSPReader
from detector import ObjectDetector
from safety_monitor import run_safety_monitor

class ConsoleAlert:
    def send(self, subject, html, evidence_path, context):
        print(f"[ALERT] {subject} → evidence: {evidence_path}")

reader = RTSPReader("rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")
reader.start()
detector = ObjectDetector("yolov8n.pt")
run_safety_monitor(reader, detector, MockSafetyLLM(), ConsoleAlert(),
                   analyze_every_s=5, alert_cooldown_s=30)




"""
# test_llm.py — run with: python test_llm.py path/to/image.jpg
import base64
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
from llm_client import OpenAIVisionClient

with open(sys.argv[1], "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

client = OpenAIVisionClient()
verdict = client.analyze(b64)
print(json.dumps(verdict, indent=2))

