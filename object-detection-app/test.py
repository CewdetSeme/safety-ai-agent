
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