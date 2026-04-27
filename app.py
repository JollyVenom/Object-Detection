import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("YOLO Object Detection App")

# ---------------- LOAD MODELS ----------------
MODEL_WEBCAM = YOLO("yolov8s-oiv7.pt")   # accurate
MODEL_VIDEO = YOLO("yolov8n.pt")         # fast
MODEL_IMAGE = YOLO("yolov8m.pt")         # accurate

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ---------------- STYLE ----------------
BOX_COLOR = (0, 0, 255)
THICKNESS = 1
FONT_SCALE = 0.5

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BOX_COLOR, 1)
    return frame


# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        results = MODEL_IMAGE(frame, imgsz=640)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, caption="Detection Result", use_container_width=True)


# ---------------- VIDEO MODE (SMOOTH PLAYBACK) ----------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Save video temporarily
        if "video_path" not in st.session_state:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.session_state.video_path = tfile.name

        # Initialize capture once
        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
            st.session_state.frame_count = 0

        cap = st.session_state.cap
        stframe = st.empty()

        # Controls
        col1, col2 = st.columns(2)
        play = col1.button("▶ Play")
        stop = col2.button("⏹ Stop")

        if stop:
            cap.release()
            st.session_state.clear()
            st.warning("Video stopped")
            st.stop()

        if play:
            ret, frame = cap.read()

            if not ret:
                st.success("Video finished")
                cap.release()
                st.session_state.clear()
                st.stop()

            # -------- OPTIMIZATION --------
            frame = cv2.resize(frame, (480, 270))
            st.session_state.frame_count += 1

            # Run detection every 5 frames
            if st.session_state.frame_count % 5 == 0:
                results = MODEL_VIDEO(frame, imgsz=320)[0]
                frame = draw_boxes(frame, results)

            stframe.image(frame, channels="BGR", use_container_width=True)

            time.sleep(0.03)  # control speed
            st.rerun()


# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = MODEL_WEBCAM(img, imgsz=512)[0]
            img = draw_boxes(img, results)
            return img

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 25},
            },
            "audio": False,
        },
    )

    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
