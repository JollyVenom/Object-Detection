import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("YOLO Object Detection App")

# ---------------- LOAD MODELS ----------------
MODEL_WEBCAM = YOLO("yolov8s-oiv7.pt")   # accuracy
MODEL_VIDEO = YOLO("yolov8n.pt")         # fast
MODEL_IMAGE = YOLO("yolov8m.pt")         # high accuracy

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


# ---------------- VIDEO MODE (STABLE) ----------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("Start Detection"):

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            frame_skip = 2
            display_skip = 2
            count = 0
            last_results = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 360))
                count += 1

                # Detection
                if count % frame_skip == 0:
                    last_results = MODEL_VIDEO(frame, imgsz=320)[0]

                if last_results is not None:
                    frame = draw_boxes(frame, last_results)

                # Reduce UI updates
                if count % display_skip == 0:
                    stframe.image(frame, channels="BGR", use_container_width=True)

                time.sleep(0.01)  # 🔥 prevents crash

            cap.release()


# ---------------- WEBCAM MODE (FIXED) ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection")

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = MODEL_WEBCAM(img, imgsz=416)[0]
            img = draw_boxes(img, results)

            return img

    webrtc_streamer(
        key="yolo-live",
        rtc_configuration=RTC_CONFIGURATION,  # 🔥 FIXED
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={
            "video": True,
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
