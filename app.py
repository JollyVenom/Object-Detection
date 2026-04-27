import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("YOLO Object Detection (Smooth Streaming)")

# ---------------- LOAD MODELS ----------------
MODEL_WEBCAM = YOLO("yolov8s-oiv7.pt")
MODEL_VIDEO = YOLO("yolov8n.pt")
MODEL_IMAGE = YOLO("yolov8m.pt")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return frame


# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        results = MODEL_IMAGE(frame)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, use_container_width=True)


# ---------------- VIDEO MODE (WebRTC style) ----------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        st.info("Streaming video smoothly...")

        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self.frame_skip = 5
                self.count = 0
                self.last_results = None

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # resize for speed
                img = cv2.resize(img, (480, 270))

                self.count += 1

                if self.count % self.frame_skip == 0:
                    self.last_results = MODEL_VIDEO(img, imgsz=320)[0]

                if self.last_results is not None:
                    img = draw_boxes(img, self.last_results)

                return img

        webrtc_streamer(
            key="video-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )


# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection")

    class WebcamProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = MODEL_WEBCAM(img, imgsz=512)[0]
            img = draw_boxes(img, results)
            return img

    webrtc_streamer(
        key="webcam",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
