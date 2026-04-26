import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n-oiv7.pt")  # fast model for web

st.title("YOLO Object Detection App")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ---------------- STYLE ----------------
BOX_COLOR = (0, 0, 255)  # Red
THICKNESS = 1            # Thin box
FONT_SCALE = 0.5


# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results):
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

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

        results = model(frame)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, caption="Detection Result", use_container_width=True)


# ---------------- VIDEO MODE ----------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            frame = draw_boxes(frame, results)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()


# ---------------- LIVE WEBCAM MODE ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Optional: resize for performance
            img = cv2.resize(img, (640, 480))

            results = model(img)[0]
            img = draw_boxes(img, results)

            return img

    webrtc_streamer(
        key="yolo-live",
        video_transformer_factory=YOLOVideoTransformer
    )
