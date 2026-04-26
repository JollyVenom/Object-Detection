import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- LOAD MODEL ----------------
# Use nano model for speed (best for real-time)
model = YOLO("yolov8n.pt")

st.title("YOLO Object Detection App")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

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

        results = model(frame, imgsz=640)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, caption="Detection Result", use_container_width=True)


# ---------------- VIDEO MODE (OPTIMIZED) ----------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        frame_skip = 3   # skip frames for speed
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % frame_skip != 0:
                continue

            # Resize for performance (important)
            frame = cv2.resize(frame, (640, 360))

            # Fast inference
            results = model(frame, imgsz=416)[0]
            frame = draw_boxes(frame, results)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()


# ---------------- WEBCAM MODE (CLEAR + FAST) ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # NO resize here → keeps clarity
            results = model(img, imgsz=640)[0]
            img = draw_boxes(img, results)

            return img

    webrtc_streamer(
        key="yolo-live",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
    )
