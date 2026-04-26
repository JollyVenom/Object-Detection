import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("YOLO Object Detection App")

# ---------------- LAZY LOAD MODEL ----------------
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Camera"])
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

    st.subheader("Image Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model = load_model("yolov8s.pt")

        image = Image.open(uploaded_file)
        frame = np.array(image)

        results = model(frame, imgsz=640)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, caption="Detection Result", use_container_width=True)


# ---------------- VIDEO MODE ----------------
elif mode == "Video":

    st.subheader("Video Detection")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        model = load_model("yolov8n.pt")

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

                if count % frame_skip == 0:
                    last_results = model(frame, imgsz=320)[0]

                if last_results is not None:
                    frame = draw_boxes(frame, last_results)

                if count % display_skip == 0:
                    stframe.image(frame, channels="BGR", use_container_width=True)

                time.sleep(0.01)

            cap.release()


# ---------------- CAMERA MODE (STABLE) ----------------
elif mode == "Camera":

    st.subheader("Capture Image from Camera")

    img_file = st.camera_input("Take a picture")

    if img_file:
        model = load_model("yolov8n.pt")

        image = Image.open(img_file)
        frame = np.array(image)

        results = model(frame, imgsz=320)[0]
        frame = draw_boxes(frame, results)

        st.image(frame, caption="Detection Result", use_container_width=True)
