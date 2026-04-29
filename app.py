import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import imageio_ffmpeg
# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("YOLO Object Detection App")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return {
        "webcam": YOLO("yolov8n-oiv7.pt"),
        "video": YOLO("yolov8s.pt"),
        "image": YOLO("yolov8m.pt")
    }

models = load_models()
MODEL_WEBCAM = models["webcam"]
MODEL_VIDEO = models["video"]
MODEL_IMAGE = models["image"]

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

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
        # Force the image to standard 3-channel RGB (Fixes RGBA/Grayscale crashes)
        image = Image.open(uploaded_file).convert("RGB")
        
        # Convert PIL Image (RGB) to OpenCV format (BGR) for YOLO
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = MODEL_IMAGE(frame, imgsz=640)[0]
        frame = draw_boxes(frame, results)

        # Convert back to RGB so Streamlit displays the colors correctly
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        st.image(frame, caption="Detection Result", use_container_width=True)


# ---------------- VIDEO MODE ----------------
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Create temp files for input, raw output, and web-ready output
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        input_path = tfile_in.name
        
        output_path = input_path.replace('.mp4', '_raw.mp4')
        final_path = input_path.replace('.mp4', '_web.mp4')

        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up OpenCV VideoWriter (using your resized dimensions: 640x360)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))

        # UI Elements for progress
        st.write("Processing video... This may take a moment.")
        progress_bar = st.progress(0)
        
        frame_skip = 2
        count = 0
        last_results = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            frame = cv2.resize(frame, (640, 360))

            # Run YOLO inference
            if count % frame_skip == 0:
                last_results = MODEL_VIDEO(frame, imgsz=416)[0]

            # Draw boxes if we have results
            if last_results is not None:
                frame = draw_boxes(frame, last_results)

            # Write the processed frame to the new video file
            out.write(frame)
            
            # Update progress bar
            if total_frames > 0:
                progress_bar.progress(min(count / total_frames, 1.0))

        # Release resources
        cap.release()
        out.release()
        
        # Convert the video to H264 codec using Python's FFmpeg binary
        st.write("Converting to web-friendly format...")
        
        # Get the path to the installed ffmpeg binary
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe() 
        
        # Run the conversion
        os.system(f'"{ffmpeg_path}" -y -i "{output_path}" -vcodec libx264 "{final_path}"')

        # Clear progress UI and display the final video
        progress_bar.empty()
        st.success("Processing Complete!")
        
        with open(final_path, 'rb') as video_file:
            st.video(video_file.read())


# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.write("Live Webcam Detection (Smooth Streaming Mode)")

    class YOLOVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_img = None

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Process every 3rd frame to prevent freezing on Streamlit Cloud CPU
            if self.frame_count % 3 == 0 or self.last_img is None:
                # Changed imgsz from 410 to 416 (MUST be a multiple of 32 for YOLO)
                results = MODEL_WEBCAM(img, imgsz=416)[0]
                img = draw_boxes(img, results)
                self.last_img = img
            else:
                # Re-use the last drawn image to save CPU
                img = self.last_img
                
            return img

    webrtc_streamer(
        key="yolo-live",
        video_transformer_factory=YOLOVideoTransformer,
        # Added STUN Servers: Required to establish a connection on Streamlit Cloud!
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
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
