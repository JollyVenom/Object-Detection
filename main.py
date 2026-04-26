import cv2
from ultralytics import YOLO

# Load YOLO model
YOLO_SIZE = 416
model = YOLO("yolov8s-oiv7.pt")

# Confidence threshold
CONF_THRESHOLD = 0.5   # only show detections >= 50%

# Ask for input (image, video, or webcam)
image_path = input("Enter image/video path (or leave blank for webcam): ").strip().strip('"')

# Box & text style (light red)
box_color = (0, 0, 200)   # BGR → Light Red
thickness = 1             # thin border
font_scale = 0.5
font_thickness = 1

def draw_boxes(frame, results):
    """Draw detection boxes and labels on the frame."""
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:   # skip weak detections
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Optional: add a filled background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 0), -1)  # semi-black bg
        cv2.putText(frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, font_thickness)

    return frame


if image_path:  
    # Detect if input is an image or a video
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Could not read image. Please check the path.")
            exit()
        results = model(frame, imgsz=YOLO_SIZE)[0]
        frame = draw_boxes(frame, results)
        cv2.imshow("Image Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
            print("❌ Could not open video. Please check the path.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=YOLO_SIZE)[0]
            frame = draw_boxes(frame, results)
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("❌ Unsupported file type. Please provide image or video.")
        exit()

else:  
    # Webcam mode
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        exit()

    print("\n--- Controls ---")
    print("[Q]: Quit")
    print("----------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=YOLO_SIZE)[0]
        frame = draw_boxes(frame, results)

        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()









