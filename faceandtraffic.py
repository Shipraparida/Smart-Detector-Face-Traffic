import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort

# ==================== FACE DETECTION FUNCTIONS ====================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_from_image():
    input_folder = "faces.jpg"  # Put your face images in this folder
    output_folder = "output_faces"
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read {filename}, skipping.")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
            print(f"Processed and saved: {filename}")

    print("Face detection completed for all images.")

def detect_faces_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Real-Time Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# ==================== TRAFFIC ANALYSIS FUNCTION ====================

def detect_and_track_vehicles(video_path='simple.mp4'):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    tracker = Sort()

    # Vehicle classes mapping
    vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    counter = {name: 0 for name in vehicle_classes.values()}

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    line_position = int(frame_height * 0.6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cls = int(cls)
            if cls in vehicle_classes:
                detections.append([x1, y1, x2, y2, float(conf)])

        tracked_objects = tracker.update(np.array(detections)) if len(detections) > 0 else []

        for x1, y1, x2, y2, track_id in tracked_objects:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Count if crossing the line
            if cy > line_position - 2 and cy < line_position + 2:
                for result in results.boxes.data:
                    cls = int(result[5])
                    if cls in vehicle_classes:
                        counter[vehicle_classes[cls]] += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)
        cv2.imshow('Traffic Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save statistics
    df = pd.DataFrame(list(counter.items()), columns=['Vehicle Type', 'Count'])
    df.to_csv('traffic_stats.csv', index=False)
    print('Counts:', counter)

# ==================== USER MENU ====================

print("Choose an option:\n1. Detect faces from images\n2. Detect faces from webcam\n3. Detect and track vehicles\n4. Exit")
choose = input("Enter 1, 2, 3, or 4: ")

while choose:
    if choose == "4":
        print("Exiting...")
        break
    elif choose == "1":
        detect_faces_from_image()
    elif choose == "2":
        detect_faces_from_webcam()
    elif choose == "3":
        detect_and_track_vehicles()
    else:
        print("Invalid choice...")
    choose = input("\nEnter 1, 2, 3, or 4: ")