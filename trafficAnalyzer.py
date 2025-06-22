import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Tracker
tracker = Sort()

# Vehicle classes mapping
vehicle_classes = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Counters
counter = {name: 0 for name in vehicle_classes.values()}

# Video setup
cap = cv2.VideoCapture('simple.mp4')
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
            detections.append([x1, y1, x2, y2, conf.item()])

    tracked_objects = tracker.update(np.array(detections))

    for x1, y1, x2, y2, track_id in tracked_objects:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

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

# Save stats
df = pd.DataFrame(list(counter.items()), columns=['Vehicle Type', 'Count'])
df.to_csv('traffic_stats.csv', index=False)
print('Counts:', counter)