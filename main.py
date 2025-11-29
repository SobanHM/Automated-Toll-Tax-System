import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TOLL_RATES


VIDEO_PATH = "videos/traffic_clip.mp4"
OUTPUT_PATH = "videos/processed_output.mp4"
LINE_Y = 500
CONF_THRESHOLD = 0.3


# Loading yolo  model

print("Loading YOLOv8 model..")
det_model = YOLO("yolov8m.pt")


# tracker start
print("Initializing DeepSort Tracker")
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)


# video cpature
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"cannot open video: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}")
print(f"Counting line: y={LINE_Y}")

# inittialize video writter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
print(f"Output video will be saved to: {OUTPUT_PATH}")

# state variables

class_counts = {}
vehicle_ids = set()

print("\nSmart Toll System Started.. Press 'q' to stop early.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete")
        break

    # 1 YOLO Detection
    results = det_model(frame, verbose=False)[0]

    # 2. Prepare detections for tracker
    detections_for_tracker = []
    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if conf < CONF_THRESHOLD:
            continue
        cls_name = det_model.names[int(cls_id)].lower()
        if cls_name not in TOLL_RATES:
            continue
        x1, y1, x2, y2 = map(float, box.tolist())
        w = x2 - x1
        h = y2 - y1
        detections_for_tracker.append([[x1, y1, w, h], float(conf), cls_name])

    # 3 Update tracker
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # 4 Count vehicles
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        cls_name = track.get_det_class()
        x1, y1, x2, y2 = track.to_ltrb()

        cy = int((y1 + y2) / 2)
        cx = int((x1 + x2) / 2)

        if cy > LINE_Y and track_id not in vehicle_ids:
            vehicle_ids.add(track_id)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Draw bounding box + id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
        cv2.putText(frame, f"ID:{track_id} {cls_name}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 5 Draw dashboard info
    cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 0, 255), 3)
    total_vehicles = sum(class_counts.values())
    total_revenue = sum(count * TOLL_RATES.get(cls, 0) for cls, count in class_counts.items())

    cv2.rectangle(frame, (20, 20), (450, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Vehicles: {total_vehicles}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Revenue: {total_revenue} RPS", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Save frames in outpurt video (new)
    out.write(frame)

    # Show window
    cv2.imshow("Smart Toll System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nUser stopped processing.")
        break

# Release everything
cap.release()
out.release()  # <--- NEW
cv2.destroyAllWindows()

# -----------------------
# FINAL SUMMARY
# -----------------------
print("\n" + "="*40)
print("Toll Plaza Report -Toll Summary")
print("="*40)
if not class_counts:
    print(" No tax-paying vehicles crossed the line.")
else:
    total_tax = 0
    for cls_name, count in class_counts.items():
        rate = TOLL_RATES.get(cls_name, 0)
        subtotal = count * rate
        total_tax += subtotal
        print(f" {cls_name.capitalize():<10}: {count}  (Tax: {subtotal} RPS)")
    print("-"*40)
    print(f"TOTAL REVENUE: {total_tax} RPS")
print("="*40 + "\n")

print(f"Video saved successfully at: {OUTPUT_PATH}")














"""
# ==================================    VEHICLES CLASSES WISE TOLL TAX ==============================

import cv2
import numpy as np
from ultralytics import YOLO
# Make sure you have installed this library: pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TOLL_RATES


VIDEO_PATH = "videos/traffic_clip.mp4"
LINE_Y = 500  # Adjusted line height (Try 500 or 600 based on your view)
CONF_THRESHOLD = 0.3

# -----------------------
# LOAD YOLO MODEL
# -----------------------
print("Loading YOLOv8 model...")
# Using 'yolov8n.pt' for speed. Use 'yolov8m.pt' for better accuracy if you have a good GPU.
det_model = YOLO("yolov8n.pt")


# Initiate trakeer
print("Initializing DeepSort Tracker...")
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}")
print(f"Counting line: y={LINE_Y}")


# State variables

class_counts = {}  # Stores total count per class (e.g., {'car': 5})
vehicle_ids = set()  # Stores IDs of vehicles that have already paid (e.g., {1, 4, 12})

print("\nSmart Toll System Started... Press 'q' to stop early.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    # 1. Run YOLO Inference
    results = det_model(frame, verbose=False)[0]

    # 2. Prepare detections for DeepSort
    # Format expected by DeepSort: [[left, top, w, h], confidence, detection_class]
    detections = []

    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if conf < CONF_THRESHOLD:
            continue

        cls_name = det_model.names[int(cls_id)].lower()

        # Only track vehicles that are in our price list
        if cls_name not in TOLL_RATES:
            continue

        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        # Append to list in the correct format for deep_sort_realtime
        detections.append([[x1, y1, w, h], float(conf), cls_name])

    # 3. Update Tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # 4. Count Logic
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        cls_name = track.get_det_class()

        # Get current position (bounding box)
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = ltrb

        # Calculate centroid (center point)
        cy = int((y1 + y2) / 2)
        cx = int((x1 + x2) / 2)

        # CHECK CROSSING:
        # We define a small "buffer" zone. If the car's center is below the line
        # and we haven't counted it yet, we bill it.
        if cy > LINE_Y and track_id not in vehicle_ids:
            vehicle_ids.add(track_id)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # Visual feedback for crossing (Green Circle)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        # Draw box and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
        cv2.putText(frame, f"ID: {track_id} {cls_name}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 5. Draw Interface
    # Draw the Toll Line
    cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 0, 255), 3)

    # Calculate Revenue
    current_revenue = 0
    total_vehicles = 0
    for c_name, count in class_counts.items():
        rate = TOLL_RATES.get(c_name, 0)
        current_revenue += count * rate
        total_vehicles += count

    # Dashboard Box
    cv2.rectangle(frame, (20, 20), (450, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Vehicles: {total_vehicles}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Revenue: {current_revenue} RPS", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smart Toll System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n User stopped processing.")
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------
# FINAL SUMMARY
# -----------------------
print("\n" + "=" * 40)
print(" Toll Plaza Report SUmmary")
print("=" * 40)

if not class_counts:
    print(" No tax-paying vehicles crossed the line.")
else:
    total_tax = 0
    for cls_name, count in class_counts.items():
        rate = TOLL_RATES.get(cls_name, 0)
        subtotal = count * rate
        total_tax += subtotal
        print(f" {cls_name.capitalize():<10} : {count}  (Tax: {subtotal} RPS)")

    print("-" * 40)
    print(f" TOTAL REVENUE: {total_tax} RPS")

print("=" * 40 + "\n")

"""