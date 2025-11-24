import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TOLL_RATES

# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = "videos/traffic_clip.mp4"  # your video path
OUTPUT_PATH = "videos/processed_output.mp4"   # <--- NEW
LINE_Y = 500
CONF_THRESHOLD = 0.3

# -----------------------
# LOAD YOLO MODEL
# -----------------------
print("⏳ Loading YOLOv8 model...")
det_model = YOLO("yolov8m.pt")

# -----------------------
# INIT TRACKER
# -----------------------
print("⏳ Initializing DeepSort Tracker...")
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"📏 Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}")
print(f"🔹 Counting line: y={LINE_Y}")

# inittialize video writter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
print(f"💾 Output video will be saved to: {OUTPUT_PATH}")

# state variables

class_counts = {}
vehicle_ids = set()

print("\n🚀 Smart Toll System Started... Press 'q' to stop early.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    # 1. YOLO Detection
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

    # 3. Update tracker
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # 4. Count vehicles
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

    # 5. Draw dashboard info
    cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 0, 255), 3)
    total_vehicles = sum(class_counts.values())
    total_revenue = sum(count * TOLL_RATES.get(cls, 0) for cls, count in class_counts.items())

    cv2.rectangle(frame, (20, 20), (450, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Vehicles: {total_vehicles}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Revenue: {total_revenue} RPS", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -----------------------
    # SAVE FRAME INTO OUTPUT VIDEO (NEW)
    # -----------------------
    out.write(frame)

    # Show window
    cv2.imshow("Smart Toll System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n🛑 User stopped processing.")
        break

# Release everything
cap.release()
out.release()  # <--- NEW
cv2.destroyAllWindows()

# -----------------------
# FINAL SUMMARY
# -----------------------
print("\n" + "="*40)
print(" 📋 TOLL PLAZA SUMMARY REPORT")
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
    print(f" 💰 TOTAL REVENUE: {total_tax} RPS")
print("="*40 + "\n")

print(f"🎉 Video saved successfully at: {OUTPUT_PATH}")

# import cv2
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from config import TOLL_RATES
#
# # -----------------------
# # CONFIG
# # -----------------------
# VIDEO_PATH = "videos/traffic_clip.mp4"  # your video path
# LINE_Y = 500                            # counting line
# CONF_THRESHOLD = 0.3                     # YOLO confidence threshold
#
# # -----------------------
# # LOAD YOLO MODEL
# # -----------------------
# print("⏳ Loading YOLOv8 model...")
# det_model = YOLO("yolov8m.pt")  # higher accuracy
#
# # -----------------------
# # INIT TRACKER
# # -----------------------
# print("⏳ Initializing DeepSort Tracker...")
# tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)
#
# # -----------------------
# # VIDEO CAPTURE
# # -----------------------
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"❌ Cannot open video: {VIDEO_PATH}")
#     exit()
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"📏 Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}")
# print(f"🔹 Counting line: y={LINE_Y}")
#
# # -----------------------
# # STATE VARIABLES
# # -----------------------
# class_counts = {}   # {'car': 0, 'truck': 0, ...}
# vehicle_ids = set() # IDs of vehicles already counted
#
# print("\n🚀 Smart Toll System Started... Press 'q' to stop early.\n")
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("✅ Video processing complete.")
#         break
#
#     # 1. YOLO Detection
#     results = det_model(frame, verbose=False)[0]
#
#     # 2. Prepare detections for tracker
#     detections_for_tracker = []
#     for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
#         if conf < CONF_THRESHOLD:
#             continue
#         cls_name = det_model.names[int(cls_id)].lower()
#         if cls_name not in TOLL_RATES:
#             continue
#         x1, y1, x2, y2 = map(float, box.tolist())
#         w = x2 - x1
#         h = y2 - y1
#         detections_for_tracker.append([[x1, y1, w, h], float(conf), cls_name])
#
#     # 3. Update tracker
#     tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
#
#     # 4. Count vehicles
#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         cls_name = track.get_det_class()
#         x1, y1, x2, y2 = track.to_ltrb()
#         cy = int((y1 + y2)/2)
#         cx = int((x1 + x2)/2)
#
#         # Count if crossed line
#         if cy > LINE_Y and track_id not in vehicle_ids:
#             vehicle_ids.add(track_id)
#             class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
#             cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
#
#         # Draw box + ID
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
#         cv2.putText(frame, f"ID:{track_id} {cls_name}", (int(x1), int(y1)-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#
#     # 5. Draw dashboard
#     cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0,0,255), 3)
#     total_vehicles = sum(class_counts.values())
#     total_revenue = sum(count*TOLL_RATES.get(cls,0) for cls,count in class_counts.items())
#     cv2.rectangle(frame, (20,20),(450,120),(0,0,0),-1)
#     cv2.putText(frame, f"Vehicles: {total_vehicles}", (40,60),
#                 cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
#     cv2.putText(frame, f"Revenue: {total_revenue} RPS", (40,100),
#                 cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#
#     cv2.imshow("Smart Toll System", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         print("\n🛑 User stopped processing.")
#         break
#
#
# cap.release()
#
# cv2.destroyAllWindows()
#
# # -----------------------
# # FINAL SUMMARY
# # -----------------------
# print("\n" + "="*40)
# print(" 📋 TOLL PLAZA SUMMARY REPORT")
# print("="*40)
# if not class_counts:
#     print(" No tax-paying vehicles crossed the line.")
# else:
#     total_tax = 0
#     for cls_name, count in class_counts.items():
#         rate = TOLL_RATES.get(cls_name,0)
#         subtotal = count*rate
#         total_tax += subtotal
#         print(f" {cls_name.capitalize():<10}: {count}  (Tax: {subtotal} RPS)")
#     print("-"*40)
#     print(f" 💰 TOTAL REVENUE: {total_tax} RPS")
# print("="*40 + "\n")

"""
# ==================================    VEHICLES CLASSES WISE TOLL TAX ==============================

import cv2
import numpy as np
from ultralytics import YOLO
# Make sure you have installed this library: pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TOLL_RATES

# -----------------------
# CONFIGURATION
# -----------------------
VIDEO_PATH = "videos/traffic_clip.mp4"
LINE_Y = 500  # Adjusted line height (Try 500 or 600 based on your view)
CONF_THRESHOLD = 0.3

# -----------------------
# LOAD YOLO MODEL
# -----------------------
print("⏳ Loading YOLOv8 model...")
# Using 'yolov8n.pt' for speed. Use 'yolov8m.pt' for better accuracy if you have a good GPU.
det_model = YOLO("yolov8n.pt")

# -----------------------
# INIT TRACKER
# -----------------------
print("⏳ Initializing DeepSort Tracker...")
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Cannot open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"📏 Video Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}")
print(f"🔹 Counting line: y={LINE_Y}")

# -----------------------
# STATE VARIABLES
# -----------------------
class_counts = {}  # Stores total count per class (e.g., {'car': 5})
vehicle_ids = set()  # Stores IDs of vehicles that have already paid (e.g., {1, 4, 12})

print("\n🚀 Smart Toll System Started... Press 'q' to stop early.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
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
        print("\n🛑 User stopped processing.")
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------
# FINAL SUMMARY
# -----------------------
print("\n" + "=" * 40)
print(" 📋 TOLL PLAZA SUMMARY REPORT")
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
    print(f" 💰 TOTAL REVENUE: {total_tax} RPS")

print("=" * 40 + "\n")

"""


# ================== 02 =====================
# import cv2
# from ultralytics import YOLO
# from config import TOLL_RATES
#
# # -------------------------------------------------------------
# # CONFIG
# # -------------------------------------------------------------
# VIDEO_PATH = "videos/traffic_clip.mp4"
# MODEL_NAME = "yolov8m.pt"
# COUNT_LINE_OFFSET_FROM_BOTTOM = 100
# SHOW_WINDOW = True
# CONFIDENCE = 0.3
# IOU = 0.5
# LINE_WIDTH = 2
# DASHBOARD_BOX = (20, 20, 550, 120)
#
# # -------------------------------------------------------------
# # 1. Open video
# # -------------------------------------------------------------
# cap = cv2.VideoCapture(VIDEO_PATH)
# assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#
# print(f"📏 Video Resolution: {width}x{height} | FPS: {fps}")
#
# line_y = height - COUNT_LINE_OFFSET_FROM_BOTTOM
# line_points = [(0, line_y), (width, line_y)]
# print("🔹 Counting line:", line_points)
#
# # -------------------------------------------------------------
# # 2. Load YOLOv8 model for class detection
# # -------------------------------------------------------------
# model = YOLO(MODEL_NAME)  # yolov8 detection model
#
# # Initialize total counts dictionary
# total_class_counts = {cls: 0 for cls in TOLL_RATES.keys()}
#
# print("\n🚀 Smart Toll System Started... Press 'q' to stop early.\n")
#
# # -------------------------------------------------------------
# # 3. Main loop
# # -------------------------------------------------------------
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("\n✅ Video Processing Complete.")
#         break
#
#     # Detect objects
#     results = model(frame, conf=CONFIDENCE, iou=IOU)[0]  # take first result object
#     frame_class_counts = {cls: 0 for cls in TOLL_RATES.keys()}
#
#     for box, cls_id in zip(results.boxes, results.boxes.cls):
#         cls_id = int(cls_id)
#         cls_name = model.names[cls_id].lower()
#         if cls_name in frame_class_counts:
#             frame_class_counts[cls_name] += 1
#             total_class_counts[cls_name] += 1
#
#     # Calculate current revenue
#     current_revenue = sum(count * TOLL_RATES[cls] for cls, count in frame_class_counts.items())
#
#     # Draw dashboard
#     x1, y1, x2, y2 = DASHBOARD_BOX
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
#     y_offset = y1 + 40
#     for cls, count in frame_class_counts.items():
#         cv2.putText(frame, f"{cls.capitalize()}: {count}", (x1 + 20, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
#         y_offset += 30
#     cv2.putText(frame, f"REVENUE: {current_revenue} RPS", (x1 + 20, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
#
#     # Show frame
#     if SHOW_WINDOW:
#         cv2.imshow("Smart Toll System", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n🛑 User stopped the process.")
#             break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # -------------------------------------------------------------
# # 4. Final summary report
# # -------------------------------------------------------------
# print("\n" + "=" * 40)
# print(" 📋 TOLL PLAZA SUMMARY REPORT")
# print("=" * 40)
#
# total_vehicles = 0
# total_tax_collected = 0
#
# for cls, count in total_class_counts.items():
#     if count > 0:
#         subtotal = count * TOLL_RATES[cls]
#         print(f"{cls.capitalize():12} : {count}  (Tax: {subtotal} RPS)")
#         total_vehicles += count
#         total_tax_collected += subtotal
#
# print("-" * 40)
# print(f" Total vehicles detected: {total_vehicles}")
# print(f" 💰 TOTAL TOLL TAX: {total_tax_collected} RPS")
# print("=" * 40 + "\n")


#  works correctly or as expected
# import cv2
# from ultralytics import solutions
# from config import TOLL_RATES
#
# # -------------------------------------------------------------
# # CONFIGURATION
# # -------------------------------------------------------------
# VIDEO_PATH = "videos/traffic_clip.mp4"
# MODEL_NAME = "yolov8m.pt"          # or yolov8n.pt
# COUNT_LINE_OFFSET_FROM_BOTTOM = 100
# SHOW_WINDOW = True
# CONFIDENCE = 0.3
# IOU = 0.5
# TRACKER = "bytetrack.yaml"
# LINE_WIDTH = 2
# DASHBOARD_BOX = (20, 20, 450, 110)
# DEFAULT_TOLL = 50                   # fallback toll per vehicle
#
# # -------------------------------------------------------------
# # 1. Open video
# # -------------------------------------------------------------
# cap = cv2.VideoCapture(VIDEO_PATH)
# assert cap.isOpened(), f"❌ Cannot open video: {VIDEO_PATH}"
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#
# print(f"📏 Video Resolution: {width}x{height}  |  FPS: {fps}")
#
# line_y = height - COUNT_LINE_OFFSET_FROM_BOTTOM
# line_points = [(0, line_y), (width, line_y)]
# print("🔹 Counting line:", line_points)
#
# # -------------------------------------------------------------
# # 2. Initialize ObjectCounter
# # -------------------------------------------------------------
# counter = solutions.ObjectCounter(
#     model=MODEL_NAME,
#     region=line_points,
#     show=True,
#     classes=None,
#     tracker=TRACKER,
#     conf=CONFIDENCE,
#     iou=IOU,
#     line_width=LINE_WIDTH,
#     show_in=True,
#     show_out=True,
#     verbose=False
# )
#
# print("\n🚀 Smart Toll System Started... Press 'q' to stop early.\n")
#
# # -------------------------------------------------------------
# # 3. Main loop
# # -------------------------------------------------------------
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("\n✅ Video Processing Complete.")
#         break
#
#     # Run counter
#     result = counter(frame)
#
#     # Use plotted frame if available
#     if hasattr(result, "plot_im") and result.plot_im is not None:
#         display_frame = result.plot_im
#     else:
#         display_frame = frame
#
#     # Calculate revenue using fallback totals
#     in_count = getattr(counter, "in_count", 0)
#     out_count = getattr(counter, "out_count", 0)
#     total_vehicles = in_count + out_count
#     current_revenue = total_vehicles * DEFAULT_TOLL
#
#     # Draw dashboard
#     x1, y1, x2, y2 = DASHBOARD_BOX
#     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
#     cv2.putText(display_frame, f"VEHICLES: {total_vehicles}", (x1 + 20, y1 + 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#     cv2.putText(display_frame, f"REVENUE: {current_revenue} RPS", (x1 + 20, y1 + 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
#
#     # Show frame
#     if SHOW_WINDOW:
#         cv2.imshow("Smart Toll System", display_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n🛑 User stopped the process.")
#             break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#
# # -------------------------------------------------------------
# # 4. Final Summary
# # -------------------------------------------------------------
# in_count = getattr(counter, "in_count", 0)
# out_count = getattr(counter, "out_count", 0)
# total_vehicles = in_count + out_count
# total_tax_collected = total_vehicles * DEFAULT_TOLL
#
# print("\n" + "=" * 40)
# print(" 📋 TOLL PLAZA SUMMARY REPORT")
# print("=" * 40)
# print(f" Total vehicles detected: {total_vehicles}")
# print(f" 💰 TOTAL TOLL TAX: {total_tax_collected} RPS")
# print("=" * 40 + "\n")
