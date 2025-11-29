"""
toll_count_infer.py
Detect + track vehicles from a video, count each vehicle once when it crosses a defined line,
and accumulate toll fees per class.
Usage:
    python toll_count_infer.py --weights path/to/best.pt --source path/to/video.mp4 --out out_video.mp4
Requirements:
    pip install ultralytics opencv-python numpy
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
from config import TOLL_RATES as toll_rates_vehicles


# toll rates in pkr
# TOLL = {"car": 50, "truck": 150, "van": 100, "bus": 200}
TOLL = toll_rates_vehicles

# class name order as in model
CLASS_NAMES = ["car", "truck", "van", "bus"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--source", required=True, help="Path to video file (or camera index)")
    p.add_argument("--out", default="output_toll_tax_video.mp4", help="Output annotated video path")
    p.add_argument("--line_y", type=int, default=450, help="Y coordinate of toll line (pixel)")
    p.add_argument("--direction", choices=["down", "up"], default="down",
                   help="Direction of crossing to count. 'down' means crossing from smaller y to larger y.")
    p.add_argument("--show", action="store_true", help="Show live window while processing")
    p.add_argument("--tracker", default="botsort.yaml",
                   help="Tracker config (ultralytics supports botsort/bytetrack etc.)")
    return p.parse_args()


def centroid_from_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return int(cx), int(cy)


def main():
    args = parse_args()

    # Load model
    model = YOLO(args.weights)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Tracking state
    previous_centroids = {}  # track_id -> (cx, cy)
    counted_ids = set()  # track ids already counted
    counts = defaultdict(int)  # class_name -> count
    total_toll = 0

    toll_line_y = args.line_y
    direction = args.direction  # "down" or "up"

    frame_idx = 0
    t0 = time.time()

    #  use model.track for integrated detection+tracking (stream)
    # If the ultralytics version supports: results = model.track(source=..., tracker=args.tracker, stream=True)
    # But here we loop frames and call model.predict with tracker param for robustness.
    # Using model.track is efficient; below uses model.track if available.
    try_model_track = True
    if try_model_track:
        # Using .track with stream=True returns a generator of results per frame
        stream = model.track(source=args.source, tracker=args.tracker, stream=True, verbose=False, device=None)
        # The stream yields ultralytics Results objects
        for r in stream:
            frame_idx += 1
            frame = r.orig_img.copy() if hasattr(r, "orig_img") and r.orig_img is not None else None
            if frame is None:
                continue

            # r.boxes: Boxes object. Try to read IDs and class names
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                # write frame and continue
                cv2.line(frame, (0, toll_line_y), (width, toll_line_y), (0, 0, 255), 2)
                out_writer.write(frame)
                if args.show:
                    cv2.imshow("out", frame);
                    cv2.waitKey(1)
                continue

            # boxes.xyxy, boxes.conf, boxes.cls, boxes.id may be present
            xyxys = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.array([])
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.array([])
            clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.array([])
            ids = None
            try:
                ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                # try alternative attribute name
                try:
                    ids = boxes.boxes_id.cpu().numpy().astype(int)
                except Exception:
                    ids = np.arange(len(xyxys), dtype=int)  # fallback - no tracking IDs

            # annotate each box
            for i, xyxy in enumerate(xyxys):
                x1, y1, x2, y2 = map(int, xyxy)
                cls_id = int(clss[i]) if i < len(clss) else -1
                track_id = int(ids[i]) if i < len(ids) else i
                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                cx, cy = centroid_from_xyxy((x1, y1, x2, y2))

                # previous centroid
                prev = previous_centroids.get(track_id, None)

                # crossing detection: count when centroid crosses the toll_line_y in the chosen direction
                crossed = False
                if prev is not None and track_id not in counted_ids:
                    prev_y = prev[1]
                    cur_y = cy
                    if direction == "down":
                        if prev_y < toll_line_y <= cur_y:
                            crossed = True
                    else:  # up
                        if prev_y > toll_line_y >= cur_y:
                            crossed = True
                # if counted, do nothing; if cross, increment count & toll
                if crossed:
                    counted_ids.add(track_id)
                    counts[label] += 1
                    toll_amt = TOLL.get(label, 0)
                    nonlocal_total = 0
                    total_toll += toll_amt
                # update centroid history
                previous_centroids[track_id] = (cx, cy)
                # draw box and id
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {int(confs[i] * 100) if i < len(confs) else ''}% ID:{track_id}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # draw centroid
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

            # draw toll line and overlay counts/toll
            cv2.line(frame, (0, toll_line_y), (width, toll_line_y), (0, 0, 255), 2)
            info_y = 30
            cv2.putText(frame, f"Total Toll: Rs {total_toll}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                        2)
            info_y += 30
            for cls in CLASS_NAMES:
                cv2.putText(frame, f"{cls}: {counts.get(cls, 0)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                info_y += 24

            out_writer.write(frame)
            if args.show:
                cv2.imshow("out", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # end stream loop
        out_writer.release()
        if args.show:
            cv2.destroyAllWindows()
        print("Finished processing (stream).")
        print("Counts:", dict(counts))
        print("Total toll:", total_toll)
        return

    # fallback (frame-by-frame detect+track)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # predict with tracking ON (ultralytics supports tracker arg in predict/track)
        results = model.track(frame, tracker=args.tracker)
        # If predict/track returns results object list:
        for r in results:
            # r.boxes similar to above adn implement same logic as stream branch
            pass

    out_writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
