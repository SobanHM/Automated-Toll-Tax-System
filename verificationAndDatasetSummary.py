
# verification : count images and labels per class
import os

DATASET_DIR = r"Merged_Dataset"
CLASS_NAMES = ["car", "truck", "van", "bus"]


def summarize_split(split):
    print(f"\n=== Split: {split} ===")

    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")

    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    label_files = [f for f in os.listdir(lbl_dir)
                   if f.endswith(".txt")]

    print(f"Images: {len(image_files)}")
    print(f"Label files: {len(label_files)}")

    # Count class occurrences
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_boxes = 0

    for lbl in label_files:
        path = os.path.join(lbl_dir, lbl)

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 1:
                continue

            cls = int(parts[0])  # YOLO class ID
            if cls in class_counts:
                class_counts[cls] += 1
                total_boxes += 1

    print(f"Total Boxes: {total_boxes}")

    for cls_id, count in class_counts.items():
        print(f"  {CLASS_NAMES[cls_id]} ({cls_id}): {count}")


def main():
    print("\n===== DATASET SUMMARY =====")
    for split in ["train", "valid", "test"]:
        summarize_split(split)

    print("\nDone.\n")


if __name__ == "__main__":
    main()

#
# import os
# from collections import defaultdict
#
# output_dir = "Merged_Dataset"
# classes = ['car', 'motorcycle', 'truck', 'van', 'bus']
#
# splits = ["train", "valid", "test"]
#
# for split in splits:
#     label_dir = os.path.join(output_dir, split, "labels")
#     class_count = defaultdict(int)
#     total_labels = 0
#
#     for label_file in os.listdir(label_dir):
#         if not label_file.endswith(".txt"):
#             continue
#         with open(os.path.join(label_dir, label_file), "r") as f:
#             lines = f.readlines()
#         total_labels += len(lines)
#         for line in lines:
#             class_id = int(line.strip().split()[0])
#             class_count[class_id] += 1
#
#     print(f"\nSplit: {split}")
#     print(f"Total label boxes: {total_labels}")
#     for cid, count in sorted(class_count.items()):
#         print(f"{classes[cid]} ({cid}): {count}")