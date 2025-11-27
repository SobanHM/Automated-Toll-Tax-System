import os
import cv2
import yaml
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

# ================= CONFIG =================
DATASET_PATH = "Merged_Dataset"  # change if your dataset folder is elsewhere
OUTPUT_MULTIPLIER = 1.5  # multiply boxes to balance classes
TARGET_SPLIT = "train"

# ================= HELPER FUNCTIONS =================
def read_yolo_labels(label_path):
    """Read YOLO-format labels safely"""
    boxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return boxes, class_ids
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))  # convert float '3.0' to int
        except:
            continue
        coords = [float(p) for p in parts[1:]]
        # Clip coordinates to [0,1]
        coords = [min(max(c, 0.0), 1.0) for c in coords]
        class_ids.append(cls)
        boxes.append(coords)
    return boxes, class_ids

def write_yolo_label(label_path, boxes, class_ids):
    with open(label_path, "w") as f:
        for cls, box in zip(class_ids, boxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")

# ================= DATASET BALANCER CLASS =================
class DatasetBalancer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # read classes from data.yaml
        data_yaml = os.path.join(dataset_path, "data.yaml")
        with open(data_yaml, "r") as f:
            data = yaml.safe_load(f)
        self.class_names = data["names"]
        self.num_classes = len(self.class_names)
        print("Loaded classes:", self.class_names)

        # Define Albumentations transform for augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))

    def get_split_report(self, split):
        images_path = os.path.join(self.dataset_path, split, "images")
        labels_path = os.path.join(self.dataset_path, split, "labels")

        class_counts = [0] * self.num_classes
        total_boxes = 0
        for lbl_file in os.listdir(labels_path):
            lbl_file_path = os.path.join(labels_path, lbl_file)
            boxes, class_ids = read_yolo_labels(lbl_file_path)
            total_boxes += len(class_ids)
            for cid in class_ids:
                if cid < self.num_classes:
                    class_counts[cid] += 1
        return class_counts, total_boxes

    def print_report(self, plot=False):
        print("\n==== DATASET REPORT ====\n")
        for split in ["train", "valid", "test"]:
            class_counts, total_boxes = self.get_split_report(split)
            images = len(os.listdir(os.path.join(self.dataset_path, split, "images")))
            print(f"Split: {split}")
            print(f" Images: {images}")
            print(f" Boxes: {total_boxes}")
            for i, cname in enumerate(self.class_names):
                print(f"  {cname} ({i}): {class_counts[i]}")
            print("")

    def augment_image(self, img_path, lbl_path, target_class, aug_idx):
        img = cv2.imread(img_path)
        boxes, class_ids = read_yolo_labels(lbl_path)
        # Filter boxes for the target class
        filtered_boxes = [b for b, cid in zip(boxes, class_ids) if cid == target_class]
        if len(filtered_boxes) == 0:
            return None
        try:
            augmented = self.transform(image=img, bboxes=filtered_boxes, class_labels=[target_class]*len(filtered_boxes))
        except Exception as e:
            print(f"Augmentation failed for {img_path}: {e}")
            return None

        aug_img = augmented['image']
        aug_boxes = augmented['bboxes']
        aug_ids = augmented['class_labels']
        return aug_img, (aug_boxes, aug_ids)

    def balance_dataset(self, target_split="train", output_multiplier=1.5):
        print("\nBalancing dataset...\n")
        images_path = os.path.join(self.dataset_path, target_split, "images")
        labels_path = os.path.join(self.dataset_path, target_split, "labels")

        # Count current boxes per class
        class_counts, _ = self.get_split_report(target_split)
        max_count = max(class_counts)
        target_per_class = int(max_count * output_multiplier)
        print(f"Target per class = {target_per_class} boxes")

        # Augment classes below target
        for class_id, count in enumerate(class_counts):
            if count >= target_per_class:
                continue
            needed = target_per_class - count
            print(f"Augmenting class {self.class_names[class_id]}: Need {needed} new samples")
            # Get all images containing this class
            candidate_files = []
            for lbl_file in os.listdir(labels_path):
                lbl_file_path = os.path.join(labels_path, lbl_file)
                boxes, class_ids = read_yolo_labels(lbl_file_path)
                if class_id in class_ids:
                    candidate_files.append(lbl_file)

            if len(candidate_files) == 0:
                continue

            for i in tqdm(range(needed)):
                lbl_file = random.choice(candidate_files)
                img_file = lbl_file.replace(".txt", ".jpg")
                img_path = os.path.join(images_path, img_file)
                lbl_path = os.path.join(labels_path, lbl_file)

                result = self.augment_image(img_path, lbl_path, class_id, i)
                if result is None:
                    continue
                aug_img, (aug_boxes, aug_ids) = result

                # Save augmented image
                aug_img_name = img_file.replace(".jpg", f"_aug{i}.jpg")
                aug_lbl_name = lbl_file.replace(".txt", f"_aug{i}.txt")
                cv2.imwrite(os.path.join(images_path, aug_img_name), aug_img)
                write_yolo_label(os.path.join(labels_path, aug_lbl_name), aug_boxes, aug_ids)
        print("\nBalancing finished.")

# ================= MAIN =================
if __name__ == "__main__":
    db = DatasetBalancer(DATASET_PATH)
    print("\nBEFORE BALANCING:")
    db.print_report()

    db.balance_dataset(target_split=TARGET_SPLIT, output_multiplier=OUTPUT_MULTIPLIER)

    print("\nAFTER BALANCING:")
    db.print_report()




# # dataset_balancer.py (fixed & upgraded - Option B)
# import os
# import cv2
# import random
# import shutil
# from tqdm import tqdm
# import albumentations as A
# from collections import defaultdict
# import yaml
#
#
# class DatasetBalancer:
#     def __init__(self, dataset_path, classes):
#         self.dataset_path = dataset_path
#         self.classes = classes
#         self.train_dir = os.path.join(dataset_path, "train")
#         self.valid_dir = os.path.join(dataset_path, "valid")
#         self.test_dir = os.path.join(dataset_path, "test")
#         self.transform = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.5),
#             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=10, p=0.6),
#             A.GaussNoise(p=0.3),
#             A.MotionBlur(p=0.3)
#         ],
#             bbox_params=A.BboxParams(
#                 format='yolo',
#                 label_fields=['class_labels'],
#                 min_visibility=0.0,
#                 check_each_transform=True
#             ))
#
#     def print_report(self, plot=False):
#         print("\n==== DATASET REPORT ====\n")
#         for split_name, split_dir in [("train", self.train_dir),
#                                       ("valid", self.valid_dir),
#                                       ("test", self.test_dir)]:
#             img_files = os.listdir(os.path.join(split_dir, "images"))
#             lbl_files = os.listdir(os.path.join(split_dir, "labels"))
#             class_counts = defaultdict(int)
#             total_boxes = 0
#             for lbl_file in lbl_files:
#                 with open(os.path.join(split_dir, "labels", lbl_file), "r") as f:
#                     lines = f.readlines()
#                 total_boxes += len(lines)
#                 for line in lines:
#                     cid = int(line.strip().split()[0])
#                     class_counts[cid] += 1
#             print(f"Split: {split_name}")
#             print(f" Images: {len(img_files)}")
#             print(f" Boxes: {total_boxes}")
#             for cid, count in sorted(class_counts.items()):
#                 print(f"  {self.classes[cid]} ({cid}): {count}")
#             print("")
#
#     def balance_dataset(self, target_split="train", output_multiplier=1.5):
#         split_dir = getattr(self, f"{target_split}_dir")
#         img_dir = os.path.join(split_dir, "images")
#         lbl_dir = os.path.join(split_dir, "labels")
#
#         # Count existing boxes per class
#         class_counts = defaultdict(int)
#         lbl_files = os.listdir(lbl_dir)
#         for lbl_file in lbl_files:
#             with open(os.path.join(lbl_dir, lbl_file), "r") as f:
#                 for line in f:
#                     cid = int(line.strip().split()[0])
#                     class_counts[cid] += 1
#
#         max_count = max(class_counts.values())
#         target_count = int(max_count * output_multiplier)
#         print(f"Target per class = {target_count} boxes")
#
#         # Augment each class if needed
#         for cid, count in class_counts.items():
#             needed = target_count - count
#             if needed <= 0:
#                 continue
#             print(f"Augmenting class {self.classes[cid]}: Need {needed} new samples")
#
#             # Collect files containing this class
#             candidate_files = []
#             for lbl_file in lbl_files:
#                 with open(os.path.join(lbl_dir, lbl_file), "r") as f:
#                     lines = f.readlines()
#                 for line in lines:
#                     line_cid = int(line.strip().split()[0])
#                     if line_cid == cid:
#                         candidate_files.append(lbl_file)
#                         break
#             if not candidate_files:
#                 continue
#
#             for i in tqdm(range(needed), desc=f"Augmenting {self.classes[cid]}"):
#                 lbl_file = random.choice(candidate_files)
#                 img_file = lbl_file.replace(".txt", ".jpg")
#                 img_path = os.path.join(img_dir, img_file)
#                 lbl_path = os.path.join(lbl_dir, lbl_file)
#
#                 # Read image
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue
#
#                 # Read labels
#                 boxes = []
#                 class_ids = []
#                 with open(lbl_path, "r") as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         line_cid = int(parts[0])
#                         if line_cid != cid:
#                             continue
#                         class_ids.append(line_cid)
#                         boxes.append([float(x) for x in parts[1:]])
#
#                 if not boxes:
#                     continue
#
#                 # Apply augmentation
#                 augmented = self.transform(image=img, bboxes=boxes, class_labels=class_ids)
#                 aug_img = augmented["image"]
#                 aug_boxes = augmented["bboxes"]
#                 aug_labels = augmented["class_labels"]
#
#                 # Clip YOLO boxes to [0,1]
#                 aug_boxes_clipped = []
#                 aug_labels_clipped = []
#                 for b, lbl in zip(aug_boxes, aug_labels):
#                     x, y, w, h = b
#                     x = min(max(x, 0.0), 1.0)
#                     y = min(max(y, 0.0), 1.0)
#                     w = min(max(w, 0.0), 1.0)
#                     h = min(max(h, 0.0), 1.0)
#                     if w == 0 or h == 0:
#                         continue
#                     aug_boxes_clipped.append([x, y, w, h])
#                     aug_labels_clipped.append(lbl)
#
#                 # Save augmented image and label
#                 new_img_name = f"aug_{i}_{img_file}"
#                 new_lbl_name = new_img_name.replace(".jpg", ".txt")
#                 cv2.imwrite(os.path.join(img_dir, new_img_name), aug_img)
#                 with open(os.path.join(lbl_dir, new_lbl_name), "w") as f:
#                     for lbl_val, b in zip(aug_labels_clipped, aug_boxes_clipped):
#                         f.write(f"{lbl_val} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
#
#         print("Balancing finished.")
#
#
# # ===========================
# # Usage example
# # ===========================
# if __name__ == "__main__":
#     DATASET_PATH = r"C:/Users/soban/PycharmProjects/Smart-Toll-Tax-System/Merged_Dataset"
#     CLASS_NAMES = ["car", "truck", "van", "bus"]
#
#     db = DatasetBalancer(DATASET_PATH, CLASS_NAMES)
#     print("\nBEFORE BALANCING:")
#     db.print_report()
#
#     db.balance_dataset(target_split="train", output_multiplier=1.5)
#
#     print("\nAFTER BALANCING:")
#     db.print_report()
