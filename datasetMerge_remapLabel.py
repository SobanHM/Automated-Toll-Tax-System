# import os
# import shutil
# import yaml
#
# # paths to the original datasets
# datasets = {
#     "dataset1": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\vans_v1i_yolov8",  # Vans
#     "dataset2": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\Truck_v1i_yolov8",  # Trucks mixed
#     "dataset3": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\Objectdetection_v1i_yolov8",  # Object detection
# }
#
# # merged dataset as putput
# output_dir = "Merged_Dataset"
# """
# D-1   nc: 1 :-> names: ['Van']
# D-2:  nc: 5 :-> names: ['Car', 'Motorcycle', 'Person', 'Truck', 'Van']
# D-3:  nc: 6 :-> names: ['bike', 'bus', 'car', 'person', 'traffic signal', 'truck']
# """
#
#
# # final classes mapping
# final_classes = ["car", "motorcycle", "truck", "van", "bus"]
#
# # old to new class mapping for each dataset
# class_mappings = {
#     "dataset1": {"Van": 3},  # dataset1 only has Van
#     "dataset2": {"Car": 0, "Motorcycle": 1, "Truck": 2, "Van": 3, "Person": None},
#     "dataset3": {"car": 0, "bike": 1, "truck": 2, "bus": 4, "person": None, "traffic signal": None}
# }
#
#
# # directory func
# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# def copy_and_remap_labels(dataset_name, split):
#     src_images = os.path.join(datasets[dataset_name], split, "images")
#     src_labels = os.path.join(datasets[dataset_name], split, "labels")
#
#     dst_images = os.path.join(output_dir, split, "images")
#     dst_labels = os.path.join(output_dir, split, "labels")
#
#     ensure_dir(dst_images)
#     ensure_dir(dst_labels)
#
#     dataset_yaml_path = os.path.join(datasets[dataset_name], "data.yaml")
#     with open(dataset_yaml_path, "r") as yf:
#         yaml_data = yaml.safe_load(yf)
#
#     old_names = yaml_data["names"]  # Example: ['Car','Motorcycle','Person','Truck','Van']
#
#     for img_file in os.listdir(src_images):
#         if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue
#         # copy image with new name if conflict appeare
#         base_name = f"{dataset_name}_{img_file}"
#         shutil.copy2(os.path.join(src_images, img_file), os.path.join(dst_images, base_name))
#
#         # corresponding label file for each
#         label_file = os.path.splitext(img_file)[0] + ".txt"
#         src_label_path = os.path.join(src_labels, label_file)
#         dst_label_path = os.path.join(dst_labels, base_name.replace(os.path.splitext(base_name)[1], ".txt"))
#
#         if not os.path.exists(src_label_path):
#             # if there is no labels create empty file
#             open(dst_label_path, 'w').close()
#             continue
#
#         with open(src_label_path, "r") as f:
#             lines = f.readlines()
#
#         new_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             old_class_id = int(parts[0])
#             # maping old class_id to new class_id
#             old_class_name = old_names[old_class_id]  # get class name from YAML using old_class_id
#             new_class_id = class_mappings[dataset_name].get(old_class_name, None)  # map to final class
#             if new_class_id is None:
#                 continue  # skip other classes : like person trafficsignal
#             # leeping bounding box coordinates as orginal
#             new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
#
#         # Write remapped label file
#         with open(dst_label_path, "w") as f:
#             f.writelines(new_lines)
#
#
# # merging datasets : splits: train, test, valiadtion
# for split in ["train", "valid", "test"]:
#     for dataset_name in datasets.keys():
#         copy_and_remap_labels(dataset_name, split)
#
#
# # create new data.yaml (defined as our own)
#
# data_yaml_path = os.path.join(output_dir, "data.yaml")
# num_classes = len(final_classes)
# yaml_content = f"""train: {os.path.join(output_dir, 'train', 'images')}
# val: {os.path.join(output_dir, 'valid', 'images')}
# test: {os.path.join(output_dir, 'test', 'images')}
#
# nc: {num_classes}
# names: {final_classes}
# """
#
# with open(data_yaml_path, "w") as f:
#     f.write(yaml_content)
#
# print("Dataset merged successfully!")
# print(f"output folder: {output_dir}")
# print(f"final classes: {final_classes}")
# print(f"data.yaml created at: {data_yaml_path}")

# paths to the original datasets
datasets = {
    "dataset1": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\vans_v1i_yolov8",  # Vans
    "dataset2": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\Truck_v1i_yolov8",  # Trucks mixed
    "dataset3": r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\Objectdetection_v1i_yolov8",  # Object detection
}

"""
D-1   nc: 1 :-> names: ['Van']
D-2:  nc: 5 :-> names: ['Car', 'Motorcycle', 'Person', 'Truck', 'Van']
D-3:  nc: 6 :-> names: ['bike', 'bus', 'car', 'person', 'traffic signal', 'truck']
"""
import os
import shutil
import yaml
from collections import defaultdict

output_dir = "Merged_Dataset"

# final unified classes (exact order)
final_classes = ["car", "truck", "van"] # ID: car -> 0 ,   truck -> 1    , van ->  2

# logic mapping by original class name -> new ID
# each dataset's data.yaml to map old numeric IDs -> old names -> this dict
name_to_newid = {
    "car": 0,
    "Car": 0,
    "truck": 1,
    "Truck": 1,
    "van": 2,
    "Van": 2,
}

# splits to process
splits = ["train", "valid", "test"]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_dataset_names(ds_path):
    """
    Load names list from data.yaml in the given dataset path.
    Returns a list where index = old class id, value = class name string.
    If no data.yaml found or parse fails, returns None.
    """
    yaml_path = os.path.join(ds_path, "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        # names may be list or dict; handle both
        if isinstance(names, dict):
            # dict form maps id->name; convert to list by id order
            max_id = max(int(k) for k in names.keys())
            lst = [None] * (max_id + 1)
            for k, v in names.items():
                lst[int(k)] = v
            return lst
        elif isinstance(names, list):
            return names
        else:
            return None
    except Exception as e:
        print(f"Warning: failed to read YAML {yaml_path}: {e}")
        return None

def remap_label_lines(lines, old_names_list, dataset_key):
    """
    Given the content lines of a label file and the old names list (index=old_id),
    return a list of remapped lines (strings) where class ids have been converted
    to final unified IDs. Skip lines whose class names are not in name_to_newid.
    """
    new_lines = []
    for lp in lines:
        parts = lp.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
        except:
            continue
        if not old_names_list or old_id < 0 or old_id >= len(old_names_list):
            # we cannot map reliably if we don't have old_names_list
            # skip this line to be safe
            continue
        old_name = old_names_list[old_id]
        if old_name is None:
            continue
        # Map by name (case-sensitive checks included because YAML holds exact names)
        new_id = name_to_newid.get(old_name, None)
        if new_id is None:
            # Not a kept class according to user's logic -> skip
            continue
        # Keep bbox coords as-is
        coords = parts[1:5]
        new_lines.append(f"{new_id} {' '.join(coords)}\n")
    return new_lines

# prepare output dirs
ensure_dir(output_dir)
for split in splits:
    ensure_dir(os.path.join(output_dir, split, "images"))
    ensure_dir(os.path.join(output_dir, split, "labels"))

# merge process
# counters for verification
image_count_per_split = defaultdict(int)
box_count_per_split = {s: defaultdict(int) for s in splits}
image_files_seen_per_split = {s: set() for s in splits}  # to avoid duplicates if any

print("Starting merge process...")

for ds_key, ds_path in datasets.items():
    print(f"\nProcessing dataset: {ds_key} -> {ds_path}")
    # Load that dataset's YAML names list (old index -> name)
    old_names = load_dataset_names(ds_path)
    if old_names is None:
        # warn but continue; mapping will be best-effort using default naming
        print(f"  Warning: could not read data.yaml names for {ds_key}. Mapping will be attempted by best-effort (may skip many labels).")
    else:
        print(f"  Loaded {len(old_names)} class names from {os.path.join(ds_path,'data.yaml')}")

    for split in splits:
        src_images = os.path.join(ds_path, split, "images")
        src_labels = os.path.join(ds_path, split, "labels")
        if not os.path.isdir(src_images) or not os.path.isdir(src_labels):
            print(f"  Skipping split {split} for {ds_key} (missing folder).")
            continue

        dst_images = os.path.join(output_dir, split, "images")
        dst_labels = os.path.join(output_dir, split, "labels")

        for img_name in os.listdir(src_images):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_img_path = os.path.join(src_images, img_name)
            # rename image to avoid collisions
            dst_img_name = f"{ds_key}__{img_name}"
            dst_img_path = os.path.join(dst_images, dst_img_name)

            # copy image
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                print(f"    Failed to copy image {src_img_path}: {e}")
                continue

            # corresponding label
            label_base = os.path.splitext(img_name)[0] + ".txt"
            src_label_path = os.path.join(src_labels, label_base)
            dst_label_name = os.path.splitext(dst_img_name)[0] + ".txt"
            dst_label_path = os.path.join(dst_labels, dst_label_name)

            # default: create empty label file (so YOLO doesn't break)
            # We'll overwrite below if there are remapped lines
            open(dst_label_path, "w", encoding="utf-8").close()

            if not os.path.exists(src_label_path):
                # nothing to remap; empty file remains
                image_count_per_split[split] += 1
                image_files_seen_per_split[split].add(dst_img_name)
                continue

            # read original label lines
            try:
                with open(src_label_path, "r", encoding="utf-8") as f:
                    orig_lines = f.readlines()
            except Exception as e:
                print(f"    Warning: could not read label file {src_label_path}: {e}")
                orig_lines = []

            # remap using old_names list -> new ids
            new_lines = remap_label_lines(orig_lines, old_names, ds_key)

            # write remapped lines (if any); if none, leave file empty
            try:
                with open(dst_label_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            except Exception as e:
                print(f"    Warning: failed to write label file {dst_label_path}: {e}")
                continue

            # update counts
            image_count_per_split[split] += 1
            image_files_seen_per_split[split].add(dst_img_name)
            for ln in new_lines:
                parts = ln.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except:
                    continue
                box_count_per_split[split][cid] += 1

print("\nMerge finished. Now writing data.yaml for merged dataset...")

# Write final data.yaml
final_yaml_path = os.path.join(output_dir, "data.yaml")
final_nc = len(final_classes)
yaml_text = f"train: {os.path.join(output_dir, 'train', 'images')}\nval: {os.path.join(output_dir, 'valid', 'images')}\ntest: {os.path.join(output_dir, 'test', 'images')}\n\nnc: {final_nc}\nnames: {final_classes}\n"
with open(final_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_text)

# verification print

print(f"\nMerged dataset created at: {os.path.abspath(output_dir)}")
for split in splits:
    n_images = len(os.listdir(os.path.join(output_dir, split, "images")))
    total_boxes = sum(box_count_per_split[split].values())
    print(f"\nSplit: {split}")
    print(f"  Images copied: {n_images}")
    print(f"  Total label boxes (kept classes): {total_boxes}")
    for cid in range(len(final_classes)):
        print(f"  {final_classes[cid]} ({cid}): {box_count_per_split[split].get(cid,0)}")

print("\nDone.")
