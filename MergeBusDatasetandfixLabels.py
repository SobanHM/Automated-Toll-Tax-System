import os

bus_dataset_path = r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Vehicles_Dataset\Buses_final_v1i_yolov8"

label_folders = [
    os.path.join(bus_dataset_path, "train", "labels"),
    os.path.join(bus_dataset_path, "valid", "labels"),
    os.path.join(bus_dataset_path, "test", "labels")
]

OLD_CLASS = "0"   # bus dataset uses class 0
NEW_CLASS = "3"   # final unified bus class

for folder in label_folders:
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()

                # change ONLY class index
                parts[0] = NEW_CLASS

                new_lines.append(" ".join(parts) + "\n")

            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("all bus labels successfully remapped from class 0 → 3")
