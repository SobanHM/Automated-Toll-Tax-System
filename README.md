# Automated Toll Tax Estimation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time **AI-powered Traffic Monitoring System** that detects vehicles, tracks their movement, and automatically calculates toll tax revenue based on vehicle type. Built using **YOLOv8**, **DeepSORT**, and **OpenCV**.

---

## This project contains 02 Modules.
## Module 01 (Using Pretrained model Yolov8 and DeepSorting for vehicle tracking to avoid toll tax re-charge across frames)
## Demo 
<img width="1782" height="859" alt="image" src="https://github.com/user-attachments/assets/51a3112e-91ed-41ba-8b50-193bed8f9eca" />


> ## Status: Module 1 Complete (Pre-trained Baseline).
> 
> ## Module 2 (CNN/Transformer/Fine-tuning) In Progress.



## 🚀 Features
- 🚗 Vehicle Detection: Classifies Cars, Trucks, Buses, and Motorcycles in real-time.
- 📍 Object Tracking: Uses DeepSORT/ByteTrack to assign unique IDs to vehicles (prevents double counting).
- 💰 Automatic Billing: Calculates total revenue dynamically based on a configurable price list.
- 📊 Live Dashboard: Displays vehicle counts and total revenue overlay on the video feed.
- 📝 Summary Report: Generates a text summary of total traffic volume and tax collected upon exit.

---

## 🛠️ Tech Stack
- Language: Python
- torch, openCV, keras, shutil, os, ...
- Detection Model: YOLOv8 via `ultralytics`
- Tracking: `deep-sort-realtime` / `ByteTrack`
- Computer Vision: OpenCV (`cv2`)

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/SobanHM/Automated-Toll-Tax-System.git](https://github.com/SobanHM/Automated-Toll-Tax-System.git)
   cd Automated-Toll-Tax-System

