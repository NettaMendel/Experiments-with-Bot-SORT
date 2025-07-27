# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 06:15:31 2025

@author: User
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pandas as pd
import torch
import gc
from ultralytics import YOLO
import time

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt").to(device)

fps_data = pd.DataFrame(columns=['fps_data'])
speed_data = pd.DataFrame(columns=['fps_data'])

# === CONFIGURATION ===
root_dir = r"C:\Users\User\Documents\vision\UrbanTracker"

# Loop through each subfolder in root_dir
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Find video file in the folder
    video_file = next((f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".avi", ".mov"))), None)
    if not video_file:
        print(f"Skipping {folder_name}: no video file found")
        continue

    video_path = os.path.join(folder_path, video_file)
    gt_path = os.path.join(folder_path, "gt.txt")
    print(folder_path)
    if not os.path.isfile(gt_path):
        print(f"Skipping {folder_name}: missing GT file")
        continue

    print(f"Processing: {folder_name}")

    # Create output directory for this video
    tracking_csv_path = os.path.join(folder_path, "yolo_tracking_output_yolo11x_orb_reid.csv")

    # Run YOLO tracking
    start_time = time.time()
    results = model.track(source=video_path, tracker='botsort.yaml', save=False, save_txt=False, persist=True, stream=True)

    total_speed = 0
    tracking_data = []
    frame_idx = 0
    for result in results:
        boxes = result.boxes
        if boxes.id is not None:
            for i in range(len(boxes.id)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                tracking_data.append([frame_idx + 1, track_id, x1, y1, x2 - x1, y2 - y1])
        speed = result.speed
        if speed is not None:
            total_speed += sum(speed.values()) / 1000
        frame_idx += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps_runtime = num_frames / elapsed_time if elapsed_time > 0 else 0
    speed_data.loc[folder_name] = num_frames / total_speed if total_speed > 0 else 0
    fps_data.loc[folder_name] = fps_runtime

    # Save tracking results
    tracking_df = pd.DataFrame(tracking_data, columns=["frame", "id", "x", "y", "w", "h"])
    tracking_df.to_csv(tracking_csv_path, index=False)

    print(f"✅ Completed: {folder_name}")

# Save summary CSVs
fps_data.to_csv(os.path.join(root_dir, "FPS_BuckTales_yolo11x_orb_reid.csv"))
speed_data.to_csv(os.path.join(root_dir, "speed_BuckTales_yolo11x_orb_reid.csv"))
