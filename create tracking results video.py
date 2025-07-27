# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:10:18 2025

@author: User
"""

import cv2
import pandas as pd
import os

# Paths
initial_video_path = r"C:\Users\User\Documents\MOT17\MOT17\train\MOT17-02-DPM\initial_video.mp4"
tracking_csv_path = r"C:\Users\User\Documents\MOT17\MOT17\train\MOT17-02-DPM\yolo_tracking_output.csv"
yolo_only_video_path = r"C:\Users\User\Documents\MOT17\MOT17\train\MOT17-02-DPM\yolo_only_output.mp4"

# Load tracking data
tracking_df = pd.read_csv(tracking_csv_path)

# Open video
cap = cv2.VideoCapture(initial_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(yolo_only_video_path, fourcc, fps, (width, height))

frame_idx = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw YOLO tracking results
    yolo_frame = tracking_df[tracking_df['frame'] == frame_idx]
    for _, row in yolo_frame.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"YOLO:{int(row['id'])}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"YOLO-only tracking video saved to: {yolo_only_video_path}")
