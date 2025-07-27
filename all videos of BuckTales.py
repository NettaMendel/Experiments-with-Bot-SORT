# -*- coding: utf-8 -*-


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pandas as pd
import torch
import gc
from ultralytics import YOLO
import time
import numpy as np



# Load yolo11x_reid model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt").to(device)

fps_data=pd.DataFrame(columns=['fps_data'])
speed_data=pd.DataFrame(columns=['fps_data'])

# === CONFIGURATION ===
root_dir = r"C:\Users\User\Documents\vision\BuckTales\MOT_Dataset"
video_dir = os.path.join(root_dir, "videos")
gt_dir = r"C:\Users\User\Documents\vision\BuckTales\MOT_Dataset\MOT17_format\BB2023\BB2023-full"

fps_data = pd.DataFrame(columns=['fps_data'])

# Loop through each video file
for video_file in os.listdir(video_dir):
    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)
    video_output_path = os.path.join(root_dir,"output_videos",video_name)
    gt_path = os.path.join(gt_dir, video_name,"gt","gt.txt")
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)

    if not os.path.isfile(gt_path):
        print(f"Skipping {video_name}: missing GT file")
        continue

    print(f"Processing: {video_name}")

    # Create output directory
    output_dir = os.path.join(root_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    #annotated_video_path = os.path.join(video_output_path, "annotated_output_yolo11x_reid.mp4")
    #yolo_only_video_path = os.path.join(video_output_path, "yolo_only_output_yolo11x_reid.mp4")
    tracking_csv_path = os.path.join(video_output_path, "yolo_tracking_output_yolo11x_reid.csv")
    #"C:\Users\User\Documents\vision\BuckTales\MOT_Dataset\output_videos"

    # Step 2: Run YOLO tracking
    start_time = time.time()
    results = model.track(source=video_path, tracker='botsort.yaml' , save=False, save_txt=False, persist=True, stream=True)

    total_speed=0
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
            total_speed=total_speed+sum(speed.values())/1000
        frame_idx += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculete fps_runtime
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get number of frames
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    fps_runtime = num_frames / elapsed_time if elapsed_time > 0 else 0
    speed_data.loc[video_name]= num_frames / total_speed if total_speed > 0 else 0
    fps_data.loc[video_name]=fps_runtime

    # Save tracking results
    tracking_df = pd.DataFrame(tracking_data, columns=["frame", "id", "x", "y", "w", "h"])
    tracking_df.to_csv(tracking_csv_path, index=False)
    """
    # Load ground truth
    gt = pd.read_csv(gt_path, header=None)
    gt.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writer
    annotated_out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw GT
        gt_frame = gt[gt['frame'] == frame_idx]
        for _, row in gt_frame.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"GT:{int(row['id'])}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw YOLO
        yolo_frame = tracking_df[tracking_df['frame'] == frame_idx]
        for _, row in yolo_frame.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"YOLO:{int(row['id'])}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        annotated_out.write(frame)
        frame_idx += 1
        gc.collect()

    cap.release()
    annotated_out.release()

    # Step 4: Create YOLO-only video
    cap = cv2.VideoCapture(video_path)
    yolo_out = cv2.VideoWriter(yolo_only_video_path, fourcc, fps, (width, height))
    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_frame = tracking_df[tracking_df['frame'] == frame_idx]
        for _, row in yolo_frame.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"YOLO:{int(row['id'])}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        yolo_out.write(frame)
        frame_idx += 1
        gc.collect()

    cap.release()
    yolo_out.release()
   """
    print(f"✅ Completed: {video_file}")


FPS_csv_path = os.path.join(root_dir, "FPS_BuckTales_yolo11x_reid.csv")
pd.DataFrame(fps_data).to_csv(FPS_csv_path)

speed_csv_path = os.path.join(root_dir, "speed_BuckTales_yolo11x_reid.csv")
speed_data.to_csv(speed_csv_path)
