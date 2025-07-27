import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pandas as pd
import torch
import gc
from ultralytics import YOLO
import time
import numpy as np


# === CONFIGURATION ===
root_dir = r"C:\Users\User\Documents\MOT17\MOT17\train"

# Dictionary mapping folder names to their specific FPS
video_fps_map = {
    "MOT17-13-SDP":	25,
    "MOT17-11-SDP":	30,
    "MOT17-10-SDP":	30,
    "MOT17-09-SDP":	30,
    "MOT17-05-SDP": 14,
    "MOT17-04-SDP":	30,
    "MOT17-02-SDP":	30,
    "MOT17-13-FRCNN":	25,
    "MOT17-11-FRCNN":	30,
    "MOT17-10-FRCNN":	30,
    "MOT17-09-FRCNN":	30,
    "MOT17-05-FRCNN":	14,
    "MOT17-04-FRCNN":	30,
    "MOT17-02-FRCNN":	30,
    "MOT17-13-DPM":	25,
    "MOT17-11-DPM":	30,
    "MOT17-10-DPM":	30,
    "MOT17-09-DPM":	30,
    "MOT17-05-DPM":	14,
    "MOT17-04-DPM":	30,
    "MOT17-02-DPM":	30
}

# Load YOLOv8n model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt").to(device)

# Initialize metrics
#accumulators = {}
#summary_data = []

fps_data=pd.DataFrame(columns=['fps_data'])
speed_data=pd.DataFrame(columns=['fps_data'])

# Loop through each video folder
for video_folder, fps in video_fps_map.items():
    video_path = os.path.join(root_dir, video_folder)
    img_folder = os.path.join(video_path, "img1")
    gt_path = os.path.join(video_path, "gt", "gt.csv")

    if not os.path.isdir(img_folder) or not os.path.isfile(gt_path):
        print(f"Skipping {video_folder}: missing img1 or gt.csv")
        continue

    print(f"Processing: {video_folder} at {fps} FPS")

    # Output paths
    initial_video_path = os.path.join(video_path, "initial_video_yolo11x_orb_reid.mp4")
    #annotated_video_path = os.path.join(video_path, "annotated_output_yolov8n.mp4")
    #yolo_only_video_path = os.path.join(video_path, "yolo_only_output_yolov8n.mp4")
    tracking_csv_path = os.path.join(video_path, "yolo_tracking_output_yolo11x_orb_reid.csv")

    # Step 1: Convert image sequence to video
    images = sorted([img for img in os.listdir(img_folder) if img.endswith(".jpg")])
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(initial_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(img_folder, image)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()

    # Step 2: Run YOLO tracking
    start_time = time.time()
    results = model.track(source=initial_video_path, tracker='botsort.yaml', save=False, save_txt=False, persist=True, stream=True)


    tracking_data = []
    speed = []
    frame_idx = 0
    total_speed=0
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
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_frames = len(images)
    fps_runtime = total_frames / elapsed_time if elapsed_time > 0 else 0
    speed_data.loc[video_folder]= total_frames / total_speed if total_speed > 0 else 0
    fps_data.loc[video_folder]=fps_runtime
    
    # Save tracking results
    tracking_df = pd.DataFrame(tracking_data, columns=["frame", "id", "x", "y", "w", "h"])
    tracking_df.to_csv(tracking_csv_path, index=False)
    """
    # Load ground truth
    gt = pd.read_csv(gt_path, header=None)
    gt.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    #######gt = gt[['frame', 'id', 'x', 'y', 'w', 'h']]
    
    # Step 3: Create annotated video with GT + YOLO
    cap = cv2.VideoCapture(initial_video_path)
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
    cap = cv2.VideoCapture(initial_video_path)
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
   

    print(f"✅ Completed: {video_folder}")


FPS_csv_path = os.path.join(root_dir, "FPS_MOT17_yolo11x_orb_reid.csv")
speed_csv_path = os.path.join(root_dir, "speed_MOT17_yolo11x_orb_reid.csv")
fps_data.to_csv(FPS_csv_path)
speed_data.to_csv(speed_csv_path)

