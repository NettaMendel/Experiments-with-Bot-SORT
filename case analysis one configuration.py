#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

import os
import cv2
import matplotlib.pyplot as plt


# In[3]:


# === 1. Load CSV files ===
video_name=''#######################
gt_df = pd.read_csv('gt.csv', header=0)       # Ground truth
yolo_df = pd.read_csv('yolo_tracking_output.csv', header=0)  # YOLO predictions

# === 2. Rename columns for clarity ===
gt_df.columns = ['frame', 'gt_id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
yolo_df.columns = ['frame', 'yolo_id', 'x', 'y', 'w', 'h']

# === 3. Make sure all numeric columns are really numeric ===
for df in [gt_df, yolo_df]:
    df[['frame', 'x', 'y', 'w', 'h']] = df[['frame', 'x', 'y', 'w', 'h']].apply(pd.to_numeric, errors='coerce')

# === 4. IoU calculation function ===
def compute_iou(boxA, boxB):
    # Convert (x, y, w, h) => (x1, y1, x2, y2)
    xA1, yA1, wA, hA = boxA
    xB1, yB1, wB, hB = boxB
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB2, yB2 = xB1 + wB, yB1 + hB

    # Compute intersection rectangle
    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Compute union area
    areaA = wA * hA
    areaB = wB * hB
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

# === 5. Match YOLO predictions to GT by frame and IoU ===
iou_threshold = 0.5
matches = []

# Go frame by frame
for frame in sorted(yolo_df['frame'].unique()):
    yolo_frame = yolo_df[yolo_df['frame'] == frame]
    gt_frame = gt_df[gt_df['frame'] == frame]

    # For each YOLO detection in the frame
    for _, yolo_row in yolo_frame.iterrows():
        best_iou = 0
        best_gt_id = None
        yolo_box = [yolo_row['x'], yolo_row['y'], yolo_row['w'], yolo_row['h']]

        # Compare to each GT box in the same frame
        for _, gt_row in gt_frame.iterrows():
            gt_box = [gt_row['x'], gt_row['y'], gt_row['w'], gt_row['h']]
            iou = compute_iou(yolo_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_row['gt_id']

        # Save if IoU is above threshold
        if best_iou >= iou_threshold:
            matches.append({
                'frame': frame,
                'yolo_id': yolo_row['yolo_id'],
                'gt_id': best_gt_id,
                'iou': best_iou
            })

# === 6. Convert to DataFrame and save or display ===
matches_df = pd.DataFrame(matches)
matches_df.to_csv('matched_results.csv', index=False, encoding='utf-8-sig', float_format='%.0f')

print(matches_df.head())
matches_df.to_csv('matches_df')


# In[4]:


def get_top_bad_frames_by_threshold(iou_threshold, top_n=10):
    # Filter all matches with IoU below the threshold
    bad_matches = matches_df[matches_df['iou'] < iou_threshold]

    # Count how many bad matches are in each frame
    frame_error_counts = bad_matches.groupby('frame').size().sort_values(ascending=False)

    # Select top N worst frames by number of bad matches
    top_bad_frame_ids = frame_error_counts.head(top_n).index.tolist()

    # Return the full match data for these frames
    return matches_df[matches_df['frame'].isin(top_bad_frame_ids)].sort_values(by=['frame', 'iou'])

# Choose your IoU threshold here
chosen_iou = 0.51

# Get the top 10 worst frames based on the chosen threshold
top_bad_frames_custom = get_top_bad_frames_by_threshold(chosen_iou, top_n=10)

# Display the results nicely in Jupyter
print(top_bad_frames_custom)

# Print just the frame IDs of the 10 worst frames
worst_frame_ids = top_bad_frames_custom['frame'].unique()
print("Worst 10 frames:", worst_frame_ids)


# In[5]:


# Loop over each of the top bad frames
for frame_id in worst_frame_ids:
    print(f"\n--- Frame {frame_id} ---")

    # Filter matches from this frame with IoU below the chosen threshold
    frame_data = matches_df[(matches_df['frame'] == frame_id) & (matches_df['iou'] < chosen_iou)]

    # Skip empty frames (just in case)
    if frame_data.empty:
        print("No bad matches in this frame.")
        continue

    # Store errors for this frame
    errors = []

    # Print info for each bad match
    for _, row in frame_data.iterrows():
        yolo_id = int(row['yolo_id'])
        gt_id = int(row['gt_id']) if not pd.isna(row['gt_id']) else "N/A"
        iou = row['iou']
        iou_diff = 1 - iou
        errors.append(iou_diff)

        print(f"YOLO ID {yolo_id} matched with GT ID {gt_id} → IoU = {iou:.3f}, error = {iou_diff:.3f}")

    # Compute stats for this frame
    avg_error = np.mean(errors)
    std_error = np.std(errors)

    print(f">>> Average error: {avg_error:.3f}")
    print(f">>> Std deviation : {std_error:.3f}")


# In[6]:


# Loop over each of the worst frames
for frame_id in worst_frame_ids:
    print(f"\n--- Frame {frame_id} ---")

    # Select YOLO and GT boxes for this frame
    yolo_frame = yolo_df[yolo_df['frame'] == frame_id]
    gt_frame = gt_df[gt_df['frame'] == frame_id]

    # Merge YOLO to GT by the matches we already calculated
    frame_matches = matches_df[(matches_df['frame'] == frame_id) & (matches_df['iou'] < chosen_iou)]

    if frame_matches.empty:
        print("No bad matches in this frame.")
        continue

    # List to store differences
    dx, dy, dw, dh = [], [], [], []

    for _, match in frame_matches.iterrows():
        # Find the matching YOLO and GT rows
        yolo_row = yolo_frame[yolo_frame['yolo_id'] == match['yolo_id']]
        gt_row = gt_frame[gt_frame['gt_id'] == match['gt_id']]

        if yolo_row.empty or gt_row.empty:
            continue  # skip if one of them is missing

        yolo_box = yolo_row.iloc[0]
        gt_box = gt_row.iloc[0]

        # Calculate the differences
        dx.append(abs(yolo_box['x'] - gt_box['x']))
        dy.append(abs(yolo_box['y'] - gt_box['y']))
        dw.append(abs(yolo_box['w'] - gt_box['w']))
        dh.append(abs(yolo_box['h'] - gt_box['h']))

        print(f"YOLO ID {int(yolo_box['yolo_id'])} vs GT ID {int(gt_box['gt_id'])}: "
              f"Δx={dx[-1]:.1f}, Δy={dy[-1]:.1f}, Δw={dw[-1]:.1f}, Δh={dh[-1]:.1f}")

    # Print statistics
    print(f">>> Δx mean: {np.mean(dx):.1f}, std: {np.std(dx):.1f}")
    print(f">>> Δy mean: {np.mean(dy):.1f}, std: {np.std(dy):.1f}")
    print(f">>> Δw mean: {np.mean(dw):.1f}, std: {np.std(dw):.1f}")
    print(f">>> Δh mean: {np.mean(dh):.1f}, std: {np.std(dh):.1f}")


# In[21]:


# Folder with extracted images
image_folder = 'C:/Users/hadar/Downloads/img1'

# Function to draw bounding boxes
def draw_boxes(image, boxes, color, label_prefix):
    for box in boxes:
        x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
        obj_id = int(box['id'])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label_prefix} {obj_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Loop over each bad frame
for frame_id in worst_frame_ids:
    filename = f"{frame_id:06d}.jpg"
    img_path = img_path = os.path.join(image_folder, filename).replace("\\", "/")

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # Load and prepare image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get GT and YOLO boxes for current frame
    gt_boxes = gt_df[gt_df['frame'] == frame_id][['x', 'y', 'w', 'h', 'gt_id']].rename(columns={'gt_id': 'id'})
    yolo_boxes = yolo_df[yolo_df['frame'] == frame_id][['x', 'y', 'w', 'h', 'yolo_id']].rename(columns={'yolo_id': 'id'})

    # 1. GT only
    gt_img = draw_boxes(img_rgb.copy(), gt_boxes.to_dict('records'), (0, 255, 0), 'GT')
    plt.imshow(gt_img)
    plt.title(f"Frame {frame_id} – Ground Truth")
    plt.axis('off')
    plt.show()

    # 2. YOLO only
    yolo_img = draw_boxes(img_rgb.copy(), yolo_boxes.to_dict('records'), (255, 0, 0), 'YOLO')
    plt.imshow(yolo_img)
    plt.title(f"Frame {frame_id} – YOLO Prediction")
    plt.axis('off')
    plt.show()

    # 3. Combined
    combined_img = draw_boxes(img_rgb.copy(), gt_boxes.to_dict('records'), (0, 255, 0), 'GT')
    combined_img = draw_boxes(combined_img, yolo_boxes.to_dict('records'), (255, 0, 0), 'YOLO')
    plt.imshow(combined_img)
    plt.title(f"Frame {frame_id} – GT (green) vs YOLO (red)")
    plt.axis('off')
    plt_name=f"{video_name}_Frame {frame_id} – GT (green) vs YOLO (red)"
    plt.show()


# In[25]:


# Loop through each of the worst frames
for frame_id in worst_frame_ids:
    print(f"\n📸 Comparison for Frame {frame_id}")

    # Get matches in this frame
    matched_frame = matches_df[matches_df['frame'] == frame_id]

    if matched_frame.empty:
        print("No matches in this frame.")
        continue

    # Get GT and YOLO boxes from the original full data
    gt_boxes = gt_df[gt_df['frame'] == frame_id]
    yolo_boxes = yolo_df[yolo_df['frame'] == frame_id]

    # Join with gt and yolo info
    matched = matched_frame.merge(gt_boxes.rename(columns={'id': 'gt_id'}), on=['frame', 'gt_id'], how='left')
    matched = matched.merge(yolo_boxes.rename(columns={'id': 'yolo_id'}), on=['frame', 'yolo_id'], how='left', suffixes=('_gt', '_yolo'))

    # Build comparison table
    comparison_rows = []
    for _, row in matched.iterrows():
        comparison_rows.append({
            'Object ID': row['gt_id'],
            'GT x': row['x_gt'], 'YOLO x': row['x_yolo'], 'Δx': round(row['x_yolo'] - row['x_gt'], 2),
            'GT y': row['y_gt'], 'YOLO y': row['y_yolo'], 'Δy': round(row['y_yolo'] - row['y_gt'], 2),
            'GT w': row['w_gt'], 'YOLO w': row['w_yolo'], 'Δw': round(row['w_yolo'] - row['w_gt'], 2),
            'GT h': row['h_gt'], 'YOLO h': row['h_yolo'], 'Δh': round(row['h_yolo'] - row['h_gt'], 2)
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df)
    comparison_df.to_csv(f'comparison_df {frame_id}')

# In[ ]:




