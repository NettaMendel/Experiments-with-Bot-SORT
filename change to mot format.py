# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:46:20 2025

@author: User
"""

import pandas as pd

# Load your CSV
df = pd.read_csv(r"C:\Users\User\Documents\vision\UrbanTracker\stmarc_video\stmarc_annotations\bounding_boxes_rouen.csv")

# Calculate width and height
df["w"] = df["x_bottom_right"] - df["x_top_left"]
df["h"] = df["y_bottom_right"] - df["y_top_left"]

# Reorder and rename columns to match MOT format
mot_df = pd.DataFrame({
    "frame": df["frame_number"],
    "id": df["object_id"],
    "x": df["x_top_left"],
    "y": df["y_top_left"],
    "w": df["w"],
    "h": df["h"],
    "conf": 1.0,     # or -1 for GT
    "class": df.get("class", 1),  # default to 1 if not in original CSV
    "vis": 1.0,
    "ignored": 0
})

# Sort by frame and id (optional but recommended)
mot_df = mot_df.sort_values(by=["frame", "id"])

# Save to txt
mot_df.to_csv(r"C:\Users\User\Documents\vision\UrbanTracker\stmarc_video\gt.txt", header=False, index=False)
