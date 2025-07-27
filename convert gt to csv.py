# -*- coding: utf-8 -*-

import os
import pandas as pd

# Define the base directory
base_dir = r'C:\Users\User\Documents\MOT17\MOT17\train'

# Define the column headers
#columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']

# Traverse each subfolder in the base directory
# For MOT17
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    gt_file_path = os.path.join(subfolder_path, 'gt', 'gt.txt')

    # Check if gt.txt exists
    if os.path.isfile(gt_file_path):
        # Read the gt.txt file into a DataFrame
        df = pd.read_csv(gt_file_path, header=None)
        #df.columns = columns
        # Define the output CSV path
        output_csv_path = os.path.join(subfolder_path, 'gt', 'gt.csv')

        # Save the DataFrame to CSV
        df.to_csv(output_csv_path, index=False, header=False)
        print(f"Created CSV for: {gt_file_path}")


