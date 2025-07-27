import sqlite3
import pandas as pd
import os

# Your paths
input_path = r"C:\Users\User\Documents\vision\UrbanTracker\stmarc_video\stmarc_annotations\stmarc_annotations\stmarc_gt.sqlite"
output_path = r"C:\Users\User\Documents\vision\UrbanTracker\stmarc_video\stmarc_annotations"

# Connect to the database
conn = sqlite3.connect(input_path)

# Get all table names
table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()

# Export each table
for table_name in table_names:
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    except Exception as e:
        print(f"Error reading {table_name}: {e}")
        continue

    # Attempt to fix encoding issues (manually re-encode problematic columns)
    for col in df.select_dtypes(include=['object']):
        try:
            df[col] = df[col].apply(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)
        except Exception as e:
            print(f"Encoding issue in column {col} of table {table_name}: {e}")

    # Save to CSV
    csv_file = os.path.join(output_path, f"{table_name}_rouen.csv")
    df.to_csv(csv_file, index=False)
    print(f"Exported {table_name} to {csv_file}")

conn.close()
