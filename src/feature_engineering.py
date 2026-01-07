import pandas as pd
import numpy as np
import os

INPUT_DIR = "data/skeleton_csv"
OUTPUT_FILE = "data/features.csv"

features_list = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(INPUT_DIR, file))
    if df.empty:
        continue

    # Velocity
    velocity = df.diff().fillna(0).values
    speed = np.linalg.norm(velocity, axis=1)

    # Acceleration
    acceleration = np.diff(speed, prepend=0)

    # Regularity (periodic motion = low irregularity)
    speed_diff = np.diff(speed)
    motion_irregularity = np.std(speed_diff)

    features = {
        "mean_speed": speed.mean(),
        "std_speed": speed.std(),
        "max_speed": speed.max(),
        "std_acceleration": acceleration.std(),
        "motion_irregularity": motion_irregularity
    }

    features_list.append(features)

pd.DataFrame(features_list).to_csv(OUTPUT_FILE, index=False)
print("features.csv created")
