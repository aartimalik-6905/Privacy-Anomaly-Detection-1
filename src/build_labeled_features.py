import os
import pandas as pd
import numpy as np
import subprocess

FEATURES = []
LABELS = []

BASE_DIR = "data/raw_videos"
EXTRACT_SCRIPT = "src/extract_skeleton.py"

def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    velocity = df.diff().fillna(0).values
    speed = np.linalg.norm(velocity, axis=1)

    return {
        "mean_speed": speed.mean(),
        "std_speed": speed.std(),
        "max_speed": speed.max(),
        "std_acceleration": np.diff(speed, prepend=0).std(),
        "motion_irregularity": np.std(np.diff(speed))
    }

for label_name, label_value in [("normal", 0), ("violent", 1)]:
    folder = os.path.join(BASE_DIR, label_name)

    for video in os.listdir(folder):
        video_path = os.path.join(folder, video)
        os.environ["VIDEO_PATH"] = video_path

        try:
            subprocess.run(
                ["python", EXTRACT_SCRIPT],
                check=True,
                capture_output=True,
                text=True
            )
        except:
            continue

        csv_path = "data/skeleton_csv/test.csv"
        if not os.path.exists(csv_path):
            continue

        features = extract_features(csv_path)
        FEATURES.append(features)
        LABELS.append(label_value)

df = pd.DataFrame(FEATURES)
df["label"] = LABELS
df.to_csv("data/labeled_features.csv", index=False)

print("Labeled features created")
