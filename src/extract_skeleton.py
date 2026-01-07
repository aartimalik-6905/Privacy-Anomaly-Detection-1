import cv2
import mediapipe as mp
import pandas as pd
import os

from human_detector import count_human_frames

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = os.environ.get("VIDEO_PATH", "data/raw_videos/test.mp4")
OUTPUT_PATH = "data/skeleton_csv/test.csv"

MIN_POSE_FRAMES = 10        # works for babies
HOG_MIN_FRAMES = 10         # blocks animals
FRAME_BUFFER_SIZE = 60

os.makedirs("data/skeleton_csv", exist_ok=True)

# ==============================
# INITIALIZE
# ==============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

cap = cv2.VideoCapture(VIDEO_PATH)

rows = []
frames_buffer = []
pose_frames = 0
frame_count = 0

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if len(frames_buffer) < FRAME_BUFFER_SIZE:
        frames_buffer.append(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        pose_frames += 1

        row = []
        for lm in result.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        rows.append(row)

cap.release()

print("Frames read:", frame_count)
print("Pose frames:", pose_frames)

# ==============================
# HARD HUMAN GATES
# ==============================

# Gate 1: Pose must exist over time (baby-safe)
if pose_frames < MIN_POSE_FRAMES:
    raise RuntimeError("NO_HUMAN_DETECTED")

# Gate 2: Independent human detector (animal-safe)
if not count_human_frames(frames_buffer, min_hits=HOG_MIN_FRAMES):
    raise RuntimeError("NO_HUMAN_DETECTED")

# ==============================
# SAVE
# ==============================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print("Skeleton CSV created:", OUTPUT_PATH)
