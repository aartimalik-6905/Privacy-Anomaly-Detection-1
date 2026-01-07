import cv2
import mediapipe as mp
import pandas as pd
import os

from src.human_detector import count_human_frames


def extract_skeleton(video_path, output_path):
    """
    Extracts MediaPipe pose skeletons from a video.
    Raises RuntimeError("NO_HUMAN_DETECTED") if no valid human is found.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        cap = cv2.VideoCapture(video_path)
    MIN_POSE_FRAMES = 10        # baby-safe
    HOG_MIN_FRAMES = 10         # animal-safe
    FRAME_BUFFER_SIZE = 60

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize MediaPipe Pose
    pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

    cap = cv2.VideoCapture(video_path)

    rows = []
    frames_buffer = []
    pose_frames = 0
    frame_count = 0

    # ------------------------------
    # MAIN LOOP
    # ------------------------------
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

    # ------------------------------
    # HARD HUMAN GATES
    # ------------------------------
    if pose_frames < MIN_POSE_FRAMES:
        raise RuntimeError("NO_HUMAN_DETECTED")

    if not count_human_frames(frames_buffer, min_hits=HOG_MIN_FRAMES):
        raise RuntimeError("NO_HUMAN_DETECTED")

    # ------------------------------
    # SAVE CSV
    # ------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return output_path


# Allows standalone testing (optional)
if __name__ == "__main__":
    extract_skeleton(
        "data/raw_videos/normal_vdo.mp4",
        "data/skeleton_csv/test.csv"
    )
