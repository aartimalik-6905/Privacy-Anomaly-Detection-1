import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import subprocess
import os

st.set_page_config(
    page_title="Privacy-Preserving Anomaly Detection",
    layout="centered"
)

st.title("🛡️ Privacy-Preserving Video Anomaly Detection")
st.write("Privacy-first human activity understanding using motion patterns, not identity.")

SKELETON_PATH = "data/skeleton_csv/test.csv"
EXTRACT_SCRIPT = "src/extract_skeleton.py"

violence_model = joblib.load("models/violence_classifier.pkl")

st.markdown(
    """
    ### 📹 How to Use This App
    - Upload a **short video containing people** (walking, talking, running, or fighting).
    - The system analyzes **only body movement patterns**, not faces or identity.
    - Possible outcomes:
        - ✅ **Normal Activity**
        - 🚨 **Violent / Aggressive Behavior**
        - ℹ️ **No human detected**
        - ℹ️ **Insufficient data**
    """
)

def run_app():
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi"],
        help="Upload any video. Faces, clothing, and identity are never analyzed."
    )

    if not uploaded_video:
        return

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    os.environ["VIDEO_PATH"] = temp_video.name

    # ✅ REQUIRED FOR CLOUD
    os.makedirs("data/skeleton_csv", exist_ok=True)

    try:
        subprocess.run(
            ["python", EXTRACT_SCRIPT],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        if "NO_HUMAN_DETECTED" in e.stderr or "NO_HUMAN_DETECTED" in e.stdout:
            st.info("ℹ️ No human detected. Analysis aborted.")
            return
        st.error("Internal processing error")
        st.text(e.stderr)
        return

    if not os.path.exists(SKELETON_PATH):
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    df = pd.read_csv(SKELETON_PATH)
    if df.empty:
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    velocity = df.diff().fillna(0).values
    speed = np.linalg.norm(velocity, axis=1)

    if len(speed) < 30:
        st.info("ℹ️ Insufficient data for analysis.")
        return

    joint_velocity = velocity.reshape(-1, 3)
    motion_variance = np.var(np.linalg.norm(joint_velocity, axis=1))
    if motion_variance < 0.0005:
        st.info("ℹ️ Camera motion detected. Analysis skipped.")
        return

    features = pd.DataFrame({
        "mean_speed": [speed.mean()],
        "std_speed": [speed.std()],
        "max_speed": [speed.max()],
        "std_acceleration": [np.diff(speed, prepend=0).std()],
        "motion_irregularity": [np.std(np.diff(speed))]
    })

    prediction = violence_model.predict(features)[0]

    if prediction == 1:
        st.error("🚨 Violent / Aggressive Behavior Detected")
    else:
        st.success("✅ Normal Activity")

st.caption(
    "Privacy note: This system does not process faces, clothing, or personal identity."
)

run_app()
