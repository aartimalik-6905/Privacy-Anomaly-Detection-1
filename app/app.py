import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import subprocess
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Privacy-Preserving Anomaly Detection",
    layout="centered"
)

st.title("🛡️ Privacy-Preserving Video Anomaly Detection")
st.write("Privacy-first human activity understanding using motion patterns, not identity.")

# --------------------------------------------------
# PATHS & MODEL
# --------------------------------------------------
SKELETON_PATH = "data/skeleton_csv/test.csv"
EXTRACT_SCRIPT = "src/extract_skeleton.py"

violence_model = joblib.load("models/violence_classifier.pkl")

# --------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------
def run_app():
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi"],
        help="Upload any video. Faces, clothing, and identity are never analyzed."
    )

    if not uploaded_video:
        return

    # 1️⃣ Save uploaded video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    os.environ["VIDEO_PATH"] = temp_video.name

    # 2️⃣ Run skeleton extraction
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
        else:
            st.error("Internal processing error")
            st.text(e.stderr)
            return

    # 3️⃣ Load skeleton CSV
    if not os.path.exists(SKELETON_PATH):
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    df = pd.read_csv(SKELETON_PATH)
    if df.empty:
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    # 4️⃣ Compute velocity & speed
    velocity = df.diff().fillna(0).values
    speed = np.linalg.norm(velocity, axis=1)

    # --------------------------------------------------
    # EARLY-EXIT GUARDS (NO SUCCESS HERE)
    # --------------------------------------------------

    # Edge Case 7: Very short video
    if len(speed) < 30:
        st.info("ℹ️ Insufficient data for analysis.")
        return

    # Edge Case 8: Low-light / blur
    

    # Edge Case 6: Camera motion / shake
    joint_velocity = velocity.reshape(-1, 3)
    motion_variance = np.var(np.linalg.norm(joint_velocity, axis=1))
    if motion_variance < 0.0005:
        st.info("ℹ️ Camera motion detected. Analysis skipped.")
        return

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------
    features = pd.DataFrame({
        "mean_speed": [speed.mean()],
        "std_speed": [speed.std()],
        "max_speed": [speed.max()],
        "std_acceleration": [np.diff(speed, prepend=0).std()],
        "motion_irregularity": [np.std(np.diff(speed))]
    })

    prediction = violence_model.predict(features)[0]

    if prediction == 1:
        st.markdown(
            "<div class='result-box violent'>🚨 Violent / Aggressive Behavior Detected</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box normal'>✅ Normal Activity</div>",
            unsafe_allow_html=True
        )

# --------------------------------------------------
# RUN APP
# --------------------------------------------------
run_app()