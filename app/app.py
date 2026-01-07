import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import sys
import os
sys.path.append(os.path.abspath("."))



# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Privacy-Preserving Anomaly Detection",
    layout="centered"
)

st.title("🛡️ Privacy-Preserving Video Anomaly Detection")

st.markdown(
    """
<div style="
    padding:16px;
    border-radius:10px;
    border:1px solid rgba(255,255,255,0.15);
    background-color: rgba(240,242,246,0.04);
">
<h4>📹 How to Use This App</h4>
<ul>
<li>Upload a short video <b>containing people</b> (walking, talking, running, or fighting).</li>
<li>The system analyzes <b>only body movement patterns</b>, not faces or identity.</li>
<li>One video at a time is processed.</li>
</ul>
<b>Possible outcomes:</b>
<ul>
<li>✅ Normal Activity</li>
<li>🚨 Violent / Aggressive Behavior</li>
<li>ℹ️ No human detected</li>
<li>ℹ️ Insufficient data</li>
</ul>
</div>
""",
    unsafe_allow_html=True
)

# --------------------------------------------------
# PATHS & MODEL
# --------------------------------------------------
SKELETON_PATH = "data/skeleton_csv/test.csv"
violence_model = joblib.load("models/violence_classifier.pkl")

# --------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------
def run_app():
    uploaded_video = st.file_uploader(
        "Upload a video:",
        type=["mp4", "avi"],
        help="Upload any video. Faces, clothing, and identity are never analyzed."
    )

    if not uploaded_video:
        return

    # Save uploaded video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Ensure output directory exists (Cloud-safe)
    os.makedirs("data/skeleton_csv", exist_ok=True)

    # -------------------------------
    # Skeleton extraction
    # -------------------------------
    try:
        extract_skeleton(
            video_path=temp_video.name,
            output_path=SKELETON_PATH
        )

    except RuntimeError as e:
        if "NO_HUMAN_DETECTED" in str(e):
            st.info("ℹ️ No human detected. Analysis aborted.")
            return
        else:
            st.error("Internal processing error")
            st.text(str(e))
            return

    # -------------------------------
    # Load skeleton CSV
    # -------------------------------
    if not os.path.exists(SKELETON_PATH):
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    df = pd.read_csv(SKELETON_PATH)
    if df.empty:
        st.info("ℹ️ No human detected. Analysis aborted.")
        return

    # -------------------------------
    # Feature computation
    # -------------------------------
    velocity = df.diff().fillna(0).values
    speed = np.linalg.norm(velocity, axis=1)

    # Short video guard
    if len(speed) < 30:
        st.info("ℹ️ Insufficient data for analysis.")
        return

    # Camera motion guard
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

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = violence_model.predict(features)[0]

    if prediction == 1:
        st.error("🚨 Violent / Aggressive Behavior Detected")
    else:
        st.success("✅ Normal Activity")


# Footer
st.caption(
    "Privacy note: This system does not process faces, clothing, or personal identity."
)

# Run app
run_app()
