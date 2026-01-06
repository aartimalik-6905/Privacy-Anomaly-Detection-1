# Privacy-Anomaly-Detection-1
Overview:
This project detects violent or aggressive human behavior in videos while preserving privacy by using skeletal motion data instead of raw visual features.

Key Features:

Skeleton-based analysis (no faces, no identity)

Robust human presence detection (blocks animals & objects)

Handles fast non-violent actions (running, dancing)

Supervised violence classification for real-world accuracy

Pipeline:

Human presence validation (MediaPipe + motion consistency)

Skeleton extraction

Motion feature computation

Supervised violence classification

Tech Stack:
Python

OpenCV

MediaPipe

NumPy, Pandas

Scikit-learn

Streamlit

Use Cases:
Privacy-first surveillance

Public safety analytics

Smart city monitoring

Ethical AI research

Limitations:
Context-free (no object or scene semantics)

Requires representative violent samples for training
