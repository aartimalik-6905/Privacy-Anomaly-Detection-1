import joblib
import pandas as pd

model = joblib.load("models/anomaly_model.pkl")
X = pd.read_csv("data/features.csv")

prediction = model.predict(X)

if prediction[0] == -1:
    print("🚨 Anomaly Detected (Aggressive Behavior)")
else:
    print("✅ Normal Activity")
