from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

X = pd.read_csv("data/features.csv")

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

joblib.dump(model, "models/anomaly_model.pkl")
