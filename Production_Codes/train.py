"""
SageMaker-compatible training script
Reads data from S3-mounted path
Writes model + metrics to S3 output path
"""

import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# SageMaker paths (DO NOT CHANGE)
# -----------------------------
INPUT_DIR = "/opt/ml/input/data/train"
OUTPUT_DIR = "/opt/ml/model"

# -----------------------------
# Load data
# -----------------------------
data_path = os.path.join(INPUT_DIR, "Training.csv")
data = pd.read_csv(data_path)

X = data.drop("Class", axis=1)
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(x_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

print("📊 Metrics:", metrics)

# -----------------------------
# Save model artifacts
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print("✅ Model and metrics saved")
