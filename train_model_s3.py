"""
train_model_s3 converted from notebook
Trains a RandomForest on data downloaded from S3 and saves model+metrics locally.
"""

import os
import boto3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------
S3_BUCKET = "mlops-creditcard"               # Your S3 bucket
S3_KEY = "data/raw/Training.csv"                      # File path in bucket
LOCAL_DIR = "./CREDITCARD/MODEL"            # Local folder structure
os.makedirs(LOCAL_DIR, exist_ok=True)

# -----------------------------
# Download CSV from S3
# -----------------------------
s3 = boto3.client('s3')
local_csv = os.path.join(LOCAL_DIR, "Training.csv")
s3.download_file(S3_BUCKET, S3_KEY, local_csv)

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv(local_csv)
X = data.drop(['Class'], axis=1)
y = data['Class']

# Train-test split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(xTrain, yTrain)

# -----------------------------
# Evaluate Model
# -----------------------------
yPred = model.predict(xTest)
metrics = {
    'Accuracy': accuracy_score(yTest, yPred),
    'Precision': precision_score(yTest, yPred),
    'Recall': recall_score(yTest, yPred),
    'F1 Score': f1_score(yTest, yPred),
}

print("Model Metrics:", metrics)

# -----------------------------
# Save Model and Metrics
# -----------------------------
model_path = os.path.join(LOCAL_DIR, "model.pkl")
metrics_path = os.path.join(LOCAL_DIR, "metrics.json")

joblib.dump(model, model_path)
pd.Series(metrics).to_json(metrics_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Metrics saved to {metrics_path}")
