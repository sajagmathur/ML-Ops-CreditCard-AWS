"""
inference.py
AWS SageMaker Batch Transform Inference Script

- Loads champion_model.pkl from /opt/ml/model
- Reads CSV input from Batch Transform
- Outputs CSV predictions
"""

import os
import joblib
import pandas as pd
from io import StringIO

MODEL_FILENAME = "champion_model.pkl"

# -------------------------------------------------
# Load model (called once per container)
# -------------------------------------------------
def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    model = joblib.load(model_path)
    print(f"🏆 Model loaded from {model_path}")
    return model

# -------------------------------------------------
# Input parsing
# -------------------------------------------------
def input_fn(request_body, request_content_type):
    if request_content_type != "text/csv":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    df = pd.read_csv(StringIO(request_body))
    print(f"📥 Batch input shape: {df.shape}")
    return df

# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
def predict_fn(input_data, model):
    df = input_data.copy()

    # Ensure ID column
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))

    # Drop non-feature columns
    drop_cols = ["ID"]
    if "CLASS" in df.columns:
        drop_cols.append("CLASS")

    features = df.drop(columns=drop_cols)

    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = [None] * len(preds)

    df["PREDICTION"] = preds
    df["PREDICTION_PROB"] = probs

    return df

# -------------------------------------------------
# Output formatting
# -------------------------------------------------
def output_fn(prediction_output, accept):
    # Force CSV output for Batch Transform
    return prediction_output.to_csv(index=False), "text/csv"
