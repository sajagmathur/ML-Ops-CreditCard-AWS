"""
inference_aws.py
AWS SageMaker Batch Transform Inference Script

- Loads champion model from S3
- Runs batch predictions on input CSV from S3
- Saves output predictions back to S3
"""

import boto3
import joblib
import pandas as pd
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
S3_BUCKET = "mlops-creditcard"
INPUT_KEY = "prod_inputs/batch_input.csv"                     # Input file
OUTPUT_PREFIX = "prod_outputs/predictions"                    # Output folder
MODEL_S3_KEY = "prod_outputs/champion_model/champion_model.pkl"  # Champion model path

# Initialize S3 client
s3 = boto3.client("s3")

# -----------------------------
# Load batch input
# -----------------------------
def load_batch_input():
    obj = s3.get_object(Bucket=S3_BUCKET, Key=INPUT_KEY)
    df = pd.read_csv(obj["Body"])
    print(f"📥 Loaded batch input: {df.shape}")
    return df

# -----------------------------
# Load champion model from S3
# -----------------------------
def load_champion_model():
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    s3.download_file(S3_BUCKET, MODEL_S3_KEY, tmp_file.name)
    model = joblib.load(tmp_file.name)
    print(f"🏆 Loaded champion model from s3://{S3_BUCKET}/{MODEL_S3_KEY}")
    return model

# -----------------------------
# Generate predictions
# -----------------------------
def generate_predictions(df, model):
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))

    features = df.drop(columns=["ID"] + (["CLASS"] if "CLASS" in df.columns else []))

    preds = model.predict(features)
    probs = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)

    df["PREDICTION"] = preds
    df["PREDICTION_PROB"] = probs
    return df

# -----------------------------
# Save predictions to S3
# -----------------------------
def save_predictions_to_s3(df):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"{OUTPUT_PREFIX}/predictions_{ts}.csv"
    s3.put_object(Bucket=S3_BUCKET, Key=output_key, Body=df.to_csv(index=False))
    print(f"📤 Predictions saved to s3://{S3_BUCKET}/{output_key}")

# -----------------------------
# Main entrypoint
# -----------------------------
def main():
    print("🚀 AWS Batch Transform Inference Started")

    df = load_batch_input()
    model = load_champion_model()
    predictions_df = generate_predictions(df, model)
    save_predictions_to_s3(predictions_df)

    print("✅ AWS Batch Transform Inference Completed")

if __name__ == "__main__":
    main()
