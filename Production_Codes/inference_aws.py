"""
inference_aws converted from notebook
Loads champion model from MLflow registry (preferred) or S3 fallback, runs batch predictions on S3 data,
and writes outputs back to S3. Also saves champion model artifacts to S3.
"""

import os
import json
import boto3
import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import tempfile
from pathlib import Path

# Configuration
# S3
S3_BUCKET = "mlops-creditcard"
INPUT_PREFIX = "inference/input"
OUTPUT_PREFIX = "inference/output"
MODEL_PREFIX = "inference/models"

# MLflow tracking detection (prefer env var, then SageMaker sqlite path used by register_model.ipynb, then local fallback)
tracking_env = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_env:
    MLFLOW_TRACKING_URI = tracking_env
else:
    sage_db = Path("/home/ec2-user/SageMaker/ML-Ops-CreditCard-AWS/mlflow.db")
    if sage_db.exists():
        MLFLOW_TRACKING_URI = f"sqlite:///{sage_db}"
    else:
        local_db = Path.cwd() / "mlflow.db"
        if local_db.exists():
            MLFLOW_TRACKING_URI = f"sqlite:///{local_db}"
        else:
            MLFLOW_TRACKING_URI = None  # no tracking URI detected

# Model name must match registration step
MODEL_NAME = "creditcard-fraud-model"

# Apply tracking URI if found
if MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(mlflow.get_tracking_uri())
        print("MLflow tracking URI set to:", mlflow.get_tracking_uri())
    except Exception as e:
        print("⚠️ Failed to set MLflow tracking URI:", e)

s3 = boto3.client("s3")


# Load Batch Inputs
def load_batch_input():
    obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key=f"{INPUT_PREFIX}/batch_input.csv"
    )
    df = pd.read_csv(obj["Body"])
    print(f"📥 Loaded batch input: {df.shape}")
    return df


# Get Champion Model (MLflow registry first; fallback to S3 pickle)
def get_champion_model():
    client = MlflowClient()

    # Try to find champion in MLflow registry
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print("⚠️ Could not query MLflow registry:", e)
        versions = []

    for v in versions:
        try:
            mv = client.get_model_version(MODEL_NAME, v.version)
            tags = mv.tags or {}
            if tags.get("status") == "production" and tags.get("role") == "champion":
                print(f"🏆 Champion model in registry: v{v.version}")
                model_uri = f"models:/{MODEL_NAME}/{v.version}"
                try:
                    # Attempt to load via MLflow (requires artifact store access)
                    model = mlflow.sklearn.load_model(model_uri)
                    return model, model_uri
                except Exception as load_err:
                    print("⚠️ Failed to load model from MLflow registry, will try S3 fallback:", load_err)
                    break
        except Exception:
            continue

    # S3 fallback: try to load a previously uploaded champion model pickle
    s3_key = f"{MODEL_PREFIX}/champion_model.pkl"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name
        print(f"Trying to download champion model from s3://{S3_BUCKET}/{s3_key} → {tmp_path}")
        s3.download_file(S3_BUCKET, s3_key, tmp_path)
        model = joblib.load(tmp_path)
        model_uri = f"s3://{S3_BUCKET}/{s3_key}"
        print(f"🏆 Champion model loaded from S3: {model_uri}")
        return model, model_uri
    except Exception as s3err:
        print("⚠️ S3 fallback failed:", s3err)

    raise Exception("❌ No champion model found in MLflow registry or S3")


# Generate Predictions
def generate_predictions(df, model):
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))

    features = df.drop(columns=["ID"] + (["CLASS"] if "CLASS" in df.columns else []))

    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = [None] * len(preds)

    df["PREDICTION"] = preds
    df["PREDICTION_PROB"] = probs
    return df


# Save Predictions to S3
def save_predictions_to_s3(df):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"{OUTPUT_PREFIX}/predictions_{ts}.csv"

    csv_buffer = df.to_csv(index=False)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=csv_buffer
    )

    print(f"📤 Predictions saved to s3://{S3_BUCKET}/{output_key}")


# Save champion model locally & upload to S3
def save_champion_model(model_uri):
    """Download model artifacts and upload a single .pkl of the model (if available) to S3.
    model_uri may be:
      - models:/<name>/<version>  (MLflow model registry)
      - runs:/<run_id>/model       (MLflow run artifact)
      - s3://bucket/path          (already on S3)
      - local file path
    """
    try:
        # If it's an MLflow registry or run URI, download artifacts
        if isinstance(model_uri, str) and (model_uri.startswith("models:/") or model_uri.startswith("runs:/")):
            print(f"Downloading artifacts for {model_uri}")
            local_dir = mlflow.artifacts.download_artifacts(model_uri=model_uri)
            # try to find a .pkl file inside
            pkl_files = list(Path(local_dir).rglob("*.pkl"))
            if pkl_files:
                model_file = str(pkl_files[0])
                upload_key = f"{MODEL_PREFIX}/champion_model.pkl"
                s3.upload_file(model_file, S3_BUCKET, upload_key)
                print(f"📦 Champion model uploaded to s3://{S3_BUCKET}/{upload_key}")
                return
            else:
                # If no pkl, zip the directory and upload
                import shutil
                zip_base = Path(tempfile.gettempdir()) / f"champion_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                shutil.make_archive(str(zip_base), 'zip', local_dir)
                zip_path = str(zip_base) + '.zip'
                upload_key = f"{MODEL_PREFIX}/champion_model_artifacts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
                s3.upload_file(zip_path, S3_BUCKET, upload_key)
                print(f"📦 Champion model artifacts uploaded to s3://{S3_BUCKET}/{upload_key}")
                return

        # If model_uri is an S3 path (already), we can copy it to consistent key or do nothing
        if isinstance(model_uri, str) and model_uri.startswith("s3://"):
            print(f"Model URI already points to S3: {model_uri}")
            return

        # If model_uri is a local path to a file, upload it
        if isinstance(model_uri, str) and os.path.exists(model_uri):
            upload_key = f"{MODEL_PREFIX}/champion_model.pkl"
            s3.upload_file(model_uri, S3_BUCKET, upload_key)
            print(f"📦 Champion model uploaded to s3://{S3_BUCKET}/{upload_key}")
            return

        print("⚠️ save_champion_model: unrecognized model_uri format, skipping upload")
    except Exception as e:
        print("❌ Failed to save champion model to S3:", e)


def main():
    print("🚀 AWS Batch Inference Started")

    batch_df = load_batch_input()
    model, model_uri = get_champion_model()
    preds_df = generate_predictions(batch_df, model)

    save_predictions_to_s3(preds_df)
    save_champion_model(model_uri)

    print("✅ AWS Batch Inference Completed")


if __name__ == "__main__":
    main()
