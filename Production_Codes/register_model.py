"""
register_model converted from notebook
Logs metrics and registers the trained model with MLflow, tagging it as challenger.
"""

# Note: in the notebook a pip install cell was present. Ensure required packages are installed
# before running this script: mlflow, boto3, scikit-learn, joblib

import os
import json
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from mlflow.tracking import MlflowClient
import boto3

# Configuration
BASE_DIR = "./CREDITCARD/MODEL"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

S3_Bucket = "mlops-creditcard"
S3_ARTIFACT_ROOT = f"s3://{S3_Bucket}"

MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"
MLFLOW_MODEL_NAME = "creditcard-fraud-model"

# Local backend Configuration (SQLite for persistence)
db_path = "/home/ec2-user/SageMaker/ML-Ops-CreditCard-AWS/mlflow.db"
os.makedirs(os.path.dirname(db_path), exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

s3 = boto3.client("s3")

try:
    s3.list_objects_v2(Bucket=S3_Bucket, MaxKeys=1)
    print("✅ S3 bucket accessible")
except Exception as e:
    print("❌ Cannot access S3 bucket:", e)

if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=f"s3://{S3_Bucket}/artifacts")

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model not found at {MODEL_PATH}")

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"metrics not found at {METRICS_PATH}")

print("✅ Artifacts validated")

mlflow.set_registry_uri(mlflow.get_tracking_uri())

# Load model & metrics (explicit disk load)
model = joblib.load(MODEL_PATH)
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

print("✅ Loaded model & metrics")
print(metrics)

run_name = f"run_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id

    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Log model using MLflow's sklearn logger
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log metrics.json as artifact
    mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")

    print("✅ Artifacts uploaded to S3")
    print("Run ID:", run_id)

MODEL_URI = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=MODEL_URI,
    name=MLFLOW_MODEL_NAME
)

print("✅ Model registered")
print("Version:", result.version)

# Verify experiment run
client = MlflowClient()
run = client.get_run(run_id)
print("📌 Run info")
print("Run ID:", run.info.run_id)
print("Experiment ID:", run.info.experiment_id)
print("Metrics:", run.data.metrics)
print("Tags:", run.data.tags)

# Verify model registry and tag the new version as challenger
registered_models = client.search_registered_models()
for model_item in registered_models:
    print(f"\n📦 Model: {model_item.name}")
    for v in model_item.latest_versions:
        print(f"   └── Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")

client.set_model_version_tag(
    name=MLFLOW_MODEL_NAME,
    version=result.version,
    key="role",
    value="challenger"
)

client.set_model_version_tag(
    name=MLFLOW_MODEL_NAME,
    version=result.version,
    key="status",
    value="staging"
)

print("🏷️ Model tagged as challenger")
