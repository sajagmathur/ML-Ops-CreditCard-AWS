"""
Champion selection script (EC2 MLflow + S3 copy + inference tar creation)

- Compares challenger vs champion using MLflow metrics
- Promotes challenger if it wins majority of metrics
- ONLY IF PROMOTED:
    Finds latest model.pkl in S3 (prod_outputs/mlflow/models/)
    Copies it to:
        s3://mlops-creditcard/prod_outputs/champion_model/champion_model.pkl
    Creates inference_aws.tar.gz containing:
        - code/inference.py
        - champion_model.pkl
    Uploads tar.gz to:
        s3://mlops-creditcard/prod_codes/inference_aws.tar.gz
"""

import boto3
from mlflow.tracking import MlflowClient
import shutil
import tempfile
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:////home/ssm-user/mlflow/mlflow.db"
MODEL_NAME = "creditcard-fraud-model"

S3_BUCKET = "mlops-creditcard"
MLFLOW_MODELS_PREFIX = "prod_outputs/mlflow/models/"
CHAMPION_KEY = "prod_outputs/champion_model/champion_model.pkl"
INFERENCE_TAR_S3_KEY = "prod_codes/inference_aws.tar.gz"

METRICS_TO_COMPARE = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

# -----------------------------
# MLflow + S3 Setup
# -----------------------------
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
s3 = boto3.client("s3")

# -----------------------------
# Helper functions
# -----------------------------
def get_versions_by_tag(tag_key, tag_value):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    result = []
    for v in versions:
        mv = client.get_model_version(MODEL_NAME, v.version)
        if (mv.tags or {}).get(tag_key) == tag_value:
            result.append(mv)
    return result


def get_latest_challenger():
    challengers = get_versions_by_tag("role", "challenger")
    return max(challengers, key=lambda x: int(x.version)) if challengers else None


def get_champion():
    champs = get_versions_by_tag("role", "champion")
    return champs[0] if champs else None


def get_metrics(model_version):
    run = client.get_run(model_version.run_id)
    return run.data.metrics


def challenger_wins(challenger_metrics, champion_metrics):
    wins = 0
    total = 0
    for m in METRICS_TO_COMPARE:
        c = challenger_metrics.get(m)
        ch = champion_metrics.get(m)
        if c is None or ch is None:
            continue
        total += 1
        if c > ch:
            wins += 1
    return total > 0 and wins > (total / 2)


def find_latest_model_pkl_s3():
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=MLFLOW_MODELS_PREFIX)

    candidates = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("model.pkl"):
                candidates.append(
                    {"Key": obj["Key"], "LastModified": obj["LastModified"]}
                )

    if not candidates:
        raise FileNotFoundError("No model.pkl found in mlflow/models")

    return max(candidates, key=lambda x: x["LastModified"])["Key"]


def copy_model_to_champion_s3():
    latest_key = find_latest_model_pkl_s3()
    print(f"⬇️ Latest model.pkl: s3://{S3_BUCKET}/{latest_key}")
    print(f"⬆️ Copying to: s3://{S3_BUCKET}/{CHAMPION_KEY}")

    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": latest_key},
        Key=CHAMPION_KEY
    )
    print("✅ Champion model updated")


def create_inference_tar():
    """
    Creates inference_aws.tar.gz containing:
      - code/inference.py
      - champion_model.pkl
    """
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # Create code/ directory
        code_dir = tmp_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Copy inference.py → code/inference.py
        shutil.copy("inference.py", code_dir / "inference.py")

        # Download champion_model.pkl → root
        s3.download_file(
            S3_BUCKET,
            CHAMPION_KEY,
            str(tmp_dir / "champion_model.pkl")
        )

        # Create tar.gz
        tar_base = tmp_dir.parent / "inference_aws"
        shutil.make_archive(str(tar_base), "gztar", tmp_dir)

        # Upload tar.gz to S3
        s3.upload_file(
            str(tar_base) + ".tar.gz",
            S3_BUCKET,
            INFERENCE_TAR_S3_KEY
        )

        print(f"📦 Uploaded s3://{S3_BUCKET}/{INFERENCE_TAR_S3_KEY}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------
# Champion Selection Logic
# -----------------------------
def main():
    print("🚀 Starting Champion Selection")

    challenger = get_latest_challenger()
    champion = get_champion()
    promoted = False

    if not challenger and not champion:
        print("❌ No models found")
        return

    if not champion and challenger:
        print("⚠️ No champion — promoting challenger")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")
        promoted = True

    elif champion and challenger:
        challenger_metrics = get_metrics(challenger)
        champion_metrics = get_metrics(champion)

        if challenger_wins(challenger_metrics, champion_metrics):
            print("🏆 Challenger wins — promoting")
            client.set_model_version_tag(MODEL_NAME, champion.version, "role", "archived")
            client.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")
            client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
            client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")
            promoted = True
        else:
            print("⚠️ No promotion")

    if promoted:
        copy_model_to_champion_s3()
        create_inference_tar()

    print(f"DEBUG → promoted={promoted}")
    print("✅ Champion selection completed")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
