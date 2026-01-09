"""
Champion selection script (EC2 MLflow + S3 copy + inference tar creation)

- Compares challenger vs champion using MLflow metrics
- Promotes challenger if it wins majority of metrics

ONLY IF PROMOTED:
1. Copies latest model.pkl to champion location
2. Creates inference_aws.tar.gz
3. Uploads inference package to S3
4. Updates monitoring reference training data
"""

import boto3
import shutil
import tempfile
from pathlib import Path
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:////home/ssm-user/mlflow/mlflow.db"
MODEL_NAME = "creditcard-fraud-model"

S3_BUCKET = "mlops-creditcard"

# MLflow artifacts
MLFLOW_MODELS_PREFIX = "prod_outputs/mlflow/models/"

# Champion locations
CHAMPION_MODEL_KEY = "prod_outputs/champion_model/champion_model.pkl"
INFERENCE_TAR_S3_KEY = "prod_codes/inference_aws.tar.gz"

# Training data (source → monitoring reference)
SOURCE_TRAINING_DATA_KEY = "data/raw/Training.csv"
MONITORING_TRAINING_DATA_KEY = (
    "monitoring_inputs/Training_Data/Training.csv"
)

METRICS_TO_COMPARE = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

# -----------------------------
# Clients
# -----------------------------
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
s3 = boto3.client("s3")

# -----------------------------
# Helper functions
# -----------------------------
def get_versions_by_tag(tag_key, tag_value):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matched = []

    for v in versions:
        mv = client.get_model_version(MODEL_NAME, v.version)
        if (mv.tags or {}).get(tag_key) == tag_value:
            matched.append(mv)

    return matched


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

    print("\n📊 Metric comparison:")

    for m in METRICS_TO_COMPARE:
        c = challenger_metrics.get(m)
        ch = champion_metrics.get(m)

        if c is None or ch is None:
            print(f"  ⚠️ {m}: skipped (missing metric)")
            continue

        total += 1

        if c > ch:
            wins += 1
            print(f"  ✅ {m}: challenger {c:.4f} > champion {ch:.4f}")
        else:
            print(f"  ❌ {m}: challenger {c:.4f} ≤ champion {ch:.4f}")

    print(f"\n📈 Result: challenger won {wins}/{total} metrics")

    return total > 0 and wins > total / 2


def find_latest_model_pkl_s3():
    paginator = s3.get_paginator("list_objects_v2")

    candidates = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=MLFLOW_MODELS_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("model.pkl"):
                candidates.append(obj)

    if not candidates:
        raise FileNotFoundError("No model.pkl found in MLflow S3 artifacts")

    return max(candidates, key=lambda x: x["LastModified"])["Key"]


def copy_model_to_champion_s3():
    latest_key = find_latest_model_pkl_s3()

    print(f"⬇️ Latest model: s3://{S3_BUCKET}/{latest_key}")
    print(f"⬆️ Champion target: s3://{S3_BUCKET}/{CHAMPION_MODEL_KEY}")

    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": latest_key},
        Key=CHAMPION_MODEL_KEY,
    )

    print("✅ Champion model updated")


def create_inference_tar():
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        shutil.copy("inference.py", tmp_dir / "inference.py")
        shutil.copy("__init__.py", tmp_dir / "__init__.py")

        s3.download_file(
            S3_BUCKET,
            CHAMPION_MODEL_KEY,
            str(tmp_dir / "champion_model.pkl"),
        )

        tar_base = tmp_dir.parent / "inference_aws"
        shutil.make_archive(str(tar_base), "gztar", tmp_dir)

        s3.upload_file(
            f"{tar_base}.tar.gz",
            S3_BUCKET,
            INFERENCE_TAR_S3_KEY,
        )

        print(f"📦 Uploaded inference package to s3://{S3_BUCKET}/{INFERENCE_TAR_S3_KEY}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def update_monitoring_training_data():
    print("📊 Updating monitoring reference training data")

    print(
        f"⬇️ Source: s3://{S3_BUCKET}/{SOURCE_TRAINING_DATA_KEY}\n"
        f"⬆️ Target: s3://{S3_BUCKET}/{MONITORING_TRAINING_DATA_KEY}"
    )

    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={
            "Bucket": S3_BUCKET,
            "Key": SOURCE_TRAINING_DATA_KEY,
        },
        Key=MONITORING_TRAINING_DATA_KEY,
    )

    print("✅ Monitoring training data updated")


# -----------------------------
# Main Logic
# -----------------------------
def main():
    print("🚀 Starting Champion Selection")

    challenger = get_latest_challenger()
    champion = get_champion()
    promoted = False

    if not challenger and not champion:
        print("❌ No models found")
        return

    if challenger and not champion:
        print("⚠️ No champion exists — promoting challenger")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")
        promoted = True

    elif challenger and champion:
        challenger_metrics = get_metrics(challenger)
        champion_metrics = get_metrics(champion)

        if challenger_wins(challenger_metrics, champion_metrics):
            print("🏆 Challenger promoted")

            client.set_model_version_tag(MODEL_NAME, champion.version, "role", "archived")
            client.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")

            client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
            client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

            promoted = True
        else:
            print("⚠️ Champion retained")

    if promoted:
        copy_model_to_champion_s3()
        create_inference_tar()
        update_monitoring_training_data()

    print(f"\nDEBUG → promoted={promoted}")
    print("✅ Champion selection completed")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
