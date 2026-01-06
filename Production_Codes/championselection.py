"""
Champion selection script (EC2 MLflow).

- Compares challenger vs champion using MLflow metrics
- Promotes challenger if it wins majority of metrics
- ONLY IF PROMOTED:
    Uploads champion_model.pkl to:
    s3://mlops-creditcard/prod_outputs/champion_model/champion_model.pkl
"""

import os
import tempfile
import shutil
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:////home/ssm-user/mlflow/mlflow.db"
MODEL_NAME = "creditcard-fraud-model"

S3_BUCKET = "mlops-creditcard"
S3_KEY = "prod_outputs/champion_model/champion_model.pkl"

METRICS_TO_COMPARE = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(mlflow.get_tracking_uri())

client = MlflowClient()
s3 = boto3.client("s3")

print("✅ MLflow tracking URI:", mlflow.get_tracking_uri())

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


def upload_champion_model(model_version):
    """
    Downloads champion model from MLflow run artifacts
    and uploads model.pkl to S3
    """
    print("⬇️ Downloading champion model from MLflow run artifacts")

    tmp_dir = tempfile.mkdtemp()
    try:
        # Download ONLY the logged model artifact
        local_model_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{model_version.run_id}/model",
            dst_path=tmp_dir,
        )

        model_pkl = os.path.join(local_model_dir, "model.pkl")
        if not os.path.exists(model_pkl):
            raise FileNotFoundError(f"model.pkl not found at {model_pkl}")

        print(f"⬆️ Uploading champion model to s3://{S3_BUCKET}/{S3_KEY}")
        s3.upload_file(model_pkl, S3_BUCKET, S3_KEY)

        print("✅ Champion model uploaded to S3")

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
        print("❌ No models found in registry")
        return

    # No champion → auto promote challenger
    if not champion and challenger:
        print("⚠️ No champion exists — promoting challenger")

        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

        champion = challenger
        promoted = True

    # Champion exists → compare
    elif champion and challenger:
        challenger_metrics = get_metrics(challenger)
        champion_metrics = get_metrics(champion)

        print("\n📊 Metric Comparison")
        for m in METRICS_TO_COMPARE:
            print(
                f"{m:<12} | "
                f"challenger={challenger_metrics.get(m)} | "
                f"champion={champion_metrics.get(m)}"
            )

        if challenger_wins(challenger_metrics, champion_metrics):
            print("\n🏆 Challenger wins — promoting")

            client.set_model_version_tag(MODEL_NAME, champion.version, "role", "archived")
            client.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")

            client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
            client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

            champion = challenger
            promoted = True
        else:
            print("\n⚠️ Challenger did not outperform champion — no promotion")

    # -------------------------
    # Upload only if promoted
    # -------------------------
    if promoted:
        upload_champion_model(champion)
    else:
        print("ℹ️ No new champion — S3 model not updated")

    print("✅ Champion selection completed")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
