"""
Champion selection script (EC2 MLflow).

- Compares challenger vs champion using MLflow metrics
- Promotes challenger if it wins majority of metrics
- Saves ONLY the champion model.pkl to S3:
  s3://mlops-creditcard/prod_outputs/champion_model/model.pkl
"""

import os
import pickle
import tempfile
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:////home/ssm-user/mlflow/mlflow.db"
MODEL_NAME = "creditcard-fraud-model"

S3_BUCKET = "mlops-creditcard"
S3_PREFIX = "prod_outputs/champion_model"

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
    if not challengers:
        return None
    return max(challengers, key=lambda x: int(x.version))


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

    if total == 0:
        return False

    return wins > (total / 2)


def export_champion_model_to_s3(model_version):
    """
    Loads champion model from MLflow and saves ONLY model.pkl to S3
    """
    print("📦 Exporting champion model to S3")

    model_uri = f"models:/{MODEL_NAME}/{model_version.version}"

    model = mlflow.sklearn.load_model(model_uri)

    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "model.pkl")

        with open(local_path, "wb") as f:
            pickle.dump(model, f)

        s3_key = f"{S3_PREFIX}/model.pkl"
        s3.upload_file(local_path, S3_BUCKET, s3_key)

    print(f"✅ Champion model saved to s3://{S3_BUCKET}/{s3_key}")

# -----------------------------
# Champion Selection Logic
# -----------------------------
def main():
    print("🚀 Starting Champion Selection")

    challenger = get_latest_challenger()
    if not challenger:
        print("❌ No challenger found")
        return

    champion = get_champion()
    challenger_metrics = get_metrics(challenger)

    # -------------------------
    # No champion exists
    # -------------------------
    if not champion:
        print("⚠️ No champion exists — promoting challenger")

        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

        export_champion_model_to_s3(challenger)
        return

    champion_metrics = get_metrics(champion)

    print("\n📊 Metric Comparison")
    for m in METRICS_TO_COMPARE:
        print(
            f"{m:<12} | "
            f"challenger={challenger_metrics.get(m)} | "
            f"champion={champion_metrics.get(m)}"
        )

    # -------------------------
    # Compare & Promote
    # -------------------------
    if challenger_wins(challenger_metrics, champion_metrics):
        print("\n🏆 Challenger wins — promoting")

        client.set_model_version_tag(MODEL_NAME, champion.version, "role", "archived")
        client.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")

        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

        export_champion_model_to_s3(challenger)

        print(f"✅ Challenger v{challenger.version} is now Champion")

    else:
        print("\n⚠️ Challenger did not outperform champion — no change")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
