"""
Champion selection script (EC2 MLflow + S3 copy)

- Compares challenger vs champion using MLflow metrics
- Promotes challenger if it wins majority of metrics
- ONLY IF PROMOTED:
    Finds latest model.pkl in S3 (prod_outputs/mlflow/models/)
    and copies it to:
    s3://mlops-creditcard/prod_outputs/champion_model/champion_model.pkl
"""

import boto3
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:////home/ssm-user/mlflow/mlflow.db"
MODEL_NAME = "creditcard-fraud-model"

S3_BUCKET = "mlops-creditcard"
MLFLOW_MODELS_PREFIX = "prod_outputs/mlflow/models/"
CHAMPION_KEY = "prod_outputs/champion_model/champion_model.pkl"

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
    """
    Loops through S3 under prod_outputs/mlflow/models/ and finds the latest model.pkl
    """
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=MLFLOW_MODELS_PREFIX)

    candidates = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("model.pkl"):
                candidates.append({"Key": key, "LastModified": obj["LastModified"]})

    if not candidates:
        raise FileNotFoundError("No model.pkl found in S3 under mlflow/models/")

    latest = max(candidates, key=lambda x: x["LastModified"])
    return latest["Key"]


def copy_model_to_champion_s3():
    """
    Copies latest model.pkl from mlflow/models/ to champion_model location
    """
    latest_key = find_latest_model_pkl_s3()
    print(f"⬇️ Latest model.pkl in S3: s3://{S3_BUCKET}/{latest_key}")
    print(f"⬆️ Copying to s3://{S3_BUCKET}/{CHAMPION_KEY}")

    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": latest_key},
        Key=CHAMPION_KEY
    )
    print("✅ Champion model updated successfully")


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

    # No champion → promote challenger
    if not champion and challenger:
        print("⚠️ No champion exists — promoting challenger")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")
        champion = challenger
        promoted = True

    # Champion exists → compare metrics
    elif champion and challenger:
        challenger_metrics = get_metrics(challenger)
        champion_metrics = get_metrics(champion)

        print("\n📊 Metric Comparison")
        for m in METRICS_TO_COMPARE:
            print(f"{m:<12} | challenger={challenger_metrics.get(m)} | champion={champion_metrics.get(m)}")

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

    print(f"DEBUG → promoted={promoted}, champion_version={champion.version if champion else None}")

    # -------------------------
    # Copy latest model.pkl from S3 only if promoted
    # -------------------------
    if promoted:
        copy_model_to_champion_s3()
    else:
        print("ℹ️ No new champion — S3 model not updated")

    print("✅ Champion selection completed")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
