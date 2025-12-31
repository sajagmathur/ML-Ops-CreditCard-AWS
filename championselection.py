"""
Champion selection script converted from notebook.
This selects a challenger model from MLflow registry and promotes it to champion
if it outperforms the current champion on a majority of configured metrics.
"""

import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Determine MLflow tracking URI (env override -> SageMaker default -> local sqlite)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    sage_db = Path("/home/ec2-user/SageMaker/ML-Ops-CreditCard-AWS/mlflow.db")
    if sage_db.exists():
        tracking_uri = f"sqlite:///{sage_db}"
    else:
        local_db = Path.cwd() / "mlflow.db"
        if local_db.exists():
            tracking_uri = f"sqlite:///{local_db}"
        else:
            # Fallback: create local sqlite file and use it (this avoids needing a running mlflow server)
            db_path = local_db
            tracking_uri = f"sqlite:///{db_path}"
            print("⚠️ No existing MLflow DB found. Using local sqlite at", db_path)
            print("If you intended to use a remote MLflow server, set the MLFLOW_TRACKING_URI environment variable or start an MLflow server.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(mlflow.get_tracking_uri())

client = MlflowClient()

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Configuration
MODEL_NAME = "creditcard-fraud-model"  # must match name used in registration

METRICS_TO_COMPARE = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]


# Utility Functions (Registry & Metrics)
def get_model_version_by_tag(client, model_name, tag_key, tag_value):
    """Return the MLflow Model Version object that matches a tag value, or None."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        try:
            mv = client.get_model_version(name=model_name, version=v.version)
            tags = mv.tags or {}
            if tags.get(tag_key) == tag_value:
                return mv
        except Exception:
            # ignore and continue
            continue
    return None


def get_model_version_metrics(client, model_name, version):
    mv = client.get_model_version(name=model_name, version=version)
    run = client.get_run(mv.run_id)
    return run.data.metrics


# Deterministic challenger selection (highest-version)
def get_all_model_versions_by_tag(client, model_name, tag_key, tag_value):
    """Return a list of ModelVersion objects that match a tag."""
    versions = client.search_model_versions(f"name='{model_name}'")
    result = []
    for v in versions:
        try:
            mv = client.get_model_version(name=model_name, version=v.version)
            if (mv.tags or {}).get(tag_key) == tag_value:
                result.append(mv)
        except Exception:
            continue
    return result


def choose_challenger_by_highest_version(client, model_name):
    """Choose the challenger with the highest numeric version.

    Falls back to lexical comparison if versions are not strictly numeric.
    """
    challengers = get_all_model_versions_by_tag(client, model_name, 'role', 'challenger')
    if not challengers:
        return None

    def ver_key(mv):
        try:
            return int(mv.version)
        except Exception:
            return mv.version  # lexical fallback

    best = max(challengers, key=ver_key)
    return best


# Metric Comparison Logic
def better_than(challenger_metrics, champion_metrics):
    """Return True if challenger beats champion on a strict majority of compared metrics.
    Only metrics present in both runs are considered; ties do not count for challenger.
    """
    challenger_wins = 0
    total_considered = 0

    for metric in METRICS_TO_COMPARE:
        c_val = challenger_metrics.get(metric)
        champ_val = champion_metrics.get(metric)

        if c_val is None or champ_val is None:
            # missing metric in one of the runs: skip
            continue

        total_considered += 1
        if c_val > champ_val:
            challenger_wins += 1

    if total_considered == 0:
        # no comparable metrics; do not promote
        return False

    return challenger_wins > (total_considered / 2)


# Champion Selection Logic
def select_champion():
    print("🚀 Starting AWS MLflow Champion Selection")

    # Choose challenger deterministically by highest version number
    challenger = choose_challenger_by_highest_version(client, MODEL_NAME)
    if not challenger:
        print("❌ No challenger model found")
        return

    print(f"ℹ️ Challenger found → version {challenger.version}")

    champion = get_model_version_by_tag(client, MODEL_NAME, "role", "champion")

    # No champion exists
    if not champion:
        print("⚠️ No champion found — promoting challenger directly")

        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

        print(f"✅ Challenger v{challenger.version} promoted to Champion")
        return

    print(f"ℹ️ Champion found → version {champion.version}")

    challenger_metrics = get_model_version_metrics(client, MODEL_NAME, challenger.version)
    champion_metrics = get_model_version_metrics(client, MODEL_NAME, champion.version)

    print("\n📊 Metrics Comparison")
    print(f"{'Metric':<25}{'Challenger':<15}{'Champion':<15}")
    print("-" * 55)

    for metric in METRICS_TO_COMPARE:
        print(
            f"{metric:<25}"
            f"{str(challenger_metrics.get(metric, 'N/A')):<15}"
            f"{str(champion_metrics.get(metric, 'N/A')):<15}"
        )

    if better_than(challenger_metrics, champion_metrics):
        print("\n🚀 Challenger outperforms Champion → Promoting")

        # Archive old champion
        client.set_model_version_tag(MODEL_NAME, champion.version, "role", "archived")
        client.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")

        # Promote challenger
        client.set_model_version_tag(MODEL_NAME, challenger.version, "role", "champion")
        client.set_model_version_tag(MODEL_NAME, challenger.version, "status", "production")

        print(f"✅ Challenger v{challenger.version} is now Champion")

    else:
        print("\n⚠️ Challenger did NOT outperform Champion — no change")


if __name__ == "__main__":
    select_champion()
