"""
SageMaker Processing Job Script
Batch Model Monitoring (NO Evidently)

Outputs:
- Drift metrics (JSON + CSV)
- Performance metrics (JSON + CSV)
- Retraining decision (CSV)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# -----------------------------
# SageMaker paths
# -----------------------------
REFERENCE_DIR = "/opt/ml/processing/input/reference"
CURRENT_DIR = "/opt/ml/processing/input/current"
MODEL_DIR = "/opt/ml/processing/input/model"

DRIFT_OUT = "/opt/ml/processing/output/drift"
METRICS_OUT = "/opt/ml/processing/output/metrics"
RETRAIN_OUT = "/opt/ml/processing/output/retraining_decision"

# -----------------------------
# Utilities
# -----------------------------
def load_csv_from_dir(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return pd.read_csv(os.path.join(directory, files[0]))


def load_model():
    path = os.path.join(MODEL_DIR, "champion_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred))
    }


def population_stability_index(ref, cur, bins=10):
    ref = ref.dropna()
    cur = cur.dropna()

    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    psi = np.sum(
        (ref_pct - cur_pct) * np.log((ref_pct + 1e-6) / (cur_pct + 1e-6))
    )
    return float(psi)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(DRIFT_OUT, exist_ok=True)
    os.makedirs(METRICS_OUT, exist_ok=True)
    os.makedirs(RETRAIN_OUT, exist_ok=True)

    print("📥 Loading data")
    ref = load_csv_from_dir(REFERENCE_DIR)
    cur = load_csv_from_dir(CURRENT_DIR)

    print("📦 Loading champion model")
    model = load_model()

    target = "Class"
    exclude = {"ID", target}

    feature_cols = [c for c in ref.columns if c not in exclude]

    ref[feature_cols] = ref[feature_cols].apply(pd.to_numeric, errors="coerce")
    cur[feature_cols] = cur[feature_cols].apply(pd.to_numeric, errors="coerce")

    print("🔮 Generating predictions")
    ref["prediction"] = model.predict(ref[feature_cols])
    cur["prediction"] = model.predict(cur[feature_cols])

    # -----------------------------
    # Performance metrics
    # -----------------------------
    ref_metrics = calc_metrics(ref[target], ref["prediction"])
    cur_metrics = calc_metrics(cur[target], cur["prediction"])

    # JSON outputs
    with open(os.path.join(METRICS_OUT, "reference_metrics.json"), "w") as f:
        json.dump(ref_metrics, f, indent=2)
    with open(os.path.join(METRICS_OUT, "current_metrics.json"), "w") as f:
        json.dump(cur_metrics, f, indent=2)

    # CSV (dashboard-friendly)
    perf_df = pd.DataFrame([
        {"dataset": "reference", **ref_metrics},
        {"dataset": "current", **cur_metrics}
    ])
    perf_df.to_csv(
        os.path.join(METRICS_OUT, "performance_metrics.csv"),
        index=False
    )

    # -----------------------------
    # Drift metrics
    # -----------------------------
    drift_rows = []
    drift_json = {}

    for col in feature_cols:
        ks_stat, ks_p = ks_2samp(ref[col].dropna(), cur[col].dropna())
        psi = population_stability_index(ref[col], cur[col])

        drift_detected = bool((ks_p < 0.05) or (psi > 0.25))

        drift_json[col] = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "psi": psi,
            "drift_detected": drift_detected
        }

        drift_rows.append({
            "feature": col,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "psi": psi,
            "drift_detected": drift_detected
        })

    # JSON output
    with open(os.path.join(DRIFT_OUT, "drift_metrics.json"), "w") as f:
        json.dump(drift_json, f, indent=2)

    # CSV output
    pd.DataFrame(drift_rows).to_csv(
        os.path.join(DRIFT_OUT, "drift_metrics.csv"),
        index=False
    )

    # -----------------------------
    # Retraining decision
    # -----------------------------
    degraded = []
    threshold = 0.10

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        if ref_metrics[metric] - cur_metrics[metric] > threshold:
            degraded.append(metric)

    drifted_features = [
        row["feature"] for row in drift_rows if row["drift_detected"]
    ]

    retrain = bool(degraded or drifted_features)

    retrain_df = pd.DataFrame({
        "retraining_required": ["YES" if retrain else "NO"],
        "performance_degradation": [", ".join(degraded)],
        "drifted_features": [", ".join(drifted_features)]
    })

    retrain_df.to_csv(
        os.path.join(RETRAIN_OUT, "retrain_decision.csv"),
        index=False
    )

    print("✅ Monitoring completed successfully")


if __name__ == "__main__":
    main()
