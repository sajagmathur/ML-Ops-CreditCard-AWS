"""
SageMaker Processing Job Script
Batch Model Monitoring using Evidently

Inputs:
- Reference data (CSV)
- Current data (CSV)
- Champion model (PKL)

Outputs:
- Evidently HTML report
- Metrics JSON
- Retraining decision CSV
"""

import os
import json
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import Dataset, DataDefinition
from evidently import BinaryClassification


# -----------------------------
# SageMaker Processing paths
# -----------------------------
REFERENCE_DIR = "/opt/ml/processing/input/reference"
CURRENT_DIR = "/opt/ml/processing/input/current"
MODEL_DIR = "/opt/ml/processing/input/model"

MONITORING_OUT = "/opt/ml/processing/output/monitoring"
RETRAIN_OUT = "/opt/ml/processing/output/retraining_decision"
METRICS_OUT = "/opt/ml/processing/output/metrics"


# -----------------------------
# Utilities
# -----------------------------
def load_csv_from_dir(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return pd.read_csv(os.path.join(directory, files[0]))


def load_model():
    model_path = os.path.join(MODEL_DIR, "champion_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("champion_model.pkl not found")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred)
    }


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(MONITORING_OUT, exist_ok=True)
    os.makedirs(RETRAIN_OUT, exist_ok=True)
    os.makedirs(METRICS_OUT, exist_ok=True)

    print("📥 Loading data")
    ref = load_csv_from_dir(REFERENCE_DIR)
    cur = load_csv_from_dir(CURRENT_DIR)

    print("📦 Loading champion model")
    model = load_model()

    target = "Class"

    exclude_cols = {
        "ID",
        target,
        "PREDICTION",
        "PREDICTION_PROB"
    }

    feature_cols = [c for c in ref.columns if c not in exclude_cols]

    ref[feature_cols] = ref[feature_cols].apply(pd.to_numeric, errors="coerce")
    cur[feature_cols] = cur[feature_cols].apply(pd.to_numeric, errors="coerce")

    print("🔮 Generating predictions")
    ref["prediction"] = model.predict(ref[feature_cols])
    cur["prediction"] = model.predict(cur[feature_cols])

    # -----------------------------
    # Evidently
    # -----------------------------
    dd = DataDefinition(
        classification=[
            BinaryClassification(
                target=target,
                prediction_labels="prediction"
            )
        ],
        categorical_columns=[target, "prediction"]
    )

    ds_ref = Dataset.from_pandas(ref, data_definition=dd)
    ds_cur = Dataset.from_pandas(cur, data_definition=dd)

    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset()
        ]
    )

    print("📊 Running Evidently report")
    result = report.run(
        reference_data=ds_ref,
        current_data=ds_cur
    )

    report_path = os.path.join(MONITORING_OUT, "evidently_report.html")
    result.save_html(report_path)

    # -----------------------------
    # Metrics
    # -----------------------------
    ref_metrics = calc_metrics(ref[target], ref["prediction"])
    cur_metrics = calc_metrics(cur[target], cur["prediction"])

    with open(os.path.join(METRICS_OUT, "reference_metrics.json"), "w") as f:
        json.dump(ref_metrics, f, indent=2)

    with open(os.path.join(METRICS_OUT, "current_metrics.json"), "w") as f:
        json.dump(cur_metrics, f, indent=2)

    # -----------------------------
    # Retraining decision
    # -----------------------------
    degraded = []
    threshold = 0.10

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        if ref_metrics[metric] - cur_metrics[metric] > threshold:
            degraded.append(metric)

    decision = "YES" if degraded else "NO"
    rationale = (
        f"10% degradation detected in: {', '.join(degraded)}"
        if degraded else
        "All metrics within acceptable threshold"
    )

    retrain_df = pd.DataFrame({
        "retraining_required": [decision],
        "rationale": [rationale]
    })

    retrain_df.to_csv(
        os.path.join(RETRAIN_OUT, "retrain_decision.csv"),
        index=False
    )

    print("✅ Monitoring completed successfully")


if __name__ == "__main__":
    main()
