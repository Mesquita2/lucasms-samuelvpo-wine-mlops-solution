"""Avalia um run no conjunto de teste e salva checkpoint local."""
import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "challenger.pkl")


def evaluate(run_id: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    *_, (X_test, y_test) = load_split()
    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CHECKPOINT_PATH)

    result = {"run_id": run_id, **metrics, "checkpoint": CHECKPOINT_PATH}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate run on test set")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    evaluate(args.run_id)


if __name__ == "__main__":
    main()
