"""Treina um challenger e registra no MLflow Model Registry."""
import argparse
import datetime as dt
import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "wine-classifier"
REGISTERED_NAME = "wine-classifier"


def train(n_estimators: int, max_depth: int) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    (X_train, y_train), (X_val, y_val), _ = load_split()

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
    }

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=f"challenger-{ts}") as run:
        mlflow.log_params(params)

        model = RandomForestClassifier(**params).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_NAME,
            input_example=X_train.head(1),
        )

        run_id = run.info.run_id
        with open(".last_run_id", "w") as f:
            f.write(run_id)
        print(f"run_id={run_id}")
        print(f"metrics={metrics}")
        return run_id


def main():
    parser = argparse.ArgumentParser(description="Train challenger model")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
