"""Cria um modelo champion inicial (mais fraco de propósito) para que o pipeline
de promoção tenha um baseline para superar.

Uso:
    uv run python -m scripts.bootstrap_champion
"""
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data import load_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "wine-classifier"
MODEL_NAME = "wine-classifier"


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    (X_train, y_train), (X_val, y_val), _ = load_split()

    # Champion intencionalmente fraco (poucas árvores rasas) para o challenger superar.
    params = {"n_estimators": 10, "max_depth": 2, "random_state": 42}

    with mlflow.start_run(run_name="champion-bootstrap"):
        mlflow.log_params(params)
        model = RandomForestClassifier(**params).fit(X_train, y_train)

        f1 = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        mlflow.log_metric("f1_macro", f1)

        result = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

    client = MlflowClient()
    version = result.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))

    print(f"Champion registrado em {MODEL_NAME} v{version} (f1_val={f1:.4f})")


if __name__ == "__main__":
    main()
