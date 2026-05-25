"""Teste A/B automático: champion vs. challenger, com promoção condicional."""
import json
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import f1_score

from src.data import load_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "wine-classifier"
THRESHOLD = float(os.getenv("PROMOTION_THRESHOLD", "0.01"))


def _f1_on_test(model_uri: str, X_test, y_test) -> float:
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)
    return float(f1_score(y_test, y_pred, average="macro", zero_division=0))


def decide_and_promote() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_version = int(champion_mv.version)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    other = [int(v.version) for v in versions if int(v.version) != champion_version]
    if not other:
        raise RuntimeError("Nenhum challenger encontrado — rode src.train antes.")
    challenger_version = max(other)

    *_, (X_test, y_test) = load_split()

    f1_champion = _f1_on_test(f"models:/{MODEL_NAME}/{champion_version}", X_test, y_test)
    f1_challenger = _f1_on_test(f"models:/{MODEL_NAME}/{challenger_version}", X_test, y_test)

    promoted = f1_challenger > f1_champion + THRESHOLD

    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", str(challenger_version))
        client.set_registered_model_alias(MODEL_NAME, "archived", str(champion_version))

    with mlflow.start_run(run_name="promotion-decision"):
        mlflow.log_metrics({
            "f1_champion": f1_champion,
            "f1_challenger": f1_challenger,
            "delta": f1_challenger - f1_champion,
        })
        mlflow.set_tag("promoted", str(promoted).lower())
        mlflow.set_tag("from_version", str(champion_version))
        mlflow.set_tag("to_version", str(challenger_version))
        mlflow.set_tag("threshold", str(THRESHOLD))

    result = {
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "f1_champion": round(f1_champion, 4),
        "f1_challenger": round(f1_challenger, 4),
        "threshold": THRESHOLD,
        "promoted": promoted,
    }
    with open("promote.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    decide_and_promote()
