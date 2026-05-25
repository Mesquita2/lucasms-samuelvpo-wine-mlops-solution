"""Smoke test do endpoint /predict usando o TestClient.
Requer um MLflow server local com modelo @champion registrado.
Pulado se a variável MLFLOW_SKIP=1 estiver definida.
"""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(os.getenv("MLFLOW_SKIP") == "1", reason="MLflow not available")
def test_predict_payload():
    from src.api.main import app

    payload = {
        "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
        "magnesium": 100, "total_phenols": 2.65, "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
        "hue": 1.05, "od280/od315_of_diluted_wines": 3.40, "proline": 1050,
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["class"] in (0, 1, 2)
        assert 0.0 <= body["proba"] <= 1.0
