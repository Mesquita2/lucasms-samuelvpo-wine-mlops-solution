"""Microsserviço FastAPI que serve o modelo @champion."""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "wine-classifier"

FEATURE_ORDER = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


class WineFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float = Field(alias="od280/od315_of_diluted_wines")
    proline: float


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        state["version"] = int(mv.version)
    except Exception as e:
        state["error"] = str(e)
    yield
    state.clear()


app = FastAPI(title="wine-classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail=state.get("error", "model not loaded"))
    return {"status": "ok", "champion_version": state["version"]}


@app.post("/predict")
def predict(payload: WineFeatures):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")

    row = payload.model_dump(by_alias=True)
    df = pd.DataFrame([[row[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)

    model = state["model"]
    probs = model.predict_proba(df)[0]
    cls = int(probs.argmax())
    return {
        "class": cls,
        "proba": float(probs[cls]),
        "all_probs": [float(p) for p in probs],
        "champion_version": state["version"],
    }
