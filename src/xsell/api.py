"""API de predicción: sirve el modelo "champion" del registro de MLflow.

Uso:
    uvicorn xsell.api:app --port 8000

La documentación interactiva queda en http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from xsell.config import ALIAS_CHAMPION, load_config
from xsell.features import MODEL_FEATURES

# La API lee la misma configuración que el entrenamiento para saber dónde
# está el registro de modelos. Se puede cambiar con una variable de entorno.
CONFIG_PATH = os.getenv("XSELL_CONFIG", "configs/train.yaml")


@dataclass
class LoadedModel:
    pipeline: Pipeline
    version: str


def load_champion(config_path: str) -> LoadedModel | None:
    """Carga la versión "champion" del registro; None si todavía no hay ninguna."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model_name"]
    try:
        version = MlflowClient().get_model_version_by_alias(name, ALIAS_CHAMPION).version
    except MlflowException:
        return None
    pipeline = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    return LoadedModel(pipeline=pipeline, version=str(version))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga una sola vez, al arrancar, no en cada petición.
    app.state.model = load_champion(CONFIG_PATH)
    yield


app = FastAPI(title="Venta cruzada de seguros", version="0.1.0", lifespan=lifespan)


def get_model(request: Request) -> LoadedModel:
    """Dependencia de FastAPI: devuelve el modelo cargado o responde 503."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="No hay ningún modelo 'champion' cargado")
    return model


class Customer(BaseModel):
    """Datos de un cliente. Los nombres coinciden con las columnas del dataset."""

    Age: int = Field(ge=18, le=120)
    Driving_License: Literal[0, 1]
    Region_Code: int = Field(ge=0)
    Previously_Insured: Literal[0, 1]
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(gt=0)
    Policy_Sales_Channel: int = Field(ge=0)


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    model_version: str
    # Probabilidad de interés de cada cliente, en el mismo orden que la petición.
    probabilities: list[float]


@app.get("/health")
def health(model: LoadedModel = Depends(get_model)) -> dict:
    """Comprueba que el servicio está arriba y con modelo cargado."""
    return {"status": "ok", "model_version": model.version}


@app.post("/predict")
def predict(request: PredictRequest, model: LoadedModel = Depends(get_model)) -> PredictResponse:
    """Devuelve la probabilidad de interés en el seguro de vehículo."""
    X = pd.DataFrame([customer.model_dump() for customer in request.customers])[MODEL_FEATURES]
    probabilities = model.pipeline.predict_proba(X)[:, 1]
    return PredictResponse(model_version=model.version, probabilities=probabilities.tolist())
