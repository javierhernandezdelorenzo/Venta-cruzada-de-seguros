"""Definición de los modelos candidatos como pipelines de scikit-learn.

Cada pipeline incluye su preprocesado, de modo que este se ajusta solo con
los datos de entrenamiento de cada partición (sin fuga de información) y el
objeto que se guarda y se sirve es uno solo.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from xsell.config import SEED
from xsell.features import build_preprocessor

# Hiperparámetros de partida de XGBoost: árboles poco profundos y aprendizaje
# lento para un dataset tabular de pocas variables. Se afinan más adelante.
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def build_logistic(drop: tuple[str, ...] = ()) -> Pipeline:
    """Regresión logística: referencia simple e interpretable."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor("linear", drop=drop)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_xgboost(drop: tuple[str, ...] = (), **params) -> Pipeline:
    """Gradient boosting: candidato principal.

    No se usa `scale_pos_weight`: reequilibrar las clases no mejora el orden
    de los clientes y sí distorsiona las probabilidades, que queremos poder
    leer como "probabilidad de interés".
    """
    model = XGBClassifier(**{**XGB_PARAMS, **params}, random_state=SEED, n_jobs=-1)
    return Pipeline(
        [
            ("preprocess", build_preprocessor("tree", drop=drop)),
            ("model", model),
        ]
    )
