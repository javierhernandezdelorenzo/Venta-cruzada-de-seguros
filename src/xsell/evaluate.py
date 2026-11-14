"""Métricas de evaluación orientadas a ranking.

El modelo se usa para decidir a quién contactar primero, así que lo que
importa es cómo ordena a los clientes, no si acierta con un umbral de 0,5.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from xsell.config import CV_FOLDS, SEED


def ranking_metrics(y_true, score) -> dict[str, float]:
    """PR-AUC (la principal, sensible al desbalanceo) y ROC-AUC."""
    return {
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
    }


def capture_at(y_true, score, pct: float) -> float:
    """Fracción de los interesados que se captura contactando al `pct` con mayor score."""
    y_sorted = np.asarray(y_true)[np.argsort(-np.asarray(score), kind="stable")]
    n_contacted = int(round(len(y_sorted) * pct))
    return y_sorted[:n_contacted].sum() / y_sorted.sum()


def lift_table(y_true, score, n_bins: int = 10) -> pd.DataFrame:
    """Tabla por tramos de score (el tramo 1 son los clientes con mayor score).

    Columnas: clientes, interesados, tasa de interés, lift sobre la tasa
    global, captura acumulada de interesados y score medio. Comparar
    `score_medio` con `tasa_interes` indica si las probabilidades están
    bien calibradas.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    df = df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    # Tramos del mismo tamaño por posición en el ranking (evita problemas con empates).
    df["tramo"] = np.arange(len(df)) * n_bins // len(df) + 1

    table = df.groupby("tramo").agg(
        clientes=("y", "size"),
        interesados=("y", "sum"),
        tasa_interes=("y", "mean"),
        score_medio=("score", "mean"),
    )
    table["lift"] = table["tasa_interes"] / df["y"].mean()
    table["captura_acumulada"] = table["interesados"].cumsum() / df["y"].sum()
    return table[
        ["clientes", "interesados", "tasa_interes", "lift", "captura_acumulada", "score_medio"]
    ]


def cross_validation_metrics(model, X, y) -> dict[str, float]:
    """Media y desviación de PR-AUC y ROC-AUC con validación cruzada estratificada."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    result = cross_validate(model, X, y, cv=cv, scoring=["average_precision", "roc_auc"])
    return {
        "cv_pr_auc_mean": result["test_average_precision"].mean(),
        "cv_pr_auc_std": result["test_average_precision"].std(),
        "cv_roc_auc_mean": result["test_roc_auc"].mean(),
        "cv_roc_auc_std": result["test_roc_auc"].std(),
    }
