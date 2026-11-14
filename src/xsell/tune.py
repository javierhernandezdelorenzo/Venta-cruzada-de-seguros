"""Búsqueda aleatoria de hiperparámetros, trazada en MLflow.

Uso:
    python -m xsell.tune --config configs/train.yaml --trials 12

Solo usa validación cruzada sobre entrenamiento: el test no interviene en la
elección de hiperparámetros. No modifica la configuración: los mejores valores
se copian a mano a configs/train.yaml, de modo que el cambio queda en git.
"""

import argparse

import mlflow
from sklearn.model_selection import ParameterSampler

from xsell.config import SEED, load_config
from xsell.data import load_data, split_data
from xsell.evaluate import cross_validation_metrics
from xsell.features import EXCLUDED, MODEL_FEATURES
from xsell.models import build_xgboost

SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


def tune(config_path: str, n_trials: int) -> dict:
    """Prueba `n_trials` combinaciones y devuelve la de mayor PR-AUC en CV."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, _, y_train, _ = split_data(load_data(cfg["data_path"]))
    X_train = X_train[MODEL_FEATURES]

    best_params, best_score = None, -1.0
    with mlflow.start_run(run_name="tuning"):
        # La semilla fija hace que se prueben siempre las mismas combinaciones.
        for i, params in enumerate(
            ParameterSampler(SEARCH_SPACE, n_iter=n_trials, random_state=SEED)
        ):
            with mlflow.start_run(run_name=f"trial-{i:02d}", nested=True):
                metrics = cross_validation_metrics(
                    build_xgboost(drop=EXCLUDED, **params), X_train, y_train
                )
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
            print(f"trial {i:02d}  PR-AUC CV {metrics['cv_pr_auc_mean']:.4f}  {params}")
            if metrics["cv_pr_auc_mean"] > best_score:
                best_params, best_score = params, metrics["cv_pr_auc_mean"]

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_pr_auc_mean", best_score)

    print(f"\nMejor combinación (PR-AUC CV {best_score:.4f}): {best_params}")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    tune(args.config, args.trials)
