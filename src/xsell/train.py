"""Entrenamiento reproducible con trazabilidad en MLflow.

Uso:
    python -m xsell.train --config configs/train.yaml

Cada ejecución deja en MLflow los parámetros, la huella de los datos, las
métricas de validación cruzada y de test, la tabla de lift y el modelo, y
registra una versión nueva del modelo con el alias "challenger".
"""

import argparse
import hashlib

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from xsell.config import ALIAS_CHALLENGER, CV_FOLDS, SEED, TEST_SIZE, load_config
from xsell.data import load_data, split_data
from xsell.evaluate import capture_at, cross_validation_metrics, lift_table, ranking_metrics
from xsell.features import EXCLUDED, MODEL_FEATURES
from xsell.models import build_xgboost

# MLflow guarda el pipeline con skops (más seguro que pickle), que exige
# declarar los tipos ajenos a scikit-learn que se permiten al cargar.
TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def file_sha256(path: str) -> str:
    """Huella del fichero de datos: permite saber con qué datos exactos se entrenó."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train(config_path: str) -> str:
    """Entrena, evalúa y registra el modelo. Devuelve la versión registrada."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    model_name = cfg["mlflow"]["registered_model_name"]

    df = load_data(cfg["data_path"])
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = X_train[MODEL_FEATURES], X_test[MODEL_FEATURES]

    model = build_xgboost(drop=EXCLUDED, **cfg["model"])

    with mlflow.start_run():
        mlflow.log_params(cfg["model"])
        mlflow.log_params(
            {
                "seed": SEED,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "features": ",".join(MODEL_FEATURES),
                "excluded": ",".join(EXCLUDED),
                "data_sha256": file_sha256(cfg["data_path"]),
                "data_rows": len(df),
            }
        )

        # 1) Validación cruzada sobre entrenamiento: estimación de rendimiento.
        mlflow.log_metrics(cross_validation_metrics(model, X_train, y_train))

        # 2) Ajuste con todo el entrenamiento y evaluación única en test.
        model.fit(X_train, y_train)
        score = model.predict_proba(X_test)[:, 1]

        metrics = {f"test_{name}": value for name, value in ranking_metrics(y_test, score).items()}
        for pct in (0.1, 0.2, 0.3):
            metrics[f"test_capture_at_{int(pct * 100)}"] = capture_at(y_test, score, pct)
        mlflow.log_metrics(metrics)
        mlflow.log_text(lift_table(y_test, score).to_csv(), "lift_table.csv")
        mlflow.log_artifact(config_path)

        # 3) Registro del modelo (pipeline completo: preprocesado + XGBoost).
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, score),
            input_example=X_test.head(5),
            registered_model_name=model_name,
            skops_trusted_types=TRUSTED_TYPES,
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(model_name, ALIAS_CHALLENGER, version)

    print(f"Modelo {model_name} v{version} registrado como '{ALIAS_CHALLENGER}'")
    print(
        f"PR-AUC  CV: {mlflow.get_run(info.run_id).data.metrics['cv_pr_auc_mean']:.4f}"
        f"   test: {metrics['test_pr_auc']:.4f}"
    )
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    train(parser.parse_args().config)
