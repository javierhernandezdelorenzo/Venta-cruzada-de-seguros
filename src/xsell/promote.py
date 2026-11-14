"""Promoción de un modelo candidato a producción.

Uso:
    python -m xsell.promote --config configs/train.yaml

Compara la versión "challenger" (la última entrenada) con la "champion" (la
que está sirviendo) y solo mueve el alias "champion" si el candidato pasa
las dos validaciones. El endpoint carga siempre la versión "champion".
"""

import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from xsell.config import ALIAS_CHALLENGER, ALIAS_CHAMPION, load_config

METRIC = "test_pr_auc"


def promote(config_path: str) -> bool:
    """Devuelve True si el candidato se ha promocionado a champion."""
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    model_name = cfg["mlflow"]["registered_model_name"]
    minimum = cfg["promotion"]["min_test_pr_auc"]
    client = MlflowClient()

    challenger = client.get_model_version_by_alias(model_name, ALIAS_CHALLENGER)
    challenger_metric = client.get_run(challenger.run_id).data.metrics[METRIC]

    # Validación 1: mínimo absoluto de calidad.
    if challenger_metric < minimum:
        print(
            f"RECHAZADO v{challenger.version}: {METRIC}={challenger_metric:.4f} < mínimo {minimum}"
        )
        return False

    # Validación 2: no empeorar al modelo que está en producción (si lo hay).
    try:
        champion = client.get_model_version_by_alias(model_name, ALIAS_CHAMPION)
    except MlflowException:
        champion = None

    if champion is not None:
        if champion.version == challenger.version:
            print(f"v{challenger.version} ya es champion; nada que hacer")
            return False
        champion_metric = client.get_run(champion.run_id).data.metrics[METRIC]
        if challenger_metric < champion_metric:
            print(
                f"RECHAZADO v{challenger.version}: {METRIC}={challenger_metric:.4f} "
                f"< champion v{champion.version} ({champion_metric:.4f})"
            )
            return False

    client.set_registered_model_alias(model_name, ALIAS_CHAMPION, challenger.version)
    print(f"PROMOCIONADO v{challenger.version} a '{ALIAS_CHAMPION}'")
    print(f"{METRIC}={challenger_metric:.4f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    promote(parser.parse_args().config)
