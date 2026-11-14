"""Test de integración: entrenar -> promocionar -> servir, de principio a fin.

Usa datos sintéticos y un MLflow temporal, así que no toca el registro real.
"""

import yaml
from fastapi.testclient import TestClient

from tests.test_api import CUSTOMER
from xsell import api
from xsell.promote import promote
from xsell.train import train


def test_train_promote_and_serve(tmp_path, csv_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow escribe sus artefactos en el directorio actual
    config = {
        "data_path": str(csv_path),
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "experiment_name": "test",
            "registered_model_name": "xsell-test",
        },
        "model": {"n_estimators": 20, "max_depth": 3},
        "promotion": {"min_test_pr_auc": 0.0},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # 1) Entrenar registra la versión 1; 2) al no haber champion, se promociona.
    version = train(str(config_path))
    assert str(version) == "1"
    assert promote(str(config_path)) is True

    # 3) La API arranca, carga el champion y predice.
    monkeypatch.setattr(api, "CONFIG_PATH", str(config_path))
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_version": "1"}
        response = client.post("/predict", json={"customers": [CUSTOMER]})
        assert response.status_code == 200
        assert 0 <= response.json()["probabilities"][0] <= 1
