import pytest
from fastapi.testclient import TestClient

from xsell.api import LoadedModel, app, get_model
from xsell.data import TARGET
from xsell.features import EXCLUDED, MODEL_FEATURES
from xsell.models import build_xgboost

CUSTOMER = {
    "Age": 44,
    "Driving_License": 1,
    "Region_Code": 28,
    "Previously_Insured": 0,
    "Vehicle_Age": "> 2 Years",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 40454.0,
    "Policy_Sales_Channel": 26,
}


@pytest.fixture
def client(raw_df):
    """Cliente de la API con un modelo pequeño inyectado (sin MLflow)."""
    pipeline = build_xgboost(drop=EXCLUDED, n_estimators=20).fit(
        raw_df[MODEL_FEATURES], raw_df[TARGET]
    )
    app.dependency_overrides[get_model] = lambda: LoadedModel(pipeline=pipeline, version="test")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_predict_returns_one_probability_per_customer(client):
    response = client.post("/predict", json={"customers": [CUSTOMER, CUSTOMER]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test"
    assert len(body["probabilities"]) == 2
    assert all(0 <= p <= 1 for p in body["probabilities"])


def test_predict_accepts_unseen_region_and_channel(client):
    customer = {**CUSTOMER, "Region_Code": 999, "Policy_Sales_Channel": 999}
    assert client.post("/predict", json={"customers": [customer]}).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [("Age", 10), ("Vehicle_Age", "3 Years"), ("Vehicle_Damage", "Maybe"), ("Annual_Premium", -5)],
)
def test_predict_rejects_invalid_values(client, field, value):
    response = client.post("/predict", json={"customers": [{**CUSTOMER, field: value}]})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    customer = {k: v for k, v in CUSTOMER.items() if k != "Age"}
    assert client.post("/predict", json={"customers": [customer]}).status_code == 422


def test_predict_rejects_empty_list(client):
    assert client.post("/predict", json={"customers": []}).status_code == 422


def test_returns_503_when_no_model_is_loaded():
    # Sin lifespan y sin modelo inyectado: la API debe avisar, no romperse.
    response = TestClient(app).post("/predict", json={"customers": [CUSTOMER]})
    assert response.status_code == 503
