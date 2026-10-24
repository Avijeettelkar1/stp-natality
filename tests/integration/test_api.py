"""Integration tests for the prediction HTTP contract."""

from fastapi.testclient import TestClient

from src.api.main import app, model_service

client = TestClient(app)

VALID_REQUEST = {
    "gestation_weeks": 39,
    "mother_age": 29,
    "weight_gain_pounds": 27,
    "born_alive_alive": 1,
    "born_alive_dead": 0,
    "born_dead": 0,
    "ever_born": 2,
    "father_age": 31,
    "plurality": 1,
    "year_fix": 2021,
    "month": 6,
    "is_male": True,
    "mother_married": True,
    "lmp_known": True,
}


class FakeModel:
    def predict(self, _features):
        return [3200.0]


def test_health_reports_model_state():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is model_service.ready


def test_predict_returns_valid_contract(monkeypatch):
    monkeypatch.setattr(model_service, "model", FakeModel())
    monkeypatch.setattr(
        model_service,
        "metadata",
        {"model_version": "test-1", "model_name": "fake"},
    )
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_weight_grams"] == 3200.0
    assert body["risk_category"] == "normal"
    assert body["model_version"] == "test-1"


def test_predict_rejects_invalid_gestation():
    request = {**VALID_REQUEST, "gestation_weeks": 50}
    response = client.post("/predict", json=request)
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(model_service, "model", None)
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 503
