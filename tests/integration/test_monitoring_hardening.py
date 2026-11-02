"""API behaviour for unknown feedback IDs and a failing monitoring database."""

from uuid import uuid4

from fastapi.testclient import TestClient

from src.api import main
from src.api.monitoring_store import PredictionNotFound

client = TestClient(main.app)

VALID_REQUEST = {
    "gestation_weeks": 39,
    "mother_age": 29,
    "plurality": 1,
    "year_fix": 2026,
    "month": 10,
    "is_male": True,
}


class FakeModel:
    def predict(self, _features):
        return [3400.0]


class UnknownIdStore:
    ready = True

    def save_feedback(self, prediction_id, actual_weight_g):
        raise PredictionNotFound(str(prediction_id))


class BrokenStore:
    ready = True

    def log_prediction(self, **record):
        raise ConnectionError("database down")


def test_feedback_for_unknown_prediction_returns_404(monkeypatch):
    monkeypatch.setattr(main, "monitoring_store", UnknownIdStore())

    response = client.post(
        "/feedback",
        json={"prediction_id": str(uuid4()), "actual_weight_g": 3350},
    )

    assert response.status_code == 404


def test_prediction_succeeds_when_logging_fails(monkeypatch):
    monkeypatch.setattr(main.model_service, "model", FakeModel())
    monkeypatch.setattr(
        main.model_service,
        "metadata",
        {"model_name": "lightgbm", "model_version": "v1"},
    )
    monkeypatch.setattr(main, "monitoring_store", BrokenStore())

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["predicted_weight_grams"] == 3400.0


def test_metrics_open_without_token(monkeypatch):
    monkeypatch.delenv("METRICS_TOKEN", raising=False)

    assert client.get("/metrics").status_code == 200


def test_metrics_require_token_when_configured(monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "s3cret")

    assert client.get("/metrics").status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.get("/metrics", headers=wrong).status_code == 401
    right = {"Authorization": "Bearer s3cret"}
    response = client.get("/metrics", headers=right)
    assert response.status_code == 200
    assert "stp_" in response.text


def test_model_info_includes_recorded_metrics(monkeypatch):
    monkeypatch.setattr(main.model_service, "model", FakeModel())
    monkeypatch.setattr(
        main.model_service,
        "metadata",
        {
            "model_name": "lightgbm",
            "model_version": "v1",
            "training_rows": 299431,
            "test_mae": 362.8,
            "test_r2": 0.389,
        },
    )

    body = client.get("/model/info").json()

    assert body["training_rows"] == 299431
    assert body["test_mae"] == 362.8
    assert body["test_r2"] == 0.389
    assert body["mlflow_run_id"] is None  # not recorded, so not invented
