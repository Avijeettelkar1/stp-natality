"""API tests for shadow inference, monitoring, feedback, support and metrics."""

from uuid import UUID, uuid4

import pytest
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
    def __init__(self, value):
        self.value = value

    def predict(self, _features):
        return [self.value]


class BrokenModel:
    def predict(self, _features):
        raise ValueError("challenger exploded")


class FakeStore:
    ready = True

    def __init__(self):
        self.predictions = []
        self.feedback = {}
        self.support = []

    def log_prediction(self, **record):
        self.predictions.append(record)

    def save_feedback(self, prediction_id, actual_weight_g):
        known = {record["prediction_id"] for record in self.predictions}
        if prediction_id not in known:
            raise PredictionNotFound(str(prediction_id))
        self.feedback[prediction_id] = actual_weight_g

    def save_support_request(self, category, message):
        self.support.append((category, message))
        return uuid4()


@pytest.fixture
def champion(monkeypatch):
    monkeypatch.setattr(main.model_service, "model", FakeModel(3400.0))
    monkeypatch.setattr(
        main.model_service,
        "metadata",
        {"model_name": "lightgbm", "model_version": "champ-1"},
    )


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(main, "monitoring_store", fake)
    return fake


def test_prediction_is_logged_with_shadow_challenger(champion, store, monkeypatch):
    monkeypatch.setattr(main.challenger_service, "model", FakeModel(3350.0))
    monkeypatch.setattr(
        main.challenger_service,
        "metadata",
        {"model_name": "xgboost", "model_version": "chall-1"},
    )

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_weight_grams"] == 3400.0  # user sees the champion only
    record = store.predictions[0]
    assert record["prediction_id"] == UUID(body["prediction_id"])
    assert record["champion_prediction_g"] == 3400.0
    assert record["challenger_name"] == "xgboost"
    assert record["challenger_prediction_g"] == 3350.0
    assert record["inputs"]["gestation_weeks"] == 39


def test_failing_challenger_does_not_affect_the_user(champion, store, monkeypatch):
    monkeypatch.setattr(main.challenger_service, "model", BrokenModel())

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["predicted_weight_grams"] == 3400.0
    assert store.predictions[0]["challenger_prediction_g"] is None


def test_failing_database_does_not_affect_the_user(champion, monkeypatch):
    class BrokenStore(FakeStore):
        def log_prediction(self, **record):
            raise ConnectionError("database down")

    monkeypatch.setattr(main, "monitoring_store", BrokenStore())

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200


def test_feedback_is_saved_for_a_known_prediction(champion, store):
    prediction_id = client.post("/predict", json=VALID_REQUEST).json()["prediction_id"]

    response = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "actual_weight_g": 3350},
    )

    assert response.status_code == 200
    assert store.feedback[UUID(prediction_id)] == 3350


def test_feedback_for_unknown_prediction_returns_404(store):
    response = client.post(
        "/feedback",
        json={"prediction_id": str(uuid4()), "actual_weight_g": 3350},
    )

    assert response.status_code == 404


def test_feedback_rejects_impossible_weight(store):
    response = client.post(
        "/feedback",
        json={"prediction_id": str(uuid4()), "actual_weight_g": 90000},
    )

    assert response.status_code == 422


def test_feedback_without_database_returns_503(monkeypatch):
    class NoStore(FakeStore):
        ready = False

    monkeypatch.setattr(main, "monitoring_store", NoStore())

    response = client.post(
        "/feedback",
        json={"prediction_id": str(uuid4()), "actual_weight_g": 3350},
    )

    assert response.status_code == 503


def test_support_request_is_saved(store):
    response = client.post(
        "/support",
        json={"category": "wrong_estimate", "message": "Estimate looked too low."},
    )

    assert response.status_code == 200
    assert store.support == [("wrong_estimate", "Estimate looked too low.")]


def test_metrics_endpoint_exposes_prometheus_metrics(champion, store):
    client.post("/predict", json=VALID_REQUEST)
    client.get("/this-route-does-not-exist")

    body = client.get("/metrics").text

    assert "stp_predictions_total" in body
    assert 'path="/predict"' in body
    assert 'path="unmatched"' in body
    assert "this-route-does-not-exist" not in body


def test_health_reports_challenger_and_database(store):
    body = client.get("/health").json()

    assert set(body) >= {"model_ready", "challenger_ready", "database_ready", "version"}
    assert body["database_ready"] is True
