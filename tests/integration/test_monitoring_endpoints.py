"""Integration tests for the monitoring surface of the API.

They run with no DATABASE_URL configured, so the monitoring store is not ready
and the write endpoints must degrade gracefully (503) while prediction and
metrics keep working.
"""

from uuid import UUID

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


def test_health_exposes_monitoring_flags():
    body = client.get("/health").json()
    assert body["challenger_ready"] is False  # no challenger configured
    assert body["database_ready"] is False  # no DATABASE_URL in tests


def test_predict_returns_prediction_id(monkeypatch):
    monkeypatch.setattr(model_service, "model", FakeModel())
    monkeypatch.setattr(
        model_service, "metadata", {"model_version": "test-1", "model_name": "fake"}
    )
    body = client.post("/predict", json=VALID_REQUEST).json()
    # prediction_id must be a valid UUID even when no database is configured.
    UUID(body["prediction_id"])
    assert body["predicted_weight_grams"] == 3200.0


def test_feedback_returns_503_without_database():
    payload = {
        "prediction_id": "00000000-0000-0000-0000-000000000000",
        "actual_weight_g": 3300,
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 503


def test_support_returns_503_without_database():
    payload = {"category": "bug", "message": "The form did not submit."}
    response = client.post("/support", json=payload)
    assert response.status_code == 503


def test_metrics_endpoint_is_prometheus_text():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "stp_http_requests_total" in response.text
    assert "stp_predictions_total" in response.text
