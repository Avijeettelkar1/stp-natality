"""Integration tests for the prediction HTTP contract and the web UI."""

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
    assert response.json()["version"]


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
    assert body["prediction_id"]
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


def test_index_serves_product_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="predict-form"' in response.text
    assert 'href="/monitoring"' in response.text
    assert "/feedback" in response.text


def test_model_info_exposes_recorded_metrics(monkeypatch):
    monkeypatch.setattr(
        model_service,
        "metadata",
        {
            "model_name": "lightgbm",
            "model_version": "v1",
            "test_mae": 362.8,
            "test_r2": 0.389,
        },
    )
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "lightgbm"
    assert body["test_mae"] == 362.8
    assert body["test_r2"] == 0.389


def test_monitoring_redirects_to_configured_dashboard(monkeypatch):
    monkeypatch.setenv("GRAFANA_DASHBOARD_URL", "https://example.grafana.net/d/test")
    response = client.get("/monitoring", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://example.grafana.net/d/test"
    assert client.get("/health").json()["monitoring_dashboard"] is True


def test_monitoring_without_configuration_returns_503(monkeypatch):
    monkeypatch.delenv("GRAFANA_DASHBOARD_URL", raising=False)
    response = client.get("/monitoring", follow_redirects=False)
    assert response.status_code == 503
    assert client.get("/health").json()["monitoring_dashboard"] is False
