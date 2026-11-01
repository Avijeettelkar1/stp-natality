"""MonitoringStore against a real PostgreSQL database.

Skipped unless TEST_DATABASE_URL points to a disposable Postgres database
(CI starts one as a service container). The tables are created from
monitoring/schema.sql, so these tests also check that file.
"""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from src.api.monitoring_store import MonitoringStore, PredictionNotFound

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
SCHEMA_SQL = Path(__file__).resolve().parents[2] / "monitoring" / "schema.sql"

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set"
)


@pytest.fixture
def store(monkeypatch):
    with psycopg.connect(TEST_DATABASE_URL) as connection:
        connection.execute(SCHEMA_SQL.read_text())
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    return MonitoringStore()


def _log(store, prediction_id):
    store.log_prediction(
        prediction_id=prediction_id,
        inputs={"gestation_weeks": 39, "mother_age": 29},
        champion_name="lightgbm",
        champion_version="v1",
        champion_prediction_g=3400.0,
        challenger_name="xgboost",
        challenger_version="v2",
        challenger_prediction_g=3350.0,
        risk_category="normal",
        latency_ms=12.5,
        app_version="test",
    )


def test_prediction_and_feedback_round_trip(store):
    prediction_id = uuid4()
    _log(store, prediction_id)
    store.save_feedback(prediction_id, 3320.0)
    store.save_feedback(prediction_id, 3330.0)  # feedback can be corrected

    with psycopg.connect(TEST_DATABASE_URL) as connection:
        row = connection.execute(
            """
            SELECT p.champion_prediction_g, p.challenger_prediction_g,
                   p.inputs ->> 'gestation_weeks', f.actual_weight_g
            FROM monitoring.predictions p
            JOIN monitoring.feedback f ON f.prediction_id = p.id
            WHERE p.id = %s
            """,
            (prediction_id,),
        ).fetchone()

    assert row == (3400.0, 3350.0, "39", 3330.0)


def test_feedback_for_unknown_prediction_raises(store):
    with pytest.raises(PredictionNotFound):
        store.save_feedback(uuid4(), 3300.0)


def test_support_request_is_stored(store):
    support_id = store.save_support_request("question", "How accurate is it?")

    with psycopg.connect(TEST_DATABASE_URL) as connection:
        row = connection.execute(
            "SELECT category, message FROM monitoring.support_requests WHERE id = %s",
            (support_id,),
        ).fetchone()

    assert row == ("question", "How accurate is it?")
