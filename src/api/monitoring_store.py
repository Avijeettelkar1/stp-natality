"""PostgreSQL persistence used for production monitoring.

Stores every prediction (champion and shadow challenger), ground-truth
feedback and support requests in the ``monitoring`` schema. The connection
string comes from the ``DATABASE_URL`` environment variable.
"""

import os
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb

CONNECT_TIMEOUT_SECONDS = 5

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS monitoring;

CREATE TABLE IF NOT EXISTS monitoring.predictions (
  id                      uuid PRIMARY KEY,
  created_at              timestamptz NOT NULL DEFAULT now(),
  app_version             text,
  champion_name           text NOT NULL,
  champion_version        text NOT NULL,
  champion_prediction_g   double precision NOT NULL,
  challenger_name         text,
  challenger_version      text,
  challenger_prediction_g double precision,
  risk_category           text NOT NULL,
  inputs                  jsonb NOT NULL,
  latency_ms              double precision
);
CREATE INDEX IF NOT EXISTS predictions_created_at_idx
  ON monitoring.predictions (created_at);

CREATE TABLE IF NOT EXISTS monitoring.feedback (
  prediction_id   uuid PRIMARY KEY REFERENCES monitoring.predictions (id),
  actual_weight_g double precision NOT NULL
    CHECK (actual_weight_g BETWEEN 200 AND 8000),
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS monitoring.support_requests (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  category   text NOT NULL,
  message    text NOT NULL
);
"""


class PredictionNotFound(Exception):
    """Feedback refers to a prediction ID that was never logged."""


class MonitoringStore:
    """Write prediction monitoring data to PostgreSQL."""

    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL")

    @property
    def ready(self) -> bool:
        return bool(self.database_url)

    def _connect(self):
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured.")

        return psycopg.connect(
            self.database_url,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
        )

    def ensure_schema(self) -> None:
        """Create the monitoring schema and tables if they do not exist."""
        with self._connect() as connection:
            connection.execute(SCHEMA_SQL)

    def log_prediction(
        self,
        *,
        prediction_id: UUID,
        inputs: dict,
        champion_name: str,
        champion_version: str,
        champion_prediction_g: float,
        challenger_name: str | None,
        challenger_version: str | None,
        challenger_prediction_g: float | None,
        risk_category: str,
        latency_ms: float,
        app_version: str,
    ) -> None:
        query = """
            INSERT INTO monitoring.predictions (
                id,
                app_version,
                champion_name,
                champion_version,
                champion_prediction_g,
                challenger_name,
                challenger_version,
                challenger_prediction_g,
                risk_category,
                inputs,
                latency_ms
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """

        values = (
            prediction_id,
            app_version,
            champion_name,
            champion_version,
            champion_prediction_g,
            challenger_name,
            challenger_version,
            challenger_prediction_g,
            risk_category,
            Jsonb(inputs),
            latency_ms,
        )

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, values)

    def save_feedback(
        self,
        prediction_id: UUID,
        actual_weight_g: float,
    ) -> None:
        query = """
            INSERT INTO monitoring.feedback (
                prediction_id,
                actual_weight_g
            )
            VALUES (%s, %s)
            ON CONFLICT (prediction_id)
            DO UPDATE SET
                actual_weight_g = EXCLUDED.actual_weight_g,
                created_at = now()
        """

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (
                            prediction_id,
                            actual_weight_g,
                        ),
                    )
        except psycopg.errors.ForeignKeyViolation as exc:
            raise PredictionNotFound(str(prediction_id)) from exc

    def save_support_request(
        self,
        category: str,
        message: str,
    ) -> UUID:
        support_id = uuid4()

        query = """
            INSERT INTO monitoring.support_requests (
                id,
                category,
                message
            )
            VALUES (%s, %s, %s)
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        support_id,
                        category,
                        message,
                    ),
                )

        return support_id
