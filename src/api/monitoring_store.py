"""PostgreSQL persistence used for production monitoring."""

import os
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb


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
            connect_timeout=10,
        )

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

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        prediction_id,
                        actual_weight_g,
                    ),
                )

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
