-- Monitoring schema for the STP Natality API.
--
-- Run once in the Supabase SQL editor (or any Postgres the API can reach via
-- DATABASE_URL) before starting the API with monitoring enabled. The API
-- writes to these tables; the Grafana dashboard reads from them. See
-- monitoring/README.md.

CREATE SCHEMA IF NOT EXISTS monitoring;

-- One row per served prediction. The champion is the live model; the
-- challenger columns are populated only when a shadow model is configured.
CREATE TABLE IF NOT EXISTS monitoring.predictions (
    id                      uuid PRIMARY KEY,
    created_at              timestamptz NOT NULL DEFAULT now(),
    app_version             text        NOT NULL,
    champion_name           text        NOT NULL,
    champion_version        text        NOT NULL,
    champion_prediction_g   double precision NOT NULL,
    challenger_name         text,
    challenger_version      text,
    challenger_prediction_g double precision,
    risk_category           text        NOT NULL,
    inputs                  jsonb        NOT NULL,
    latency_ms              double precision NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_created_at_idx
    ON monitoring.predictions (created_at);

-- Actual birth weight reported back for a prediction. One row per prediction
-- (the API upserts on prediction_id).
CREATE TABLE IF NOT EXISTS monitoring.feedback (
    prediction_id   uuid PRIMARY KEY REFERENCES monitoring.predictions (id),
    actual_weight_g double precision NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Free-text support/contact requests submitted from the UI.
CREATE TABLE IF NOT EXISTS monitoring.support_requests (
    id         uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    category   text NOT NULL,
    message    text NOT NULL
);

CREATE INDEX IF NOT EXISTS support_requests_created_at_idx
    ON monitoring.support_requests (created_at);

-- One row per scheduled Evidently drift run (reference vs recent production
-- inputs). Written by scripts/run_drift_monitoring.py; read by the Grafana
-- "Evidently data drift" panels. Fields are nullable so an insufficient_data
-- run still records that it happened.
CREATE TABLE IF NOT EXISTS monitoring.drift (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       timestamptz NOT NULL DEFAULT now(),
    generated_at     timestamptz NOT NULL,
    status           text        NOT NULL DEFAULT 'ok',
    reference_rows   integer,
    current_rows     integer,
    dataset_drift    boolean,
    drifted_features integer,
    total_features   integer,
    drift_share      double precision,
    features         jsonb
);

CREATE INDEX IF NOT EXISTS drift_generated_at_idx
    ON monitoring.drift (generated_at);
