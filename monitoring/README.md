# Monitoring

The monitoring component combines three sources in one Grafana Cloud dashboard
(`dashboards/stp-natality-monitoring.json`):

| Source | What it shows | How it gets there |
|---|---|---|
| Supabase Postgres, schema `monitoring` | Predictions, champion vs. shadow challenger, input drift, live MAE once actual weights arrive, support requests | The API writes every prediction and `/feedback` call |
| Prometheus metrics from `/metrics` | Request rate, 5xx errors, latency, predictions per minute, shadow difference, log write errors | Grafana Cloud scrapes the public `/metrics` endpoint |
| Synthetic Monitoring | Availability of `/health` from outside | Grafana Cloud HTTP check |

Alert rules are listed in [`alerts/alert-rules.md`](alerts/alert-rules.md).

## What the API produces

The API is instrumented to feed both data sources above:

- **Prometheus** — `GET /metrics` exposes `stp_http_requests_total`,
  `stp_http_request_duration_seconds`, `stp_predictions_total`,
  `stp_shadow_prediction_difference_grams` and `stp_monitoring_write_errors_total`.
  Always on; no configuration needed.
- **Postgres** — when `DATABASE_URL` is set, every `/predict` writes a row to
  `monitoring.predictions`, `/feedback` upserts the actual weight into
  `monitoring.feedback`, and `/support` writes to `monitoring.support_requests`.
  With no `DATABASE_URL` the API still serves predictions; the write endpoints
  return `503` and `/health` reports `database_ready: false`.
- **Shadow model** — the serving image ships an XGBoost challenger at
  `models/challenger/` (built by `scripts/make_challenger.py`, tracked in DVC),
  and the Dockerfile sets `CHALLENGER_MODEL_PATH` so it runs by default. On each
  prediction the challenger also predicts; its value is logged and the
  champion↔challenger difference recorded (`stp_shadow_prediction_difference_grams`,
  dashboard panels "Champion vs challenger" / "Shadow difference"), but it never
  affects the response. `/health` reports `challenger_ready`. Unset
  `CHALLENGER_MODEL_PATH` to disable it.

## Setup

### 1. Create the schema, tables, and read-only user

Run [`schema.sql`](schema.sql) once in the Supabase SQL editor (creates the
`monitoring` schema and the three tables the dashboard reads), then create the
Grafana read-only role (choose your own password):

```sql
-- after running schema.sql
CREATE ROLE grafana_reader WITH LOGIN PASSWORD 'choose-a-strong-password';
GRANT USAGE ON SCHEMA monitoring TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT SELECT ON TABLES TO grafana_reader;
```

Grafana only reads; the API writes with its own credentials. Point the API at
the database by setting `DATABASE_URL` (Supabase → Connect → the pooled
connection string) in its environment before starting it.

### 2. PostgreSQL data source

Grafana Cloud → **Connections → Add new connection → PostgreSQL**:

- Host URL: Supabase **Session pooler** host and port (Supabase → Connect)
- Database: `postgres`
- User: `grafana_reader.<project-ref>` (pooler usernames include the project ref)
- Password: the `grafana_reader` password
- TLS/SSL mode: `require`

### 3. Prometheus metrics

Grafana Cloud → **Connections → Add new connection → Metrics Endpoint**:

- Scrape URL: `https://stp-natality.onrender.com/metrics`
- Interval: 60 seconds, no authentication

The metrics land in the stack's hosted Prometheus data source
(`grafanacloud-<stack>-prom`). Scraping every minute also keeps the free Render
instance awake.

### 4. Uptime check

Grafana Cloud → **Testing & synthetics → Synthetics → Add new check → HTTP**:
`https://stp-natality.onrender.com/health`, every 5 minutes, two EU probes.

### 5. Import the dashboard

**Dashboards → New → Import** → upload `dashboards/stp-natality-monitoring.json`
→ select the PostgreSQL data source for "Supabase (Postgres)" and the hosted
Prometheus data source for "Grafana Cloud Prometheus" → **Import**.

### 6. Keep it versioned

After changing the dashboard in Grafana, export it (**Share → Export → Save to
file**, with "Export for sharing externally" on), replace the JSON here and
commit.
