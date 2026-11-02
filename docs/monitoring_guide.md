# Utilization, Monitoring, and Maintenance

## Demonstration

1. Show MLflow candidate runs and the selected model.
2. Start the API and inspect `/health`, `/model/info`, and `/docs`.
3. Submit normal, low-weight-risk, and invalid requests.
4. Generate a temporal drift report:

```bash
python scripts/monitor_drift.py
```

The report compares training and recent test distributions with two-sample
Kolmogorov-Smirnov tests and writes `reports/drift_report.json`. A feature is
flagged only when `p < 0.01` and the KS statistic is at least `0.10`, avoiding
alerts for statistically significant but negligible shifts in this large sample.

5. Run the Evidently drift-monitoring job (optional dependency):

```bash
uv sync --extra monitoring
# Reference (training data) vs recent production inputs in Supabase:
DATABASE_URL=... uv run python scripts/run_drift_monitoring.py
# Or an offline comparison with no database:
uv run python scripts/run_drift_monitoring.py --current-parquet data/processed/test.parquet
```

This is a standalone job (meant to run on a schedule, e.g. GitHub Actions) and
never runs inside `/predict`. It loads the training reference, queries recent
`monitoring.predictions.inputs` from Postgres (connection from `DATABASE_URL`),
extracts the monitored fields, and runs Evidently's `DataDriftPreset`. It writes
`reports/drift/latest.html` (keep as a presentation artifact) and a small
`reports/drift/latest.json` summary for Grafana/alerts:

```json
{
  "generated_at": "...",
  "reference_rows": 10000,
  "current_rows": 500,
  "dataset_drift": true,
  "drifted_features": 3,
  "total_features": 14,
  "drift_share": 0.214,
  "features": {"gestation_weeks": {"drifted": true, "score": 0.18}}
}
```

If there are too few production rows, the job does not fail — it writes
`{"status": "insufficient_data", ...}` and exits cleanly. Evidently is not part
of the serving image; it stays in the `monitoring` extra.

This job runs on a schedule via
[`.github/workflows/drift-monitoring.yml`](../.github/workflows/drift-monitoring.yml)
(daily, plus manual `workflow_dispatch`). It pulls the reference data from DVC,
runs the drift job against production inputs, and uploads
`reports/drift/latest.{html,json}` as a build artifact. It requires repository
secrets `DATABASE_URL` (Supabase connection string) and the existing
`DAGSHUB_USER` / `DAGSHUB_TOKEN`. Without `DATABASE_URL` the job warns and skips
rather than failing.

When `DATABASE_URL` is set, each run also appends one row to the
`monitoring.drift` table (drift share, drifted-feature count, per-feature
scores, or just `status = insufficient_data`). The Grafana dashboard's
**"Evidently data drift"** panels read this table to trend drift share over time
and show the latest per-feature breakdown — alongside the live input-drift
panels fed directly from `monitoring.predictions`.

## What to monitor

- Service: availability, HTTP errors, and latency.
- Inputs: schema failures, missingness, and distribution drift.
- Outputs: predicted-weight distribution and risk-category frequency.
- Model: MAE and subgroup errors after delayed ground truth becomes available.

A small p-value alone does not imply harmful drift. With millions of records,
tiny changes can be statistically significant; inspect effect size and clinical
relevance before retraining.

## Maintenance

Retraining is considered when schema-valid new labeled data arrives, model
performance worsens materially, or important input distributions move. The new
model must repeat validation, held-out evaluation, subgroup checks, and
human approval before replacing the current artifact. Roll back when validation
or operational checks fail.
