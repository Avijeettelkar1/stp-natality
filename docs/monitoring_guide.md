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
