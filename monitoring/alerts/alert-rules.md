# Alert rules

Grafana-managed alert rules for the stp-natality service. Create them in
Grafana Cloud under **Alerting → Alert rules → New alert rule**, with the
team's e-mail as contact point. Thresholds come from the Support Plan.

| Rule | Data source | Query | Condition | For | Severity |
|---|---|---|---|---|---|
| Service down | Synthetic Monitoring | HTTP check on `/health` | check fails | 10 min | critical |
| Server errors | Prometheus | `sum(rate(stp_http_requests_total{status=~"5.."}[5m]))` | > 0 | 10 min | critical |
| Slow predictions | Prometheus | `histogram_quantile(0.95, sum by (le) (rate(stp_http_request_duration_seconds_bucket{path="/predict"}[5m])))` | > 2 (seconds) | 15 min | warning |
| No model loaded | Prometheus | `sum(increase(stp_predictions_total{result="champion_error"}[15m]))` | > 0 | 0 min | critical |
| Prediction log failing | Prometheus | `sum(increase(stp_monitoring_write_errors_total[1h]))` | > 0 | 0 min | warning |
| No traffic | Supabase (PostgreSQL) | `SELECT count(*) FROM monitoring.predictions WHERE created_at > now() - interval '24 hours'` | < 1 | 0 min | info |
| Shadow disagreement | Supabase (PostgreSQL) | `SELECT avg(abs(champion_prediction_g - challenger_prediction_g)) FROM monitoring.predictions WHERE created_at > now() - interval '24 hours' AND challenger_prediction_g IS NOT NULL` | > 200 (grams) | 0 min | warning |

## Responses

- **Service down / no model loaded:** check the latest CD run and Render deploy
  log; if the last deployment broke the service, run the Rollback workflow
  with the previous commit SHA.
- **Server errors / slow predictions:** check Render logs and metrics; Render's
  free tier has 0.1 CPU, so sustained load is expected to be slow.
- **Prediction log failing:** check that `DATABASE_URL` is set in Render and the
  Supabase project is not paused. Predictions keep working without the log.
- **No traffic:** expected outside demos; confirms the alert path works.
- **Shadow disagreement:** the challenger behaves differently from the
  champion on live inputs; review before any promotion.

Data drift and retraining are handled by the scheduled monitoring job, not by
Grafana alerts.
