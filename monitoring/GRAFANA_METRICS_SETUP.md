# Part 6 - Connect `/metrics` to Grafana Cloud

The API is already instrumented and exposes Prometheus metrics at:

`https://stp-natality.onrender.com/metrics`

No change to `src/api/main.py` is required for this part.

## 1. Check the Render endpoint first

Open this URL in a browser:

`https://stp-natality.onrender.com/metrics`

You should see Prometheus text output. After the service has handled traffic, metric families include:

- `stp_http_requests_total`
- `stp_http_request_duration_seconds`
- `stp_predictions_total`
- `stp_shadow_prediction_difference_grams`
- `stp_monitoring_write_errors_total`

You can also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_grafana_metrics.ps1
```

## 2. Add the Metrics Endpoint integration in Grafana Cloud

In your Grafana Cloud stack:

1. Open **Connections**.
2. Choose **Add new connection**.
3. Search for **Metrics Endpoint**.
4. Open the Metrics Endpoint integration.
5. Create a new scrape job.
6. Use this URL:

   `https://stp-natality.onrender.com/metrics`

7. Authentication: **None**.
8. Use a scrape interval of **60 seconds** for this student project.
9. Save/start the scrape job.

Grafana Cloud will scrape the public Prometheus-compatible URL and store the samples in the stack's hosted Prometheus service. No separate Prometheus server or Grafana Alloy instance is required for this public endpoint setup.

## 3. Verify ingestion in Grafana Explore

Wait for the first scrape, then open **Explore** and choose the Grafana Cloud Prometheus data source.

Run these PromQL queries one by one:

```promql
sum by (path) (rate(stp_http_requests_total[5m]))
```

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(stp_http_request_duration_seconds_bucket[5m]))
)
```

```promql
sum(rate(stp_predictions_total{result="success"}[5m])) * 60
```

```promql
sum(rate(stp_http_requests_total{status=~"5.."}[5m])) or vector(0)
```

```promql
sum by (operation) (
  increase(stp_monitoring_write_errors_total[1h])
) or vector(0)
```

The shadow-model query will only contain useful data after the challenger is deployed:

```promql
sum(rate(stp_shadow_prediction_difference_grams_sum[15m]))
/
sum(rate(stp_shadow_prediction_difference_grams_count[15m]))
```

## 4. Import/use the existing dashboard

The repository already contains:

`monitoring/dashboards/stp-natality-monitoring.json`

In Grafana:

1. **Dashboards -> New -> Import**.
2. Upload `monitoring/dashboards/stp-natality-monitoring.json`.
3. For **Supabase (Postgres)** choose your existing Supabase PostgreSQL data source.
4. For **Grafana Cloud Prometheus** choose the hosted Prometheus data source that receives the Metrics Endpoint scrape.
5. Import.

The dashboard already contains application panels for:

- request rate by path
- 5xx error rate
- p50/p95 latency
- successful predictions per minute
- champion prediction errors
- average champion/challenger difference
- monitoring database write failures

## 5. Generate traffic if panels are empty

If the service has just restarted or there are no recent requests, make a few requests to:

`https://stp-natality.onrender.com/health`

and make one normal prediction from the website.

Then check Explore again.

## 6. Definition of done

Part 6 is complete when all of these are true:

- `/metrics` is publicly reachable on Render.
- The Grafana Cloud Metrics Endpoint scrape job reports healthy/successful.
- `stp_http_requests_total` returns data in Explore.
- p95 latency returns data after requests have been made.
- `stp_predictions_total{result="success"}` increases after a website prediction.
- The imported dashboard's Prometheus panels show data.

The shadow-difference panel is allowed to be empty until the challenger model is deployed.
