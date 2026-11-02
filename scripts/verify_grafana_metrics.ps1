param(
    [string]$MetricsUrl = "https://stp-natality.onrender.com/metrics",
    [string]$HealthUrl = "https://stp-natality.onrender.com/health"
)

$ErrorActionPreference = "Stop"

Write-Host "Warming the API with a health request..."
try {
    Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 30 | Out-Null
} catch {
    Write-Warning "Health request failed: $($_.Exception.Message)"
}

Write-Host "Reading Prometheus metrics from $MetricsUrl ..."
$response = Invoke-WebRequest -Uri $MetricsUrl -UseBasicParsing -TimeoutSec 30

if ($response.StatusCode -ne 200) {
    throw "Metrics endpoint returned HTTP $($response.StatusCode)."
}

$content = $response.Content

if ($content -notmatch "# HELP" -or $content -notmatch "# TYPE") {
    throw "Endpoint responded, but it does not look like Prometheus exposition text."
}

$required = @(
    "stp_http_requests_total",
    "stp_http_request_duration_seconds"
)

$optionalUntilUsed = @(
    "stp_predictions_total",
    "stp_shadow_prediction_difference_grams",
    "stp_monitoring_write_errors_total"
)

Write-Host ""
Write-Host "Required metric families:"
foreach ($metric in $required) {
    if ($content -match [regex]::Escape($metric)) {
        Write-Host "  [OK] $metric"
    } else {
        Write-Host "  [MISSING] $metric"
        throw "Required metric family '$metric' was not found."
    }
}

Write-Host ""
Write-Host "Metrics that appear after the relevant feature is used:"
foreach ($metric in $optionalUntilUsed) {
    if ($content -match [regex]::Escape($metric)) {
        Write-Host "  [OK] $metric"
    } else {
        Write-Host "  [NOT SEEN YET] $metric"
    }
}

Write-Host ""
Write-Host "PASS: /metrics is reachable and exposes Prometheus-formatted application metrics."
Write-Host "Next: Grafana Cloud -> Connections -> Add new connection -> Metrics Endpoint."
Write-Host "Scrape URL: $MetricsUrl"
