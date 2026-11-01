"""Evidently-based data-drift reporting.

A richer complement to the Kolmogorov-Smirnov check in
:mod:`src.monitoring.drift`: it runs Evidently's ``DataDriftPreset`` over the
model features, which picks an appropriate statistical test per column, and
produces both an interactive HTML report and a compact JSON summary.

Evidently is an optional, heavier dependency (``pip install
'stp-natality[monitoring]'``) and is not imported by the serving API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> Report:
    """Run Evidently's DataDriftPreset over ``features`` and return the report."""
    columns = [f for f in features if f in reference.columns and f in current.columns]
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[columns],
        current_data=current[columns],
    )
    return report


def summarize(report: Report) -> dict:
    """Reduce an Evidently report to a compact, JSON-serialisable summary."""
    metrics = report.as_dict().get("metrics", [])

    dataset: dict = {}
    per_feature: dict = {}
    for metric in metrics:
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result and not dataset:
            dataset = result
        if "drift_by_columns" in result:
            for name, col in result["drift_by_columns"].items():
                per_feature[name] = {
                    "drift_detected": bool(col.get("drift_detected")),
                    "drift_score": _round(col.get("drift_score")),
                    "stattest": col.get("stattest_name"),
                }

    drifted = sorted(n for n, c in per_feature.items() if c["drift_detected"])
    return {
        "tool": "evidently",
        "n_features": dataset.get("number_of_columns", len(per_feature)),
        "n_drifted": dataset.get("number_of_drifted_columns", len(drifted)),
        "share_drifted": _round(dataset.get("share_of_drifted_columns")),
        "dataset_drift": bool(dataset.get("dataset_drift", bool(drifted))),
        "drifted_features": drifted,
        "features": per_feature,
    }


def _round(value, digits: int = 6):
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def write_outputs(report: Report, html_path: Path, json_path: Path) -> dict:
    """Save the HTML report and the JSON summary; return the summary dict."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    summary = summarize(report)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
