"""Evidently-based data-drift reporting (Evidently 0.7+ API).

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

import evidently
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.models.pipeline import BINARY_FEATURES, NUMERIC_FEATURES


def _to_dataset(df: pd.DataFrame, features: list[str]) -> Dataset:
    """Wrap a DataFrame as an Evidently Dataset with a typed data definition."""
    numerical = [f for f in NUMERIC_FEATURES if f in features and f in df.columns]
    categorical = [f for f in BINARY_FEATURES if f in features and f in df.columns]
    schema = DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
    )
    return Dataset.from_pandas(df[numerical + categorical], data_definition=schema)


def build_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]):
    """Run Evidently's DataDriftPreset and return the evaluation result.

    The result is the object returned by ``Report.run`` and exposes
    ``save_html`` and ``dict``.
    """
    report = Report([DataDriftPreset()])
    # Evidently 0.7 takes the current dataset first, the reference second.
    return report.run(
        _to_dataset(current, features),
        _to_dataset(reference, features),
    )


def summarize(result) -> dict:
    """Reduce an Evidently result to a compact, JSON-serialisable summary."""
    metrics = result.dict().get("metrics", [])

    n_drifted = 0
    share = 0.0
    drift_threshold = 0.5
    per_feature: dict = {}

    for metric in metrics:
        config = metric.get("config", {})
        mtype = config.get("type", "")
        value = metric.get("value")
        if "DriftedColumnsCount" in mtype:
            n_drifted = int((value or {}).get("count", 0))
            share = _round((value or {}).get("share"))
            drift_threshold = config.get("drift_share", 0.5)
        elif "ValueDrift" in mtype:
            method = config.get("method") or ""
            threshold = config.get("threshold")
            # p-value tests flag drift when the score is below the threshold;
            # distance tests flag drift when it is at or above the threshold.
            detected = (
                value < threshold if "p_value" in method.lower() else value >= threshold
            )
            per_feature[config.get("column")] = {
                "drift_detected": bool(detected),
                "drift_score": _round(value),
                "method": method,
                "threshold": threshold,
            }

    drifted = sorted(n for n, c in per_feature.items() if c["drift_detected"])
    return {
        "tool": "evidently",
        "evidently_version": evidently.__version__,
        "n_features": len(per_feature),
        "n_drifted": n_drifted or len(drifted),
        "share_drifted": share,
        "dataset_drift": bool(share >= drift_threshold),
        "drifted_features": drifted,
        "features": per_feature,
    }


def _round(value, digits: int = 6):
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def write_outputs(result, html_path: Path, json_path: Path) -> dict:
    """Save the HTML report and the JSON summary; return the summary dict."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(html_path))
    summary = summarize(result)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
