"""Small, reproducible batch-drift report for numeric model features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp


def compare_batches(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    alpha: float = 0.01,
) -> dict:
    """Compare numeric feature distributions with a two-sample KS test."""
    feature_results = {}
    for feature in features:
        reference_values = reference[feature].dropna()
        current_values = current[feature].dropna()
        if reference_values.empty or current_values.empty:
            feature_results[feature] = {
                "status": "insufficient_data",
                "ks_statistic": None,
                "p_value": None,
            }
            continue
        statistic, p_value = ks_2samp(reference_values, current_values)
        feature_results[feature] = {
            "status": "drift_detected" if p_value < alpha else "no_drift_detected",
            "ks_statistic": round(float(statistic), 6),
            "p_value": round(float(p_value), 8),
        }

    detected = [
        name
        for name, result in feature_results.items()
        if result["status"] == "drift_detected"
    ]
    return {
        "method": "two_sample_kolmogorov_smirnov",
        "alpha": alpha,
        "drift_detected": bool(detected),
        "drifted_features": detected,
        "features": feature_results,
    }


def write_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
