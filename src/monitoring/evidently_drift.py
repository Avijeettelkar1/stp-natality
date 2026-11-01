"""Evidently data-drift monitoring for production inputs.

Compares the model's reference/training data against the recent production
prediction inputs logged in ``monitoring.predictions.inputs`` (Postgres), using
the open-source Evidently library (0.7+ API). It is a read-only monitoring job:
it never touches the prediction endpoint.

The pure helpers (feature extraction, reference loading, the insufficient-data
path, summary shaping) do not import Evidently, so they are importable and
testable without the optional ``monitoring`` extra installed. Evidently is
imported lazily inside :func:`build_report`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# The raw request fields we monitor for drift (as stored in
# monitoring.predictions.inputs and present in the processed reference data).
NUMERIC_FEATURES = [
    "gestation_weeks",
    "mother_age",
    "weight_gain_pounds",
    "father_age",
    "plurality",
    "year_fix",
    "month",
    "born_alive_alive",
    "born_alive_dead",
    "born_dead",
    "ever_born",
]
CATEGORICAL_FEATURES = ["is_male", "mother_married", "lmp_known"]
MONITORED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Minimum production rows before a drift comparison is meaningful.
DEFAULT_MIN_ROWS = 100


def extract_inputs(rows: list[dict]) -> pd.DataFrame:
    """Build a feature DataFrame from raw ``inputs`` JSON dicts.

    Each row is one ``monitoring.predictions.inputs`` payload. Booleans are
    coerced to 0/1 and missing fields become NaN, so the frame lines up with
    the numeric reference data regardless of which optional fields were sent.
    """
    records = []
    for row in rows:
        record = {}
        for feature in MONITORED_FEATURES:
            value = (row or {}).get(feature)
            if isinstance(value, bool):
                value = int(value)
            record[feature] = value
        records.append(record)

    frame = pd.DataFrame(records, columns=MONITORED_FEATURES)
    return frame.apply(pd.to_numeric, errors="coerce").astype("float64")


def load_reference(path: Path) -> pd.DataFrame:
    """Load the monitored features from the processed/training reference set."""
    frame = pd.read_parquet(path, columns=MONITORED_FEATURES)
    return frame.apply(pd.to_numeric, errors="coerce").astype("float64")


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Run Evidently's DataDriftPreset over the monitored features.

    Returns the Evidently run result (exposes ``save_html`` and ``dict``).
    Imports Evidently lazily so the rest of the module stays dependency-free.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    schema = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )
    reference_ds = Dataset.from_pandas(
        reference[MONITORED_FEATURES], data_definition=schema
    )
    current_ds = Dataset.from_pandas(
        current[MONITORED_FEATURES], data_definition=schema
    )
    report = Report([DataDriftPreset()])
    # Evidently 0.7 takes the current dataset first, the reference second.
    return report.run(current_ds, reference_ds)


def _round(value, digits: int = 6):
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def summarize(result, reference_rows: int, current_rows: int) -> dict:
    """Shape an Evidently result into the machine-readable summary contract."""
    metrics = result.dict().get("metrics", [])

    drift_share = 0.0
    drifted = 0
    drift_threshold = 0.5
    features: dict = {}

    for metric in metrics:
        config = metric.get("config", {})
        mtype = config.get("type", "")
        value = metric.get("value")
        if "DriftedColumnsCount" in mtype:
            drifted = int((value or {}).get("count", 0))
            drift_share = _round((value or {}).get("share"))
            drift_threshold = config.get("drift_share", 0.5)
        elif "ValueDrift" in mtype:
            method = (config.get("method") or "").lower()
            threshold = config.get("threshold")
            # p-value tests flag drift below the threshold; distance tests at/above.
            is_drift = value < threshold if "p_value" in method else value >= threshold
            features[config.get("column")] = {
                "drifted": bool(is_drift),
                "score": _round(value),
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "dataset_drift": bool(drift_share >= drift_threshold),
        "drifted_features": drifted or sum(f["drifted"] for f in features.values()),
        "total_features": len(features) or len(MONITORED_FEATURES),
        "drift_share": drift_share,
        "features": features,
    }


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    min_rows: int = DEFAULT_MIN_ROWS,
):
    """Compare reference vs current and return ``(result, summary)``.

    When there are fewer than ``min_rows`` production rows the job does not
    fail: it returns ``(None, {"status": "insufficient_data", ...})``.
    """
    current_rows = len(current)
    if current_rows < min_rows:
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "reference_rows": len(reference),
            "current_rows": current_rows,
            "required_rows": min_rows,
        }
        return None, summary

    result = build_report(reference, current)
    return result, summarize(result, len(reference), current_rows)


def write_outputs(result, summary: dict, html_path: Path, json_path: Path) -> None:
    """Persist the JSON summary always, and the HTML report when one exists."""
    import json

    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if result is not None:
        result.save_html(str(html_path))
