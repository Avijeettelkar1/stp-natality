"""Unit tests for the Evidently drift-monitoring module.

The pure helpers (extraction, insufficient-data path) run everywhere, including
CI where the optional ``evidently`` extra is not installed. The tests that
actually build a report are skipped when Evidently is missing.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.evidently_drift import (
    CATEGORICAL_FEATURES,
    MONITORED_FEATURES,
    drift_row_from_summary,
    extract_inputs,
    run_drift,
)

# ---- production input extraction -------------------------------------------


def test_extract_inputs_builds_monitored_columns():
    rows = [
        {
            "gestation_weeks": 39,
            "mother_age": 30,
            "weight_gain_pounds": 28,
            "father_age": 32,
            "plurality": 1,
            "year_fix": 2021,
            "month": 6,
            "born_alive_alive": 1,
            "born_alive_dead": 0,
            "born_dead": 0,
            "ever_born": 2,
            "is_male": True,
            "mother_married": False,
            "lmp_known": True,
            "extra_ignored_field": "x",
        }
    ]
    df = extract_inputs(rows)
    assert list(df.columns) == MONITORED_FEATURES
    assert len(df) == 1
    # Booleans coerced to 0/1 floats.
    assert df.loc[0, "is_male"] == 1.0
    assert df.loc[0, "mother_married"] == 0.0
    assert all(df[c].dtype == "float64" for c in MONITORED_FEATURES)


def test_extract_inputs_missing_optional_fields_become_nan():
    rows = [{"gestation_weeks": 40, "mother_age": 28}]  # most fields absent
    df = extract_inputs(rows)
    assert df.loc[0, "gestation_weeks"] == 40.0
    assert np.isnan(df.loc[0, "father_age"])
    assert np.isnan(df.loc[0, "mother_married"])


def test_extract_inputs_empty_rows():
    df = extract_inputs([])
    assert list(df.columns) == MONITORED_FEATURES
    assert len(df) == 0


# ---- insufficient production data (must not fail) --------------------------


def test_run_drift_returns_insufficient_data_when_too_few_rows():
    reference = _reference(500)
    current = extract_inputs([{"gestation_weeks": 39, "mother_age": 30}] * 3)
    result, summary = run_drift(reference, current, min_rows=100)
    assert result is None
    assert summary["status"] == "insufficient_data"
    assert summary["current_rows"] == 3
    assert summary["required_rows"] == 100
    assert "generated_at" in summary


# ---- monitoring.drift row mapping ------------------------------------------


def test_drift_row_from_full_summary():
    summary = {
        "generated_at": "2026-11-02T00:00:00+00:00",
        "reference_rows": 299431,
        "current_rows": 256,
        "dataset_drift": True,
        "drifted_features": 9,
        "total_features": 14,
        "drift_share": 0.64,
        "features": {"gestation_weeks": {"drifted": True, "score": 0.18}},
    }
    row = drift_row_from_summary(summary)
    assert row["status"] == "ok"
    assert row["drift_share"] == 0.64
    assert row["dataset_drift"] is True
    assert row["features"]["gestation_weeks"]["drifted"] is True


def test_drift_row_from_insufficient_summary():
    summary = {
        "generated_at": "2026-11-02T00:00:00+00:00",
        "status": "insufficient_data",
        "reference_rows": 299431,
        "current_rows": 5,
        "required_rows": 100,
    }
    row = drift_row_from_summary(summary)
    assert row["status"] == "insufficient_data"
    # Drift fields absent -> None, so they land as SQL NULLs.
    assert row["drift_share"] is None
    assert row["dataset_drift"] is None
    assert row["features"] is None


# ---- expected JSON summary structure (needs Evidently) ---------------------


def test_summary_structure_and_drift_detection():
    pytest.importorskip("evidently")
    reference = _reference(500, gestation_mean=39)
    current = _reference(300, gestation_mean=33)  # shift gestation strongly
    result, summary = run_drift(reference, current, min_rows=50)
    assert result is not None
    assert set(summary) >= {
        "generated_at",
        "reference_rows",
        "current_rows",
        "dataset_drift",
        "drifted_features",
        "total_features",
        "drift_share",
        "features",
    }
    assert summary["reference_rows"] == 500
    assert summary["current_rows"] == 300
    assert summary["total_features"] == len(MONITORED_FEATURES)
    # Every feature entry has the drifted/score contract.
    for entry in summary["features"].values():
        assert set(entry) == {"drifted", "score"}
    assert summary["features"]["gestation_weeks"]["drifted"] is True


def _reference(n: int, gestation_mean: float = 39) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "gestation_weeks": rng.normal(gestation_mean, 1.5, n),
        "mother_age": rng.normal(29, 5, n),
        "weight_gain_pounds": rng.normal(30, 10, n),
        "father_age": rng.normal(32, 6, n),
        "plurality": rng.choice([1, 1, 1, 2], n).astype(float),
        "year_fix": rng.choice([2018, 2019, 2020], n).astype(float),
        "month": rng.integers(1, 13, n).astype(float),
        "born_alive_alive": rng.integers(0, 3, n).astype(float),
        "born_alive_dead": np.zeros(n),
        "born_dead": np.zeros(n),
        "ever_born": rng.integers(1, 4, n).astype(float),
    }
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(data)[MONITORED_FEATURES]
