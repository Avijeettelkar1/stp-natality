"""Unit tests for the Evidently drift helper.

Skipped automatically where Evidently is not installed (e.g. the CI extra),
so it never breaks the lint-and-test job.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from src.monitoring.evidently_drift import (  # noqa: E402
    build_report,
    summarize,
)

FEATURES = ["gestation_weeks", "mother_age"]


def _frame(mean: float, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "gestation_weeks": rng.normal(mean, 1.5, n),
            "mother_age": rng.normal(29, 5, n),
        }
    )


def test_summary_has_expected_shape():
    report = build_report(_frame(39), _frame(39), FEATURES)
    summary = summarize(report)
    assert set(summary) >= {
        "tool",
        "n_features",
        "n_drifted",
        "share_drifted",
        "dataset_drift",
        "drifted_features",
        "features",
    }
    assert summary["tool"] == "evidently"
    assert set(summary["features"]) == set(FEATURES)


def test_detects_shifted_distribution():
    # Shift gestation strongly; it should be flagged as drifted.
    report = build_report(_frame(39), _frame(33), FEATURES)
    summary = summarize(report)
    assert "gestation_weeks" in summary["drifted_features"]
    assert summary["dataset_drift"] is True


def test_no_drift_for_identical_distributions():
    report = build_report(_frame(39), _frame(39), FEATURES)
    summary = summarize(report)
    assert summary["features"]["gestation_weeks"]["drift_detected"] is False
