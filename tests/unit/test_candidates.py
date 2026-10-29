"""Unit tests for src/models/candidates.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.candidates import BASELINE_NAME, build_candidates, select_best
from src.models.pipeline import MODEL_FEATURES


@pytest.fixture
def small_training_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(200, len(MODEL_FEATURES))), columns=MODEL_FEATURES
    )
    y = pd.Series(3300 + 200 * X["gestation_weeks"] + rng.normal(0, 50, 200))
    return X, y


def test_candidates_include_baseline_and_all_models():
    names = set(build_candidates())
    assert names == {BASELINE_NAME, "ridge", "random_forest", "lightgbm", "xgboost"}


def test_every_candidate_fits_and_predicts(small_training_data):
    X, y = small_training_data
    for name, pipeline in build_candidates().items():
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X), name
        assert np.isfinite(predictions).all(), name


def test_candidates_are_fresh_on_each_call():
    first = build_candidates()
    second = build_candidates()
    assert first["lightgbm"] is not second["lightgbm"]


def test_select_best_picks_lowest_validation_mae():
    results = {
        "dummy_mean": {"val_mae": 436.6},
        "ridge": {"val_mae": 379.2},
        "lightgbm": {"val_mae": 366.6},
        "xgboost": {"val_mae": 371.0},
    }
    assert select_best(results) == "lightgbm"


def test_select_best_rejects_empty_results():
    with pytest.raises(ValueError):
        select_best({})
