"""Unit tests for src/models/pipeline.py and src/models/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.models.evaluate import (
    compute_clinical_metrics,
    compute_regression_metrics,
    evaluate_split,
)
from src.models.pipeline import (
    BINARY_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    build_preprocessor,
    get_X_y,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame matching the processed parquet schema."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            # Numeric features
            "gestation_weeks": rng.uniform(30, 44, n),
            "mother_age": rng.integers(18, 45, n).astype(float),
            "weight_gain_pounds": rng.uniform(10, 50, n),
            "born_alive_alive": rng.integers(0, 4, n).astype(float),
            "born_alive_dead": rng.integers(0, 2, n).astype(float),
            "born_dead": rng.integers(0, 2, n).astype(float),
            "ever_born": rng.integers(1, 5, n).astype(float),
            "father_age": rng.uniform(18, 55, n),
            "plurality": rng.choice([1.0, 1.0, 1.0, 2.0], n),
            "parity": rng.integers(0, 4, n).astype(float),
            "year_fix": rng.integers(1990, 2018, n).astype(float),
            "month": rng.integers(1, 13, n).astype(float),
            "birth_month_sin": np.sin(2 * np.pi * rng.integers(1, 13, n) / 12),
            "birth_month_cos": np.cos(2 * np.pi * rng.integers(1, 13, n) / 12),
            # Binary features
            "is_male": rng.integers(0, 2, n).astype(float),
            "mother_married": rng.integers(0, 2, n).astype(float),
            "lmp_known": rng.integers(0, 2, n).astype(float),
            "is_multiple_birth": rng.integers(0, 2, n).astype(float),
            "gestation_preterm": rng.integers(0, 2, n).astype(float),
            "gestation_post_term": rng.integers(0, 2, n).astype(float),
            "mother_age_group": rng.integers(0, 4, n).astype(float),
            # Target
            "weight_grams": rng.normal(3400, 500, n),
            # Non-feature columns (should be dropped)
            "weight_pounds": rng.normal(7.5, 1.1, n),
            "source_year": 2000,
            "state": "CA",
        }
    )


# ── Tests: pipeline.py ─────────────────────────────────────────────────────────


def test_model_features_includes_numeric_and_binary():
    assert set(NUMERIC_FEATURES).issubset(set(MODEL_FEATURES))
    assert set(BINARY_FEATURES).issubset(set(MODEL_FEATURES))


def test_build_preprocessor_returns_column_transformer(sample_df):
    from sklearn.compose import ColumnTransformer

    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_preprocessor_with_scaling(sample_df):
    X, _ = get_X_y(sample_df)
    preprocessor = build_preprocessor(scale_numeric=True)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


def test_build_pipeline_returns_sklearn_pipeline(sample_df):
    from sklearn.pipeline import Pipeline

    model = DummyRegressor()
    pipe = build_pipeline(model)
    assert isinstance(pipe, Pipeline)


def test_build_pipeline_can_fit_and_predict(sample_df):
    X, y = get_X_y(sample_df)
    pipe = build_pipeline(DummyRegressor(strategy="mean"))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_get_X_y_selects_model_features(sample_df):
    X, y = get_X_y(sample_df)
    assert "weight_grams" not in X.columns
    assert "weight_pounds" not in X.columns
    assert "state" not in X.columns
    assert "source_year" not in X.columns


def test_get_X_y_target_is_weight_grams(sample_df):
    _, y = get_X_y(sample_df)
    assert y.name == "weight_grams"
    assert len(y) == len(sample_df)


def test_get_X_y_raises_for_missing_target(sample_df):
    df = sample_df.drop(columns=["weight_grams"])
    with pytest.raises(ValueError, match="weight_grams"):
        get_X_y(df)


def test_get_X_y_all_float64(sample_df):
    X, _ = get_X_y(sample_df)
    assert (X.dtypes == "float64").all(), "All feature columns must be float64"


def test_pipeline_handles_missing_values(sample_df):
    """Pipeline must not fail when features contain NaN."""
    sample_df.loc[sample_df.index[:20], "gestation_weeks"] = np.nan
    sample_df.loc[sample_df.index[:10], "father_age"] = np.nan
    X, y = get_X_y(sample_df)
    pipe = build_pipeline(DummyRegressor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert not np.isnan(preds).any()


# ── Tests: evaluate.py ─────────────────────────────────────────────────────────


def test_compute_regression_metrics_perfect_predictions():
    y = np.array([3000.0, 3500.0, 2800.0])
    metrics = compute_regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_compute_regression_metrics_keys():
    y = np.array([3000.0, 3500.0])
    y_pred = np.array([3100.0, 3400.0])
    metrics = compute_regression_metrics(y, y_pred)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}


def test_compute_regression_metrics_mae_value():
    y = np.array([3000.0, 4000.0])
    y_pred = np.array([2800.0, 4200.0])
    metrics = compute_regression_metrics(y, y_pred)
    assert abs(metrics["mae"] - 200.0) < 1e-6


def test_compute_clinical_metrics_all_correct():
    """Perfect predictions should yield sensitivity=1, specificity=1."""
    y_true = np.array([2000.0, 2000.0, 3500.0, 3500.0])
    metrics = compute_clinical_metrics(y_true, y_true)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_compute_clinical_metrics_all_wrong():
    """Predicting normal for all LBW → sensitivity=0."""
    y_true = np.array([2000.0, 2000.0])  # both LBW
    y_pred = np.array([3500.0, 3500.0])  # predicted normal
    metrics = compute_clinical_metrics(y_true, y_pred)
    assert metrics["sensitivity"] == 0.0


def test_compute_clinical_metrics_keys():
    y = np.array([2000.0, 3500.0])
    metrics = compute_clinical_metrics(y, y)
    expected = {
        "lbw_prevalence_pct",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "n_lbw_true",
        "n_lbw_pred",
    }
    assert expected == set(metrics.keys())


def test_compute_clinical_metrics_no_lbw():
    """All weights above threshold — sensitivity/precision edge case."""
    y = np.array([3000.0, 3500.0, 4000.0])
    metrics = compute_clinical_metrics(y, y)
    assert metrics["n_lbw_true"] == 0
    assert metrics["sensitivity"] == 0.0  # no positives to find


def test_evaluate_split_returns_prefixed_keys(sample_df):
    X, y = get_X_y(sample_df)
    pipe = build_pipeline(DummyRegressor(strategy="mean"))
    pipe.fit(X, y)
    result = evaluate_split(pipe, X, y, split_name="val")
    assert all(k.startswith("val_") for k in result.keys())


def test_evaluate_split_train_prefix(sample_df):
    X, y = get_X_y(sample_df)
    pipe = build_pipeline(DummyRegressor(strategy="mean"))
    pipe.fit(X, y)
    result = evaluate_split(pipe, X, y, split_name="train")
    assert "train_mae" in result
    assert "train_sensitivity" in result
