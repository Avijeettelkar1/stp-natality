"""Unit tests for src/data/split.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import get_feature_columns, get_X_y, temporal_split


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Multi-year DataFrame for split testing."""
    years = list(range(1990, 2020))
    n = len(years) * 100
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "year_fix": np.repeat(years, 100),
            "weight_grams": rng.normal(3400, 500, n),
            "weight_pounds": rng.normal(7.5, 1.1, n),
            "gestation_weeks": rng.integers(32, 42, n),
            "mother_age": rng.integers(18, 45, n),
            "is_male": rng.integers(0, 2, n),
            "source_year": 2000,
            "source_file": "newborn_2002",
            "lmp": None,
        }
    )


def test_temporal_split_returns_three_dfs(sample_df):
    train, val, test = temporal_split(sample_df)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_temporal_split_no_overlap(sample_df):
    train, val, test = temporal_split(sample_df)
    assert train["year_fix"].max() < 2010
    assert val["year_fix"].min() >= 2010
    assert val["year_fix"].max() < 2016
    assert test["year_fix"].min() >= 2016


def test_temporal_split_covers_all_rows(sample_df):
    train, val, test = temporal_split(sample_df)
    assert len(train) + len(val) + len(test) == len(sample_df)


def test_temporal_split_missing_year_col_raises():
    df = pd.DataFrame({"weight_grams": [3400]})
    with pytest.raises(ValueError, match="year_fix"):
        temporal_split(df)


def test_get_feature_columns_excludes_targets(sample_df):
    features = get_feature_columns(sample_df)
    assert "weight_grams" not in features
    assert "weight_pounds" not in features
    assert "source_year" not in features


def test_get_X_y_shapes(sample_df):
    X, y = get_X_y(sample_df)
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    assert "weight_grams" not in X.columns


def test_get_X_y_missing_target_raises(sample_df):
    df = sample_df.drop(columns=["weight_grams"])
    with pytest.raises(ValueError, match="weight_grams"):
        get_X_y(df)
