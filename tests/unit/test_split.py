"""Unit tests for src/data/split.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    get_feature_columns,
    get_X_y,
    load_splits,
    save_splits,
    temporal_split,
)


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


# ── Tests: save_splits / load_splits ──────────────────────────────────────────


def test_save_splits_creates_parquet_files(tmp_path, sample_df):
    train, val, test = temporal_split(sample_df)
    save_splits(train, val, test, out_dir=tmp_path)
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()


def test_load_splits_returns_correct_shapes(tmp_path, sample_df):
    train, val, test = temporal_split(sample_df)
    save_splits(train, val, test, out_dir=tmp_path)
    train_l, val_l, test_l = load_splits(data_dir=tmp_path)
    assert len(train_l) == len(train)
    assert len(val_l) == len(val)
    assert len(test_l) == len(test)


def test_save_and_load_roundtrip_preserves_columns(tmp_path, sample_df):
    train, val, test = temporal_split(sample_df)
    save_splits(train, val, test, out_dir=tmp_path)
    train_l, _, _ = load_splits(data_dir=tmp_path)
    assert set(train_l.columns) == set(train.columns)


def test_load_splits_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(data_dir=tmp_path)


def test_temporal_split_empty_splits_do_not_raise(sample_df):
    """All-early data → empty val and test; _log_split_info must not crash."""
    early_df = sample_df[sample_df["year_fix"] < 2000].copy()
    train, val, test = temporal_split(early_df)
    assert len(train) > 0
    assert len(val) == 0
    assert len(test) == 0
