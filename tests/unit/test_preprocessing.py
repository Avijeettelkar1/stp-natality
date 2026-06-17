"""Unit tests for src/data/preprocess.py and src/data/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_cyclical_month,
    add_gestation_flags,
    add_lmp_known_flag,
    add_mother_age_group,
    add_multiple_birth_flag,
    add_parity,
    engineer_features,
)
from src.data.preprocess import (
    cap_outliers,
    cast_booleans,
    convert_target,
    drop_leakage_columns,
    drop_null_targets,
    handle_missing_values,
    run_preprocessing,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a small representative raw natality DataFrame for testing."""
    return pd.DataFrame(
        {
            "year_fix": [1990, 2000, 2010, 2018, 2005],
            "month": [1, 6, 11, 3, 8],
            "day": [15, 22, 3, 17, 9],
            "weight_pounds": [7.5, 5.2, 8.1, 4.0, np.nan],
            "plurality": [1, 1, 2, 1, 1],
            "is_male": ["true", "false", "true", "false", "true"],
            "apgar_1min": [8, 9, 7, 8, np.nan],
            "apgar_5min": [9, 10, 9, 9, np.nan],
            "mother_age": [28, 35, 22, 42, 30],
            "mother_married": ["true", "false", "true", "true", "false"],
            "gestation_weeks": [39, 38, 32, 40, 47],  # 47 = outlier
            "lmp": ["01012000", "9999", "03152010", None, "99"],
            "weight_gain_pounds": [25.0, 18.0, 30.0, 22.0, 15.0],
            "born_alive_alive": [1, 0, 2, 3, 0],
            "born_alive_dead": [0, 0, 0, 1, 0],
            "born_dead": [0, 1, 0, 0, 0],
            "ever_born": [2, 1, 3, 5, 1],
            "father_age": [30, 99, 28, 45, 33],  # 99 = unknown
            "record_weight": [1, 1, 2, 1, 2],
            "cigarette_use": [None, None, None, None, None],
            "cigarettes_per_day": [None, None, None, None, None],
            "alcohol_use": [None, None, None, None, None],
            "drinks_per_week": [None, None, None, None, None],
            "mother_residence_state": ["CA", "TX", "NY", "FL", "WA"],
            "mother_birth_state": ["CA", "MX", "NY", "FL", "OH"],
            "state": ["CA", "TX", "NY", "FL", "WA"],
        }
    )


# ── Tests: drop_leakage_columns ────────────────────────────────────────────────


def test_drop_leakage_removes_apgar_columns(sample_raw_df):
    result = drop_leakage_columns(sample_raw_df)
    assert "apgar_1min" not in result.columns
    assert "apgar_5min" not in result.columns


def test_drop_leakage_removes_day_column(sample_raw_df):
    result = drop_leakage_columns(sample_raw_df)
    assert "day" not in result.columns


def test_drop_leakage_preserves_other_columns(sample_raw_df):
    result = drop_leakage_columns(sample_raw_df)
    assert "weight_pounds" in result.columns
    assert "gestation_weeks" in result.columns
    assert "mother_age" in result.columns


# ── Tests: cast_booleans ───────────────────────────────────────────────────────


def test_cast_booleans_converts_is_male(sample_raw_df):
    result = cast_booleans(sample_raw_df)
    assert result["is_male"].dtype == pd.Int8Dtype()
    assert result["is_male"].iloc[0] == 1  # "true" → 1
    assert result["is_male"].iloc[1] == 0  # "false" → 0


def test_cast_booleans_converts_mother_married(sample_raw_df):
    result = cast_booleans(sample_raw_df)
    assert result["mother_married"].iloc[0] == 1
    assert result["mother_married"].iloc[1] == 0


def test_cast_booleans_handles_none_columns(sample_raw_df):
    """cigarette_use is all-None — should not raise an error."""
    result = cast_booleans(sample_raw_df)
    assert "cigarette_use" in result.columns


# ── Tests: handle_missing_values ──────────────────────────────────────────────


def test_handle_missing_father_age_99(sample_raw_df):
    result = handle_missing_values(sample_raw_df)
    # Row 1 had father_age = 99, should now be NaN
    assert pd.isna(result["father_age"].iloc[1])


def test_handle_missing_gestation_47_not_sentinel(sample_raw_df):
    """gestation_weeks=47 is an outlier but NOT a sentinel (99).

    Should not be set to NaN here — that happens in cap_outliers.
    """
    result = handle_missing_values(sample_raw_df)
    assert result["gestation_weeks"].iloc[4] == 47  # still 47 before cap_outliers


def test_handle_missing_lmp_9999(sample_raw_df):
    """lmp values of '9999' should become NaN."""
    result = handle_missing_values(sample_raw_df)
    assert pd.isna(result["lmp"].iloc[1])  # "9999" → NaN


def test_handle_missing_lmp_99(sample_raw_df):
    """lmp values of '99' should become NaN."""
    result = handle_missing_values(sample_raw_df)
    assert pd.isna(result["lmp"].iloc[4])  # "99" → NaN


# ── Tests: cap_outliers ────────────────────────────────────────────────────────


def test_cap_outliers_clips_gestation_weeks(sample_raw_df):
    result = cap_outliers(sample_raw_df)
    assert result["gestation_weeks"].max() <= 44


def test_cap_outliers_adds_outlier_weight_flag(sample_raw_df):
    result = cap_outliers(sample_raw_df)
    assert "outlier_weight" in result.columns


# ── Tests: convert_target ──────────────────────────────────────────────────────


def test_convert_target_adds_weight_grams(sample_raw_df):
    result = convert_target(sample_raw_df)
    assert "weight_grams" in result.columns


def test_convert_target_correct_value():
    df = pd.DataFrame({"weight_pounds": [1.0]})
    result = convert_target(df)
    assert abs(result["weight_grams"].iloc[0] - 453.592) < 0.01


def test_convert_target_handles_nan():
    df = pd.DataFrame({"weight_pounds": [np.nan]})
    result = convert_target(df)
    assert pd.isna(result["weight_grams"].iloc[0])


# ── Tests: drop_null_targets ──────────────────────────────────────────────────


def test_drop_null_targets_removes_nan_weight(sample_raw_df):
    result = drop_null_targets(sample_raw_df)
    # Row 4 had weight_pounds = NaN → should be removed
    assert result["weight_pounds"].isna().sum() == 0


def test_drop_null_targets_preserves_valid_rows(sample_raw_df):
    result = drop_null_targets(sample_raw_df)
    assert len(result) == 4  # 5 rows minus the NaN row


# ── Tests: run_preprocessing (integration) ───────────────────────────────────


def test_run_preprocessing_returns_dataframe(sample_raw_df):
    result = run_preprocessing(sample_raw_df)
    assert isinstance(result, pd.DataFrame)


def test_run_preprocessing_no_null_targets(sample_raw_df):
    result = run_preprocessing(sample_raw_df)
    assert result["weight_grams"].isna().sum() == 0


def test_run_preprocessing_no_leakage_cols(sample_raw_df):
    result = run_preprocessing(sample_raw_df)
    assert "apgar_1min" not in result.columns
    assert "apgar_5min" not in result.columns


# ── Tests: feature_engineering ────────────────────────────────────────────────


@pytest.fixture
def clean_df(sample_raw_df) -> pd.DataFrame:
    """Return a preprocessed DataFrame for feature engineering tests."""
    return run_preprocessing(sample_raw_df)


def test_add_parity_is_sum_of_prior_births(clean_df):
    result = add_parity(clean_df)
    assert "parity" in result.columns
    # Row 0: born_alive_alive=1, born_alive_dead=0, born_dead=0 → parity=1
    assert result["parity"].iloc[0] == 1


def test_add_multiple_birth_flag_detects_twins(clean_df):
    result = add_multiple_birth_flag(clean_df)
    assert "is_multiple_birth" in result.columns
    # Row 2 had plurality=2 → is_multiple_birth=1
    twin_row = result[result["plurality"] == 2]
    assert twin_row["is_multiple_birth"].iloc[0] == 1


def test_add_cyclical_month_range(clean_df):
    result = add_cyclical_month(clean_df)
    assert "birth_month_sin" in result.columns
    assert "birth_month_cos" in result.columns
    assert result["birth_month_sin"].between(-1, 1).all()
    assert result["birth_month_cos"].between(-1, 1).all()


def test_add_lmp_known_flag_binary(clean_df):
    result = add_lmp_known_flag(clean_df)
    assert "lmp_known" in result.columns
    assert set(result["lmp_known"].dropna().unique()).issubset({0, 1})


def test_add_mother_age_group_in_range(clean_df):
    result = add_mother_age_group(clean_df)
    assert "mother_age_group" in result.columns
    valid_groups = {0, 1, 2, 3}
    actual = set(result["mother_age_group"].dropna().unique())
    assert actual.issubset(valid_groups)


def test_add_gestation_flags_preterm(clean_df):
    result = add_gestation_flags(clean_df)
    assert "gestation_preterm" in result.columns
    # Row 2 had gestation_weeks=32 → preterm
    preterm_rows = result[result["gestation_weeks"] < 37]
    assert (preterm_rows["gestation_preterm"] == 1).all()


def test_add_weight_gain_kg_conversion(clean_df):
    """weight_gain_kg should equal weight_gain_pounds × 0.453592."""
    from src.data.feature_engineering import add_weight_gain_kg

    result = add_weight_gain_kg(clean_df)
    assert "weight_gain_kg" in result.columns
    # Verify conversion factor within floating-point tolerance
    expected = clean_df["weight_gain_pounds"] * 0.453592
    diff = (result["weight_gain_kg"] - expected.round(2)).abs()
    assert diff.max() < 0.01, f"Unexpected kg conversion error: max diff = {diff.max()}"


def test_add_weight_gain_kg_non_negative(clean_df):
    """weight_gain_kg should not be negative (physical constraint)."""
    from src.data.feature_engineering import add_weight_gain_kg

    result = add_weight_gain_kg(clean_df)
    non_null = result["weight_gain_kg"].dropna()
    assert (non_null >= 0).all()


def test_engineer_features_runs_without_error(clean_df):
    result = engineer_features(clean_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(clean_df)


def test_engineer_features_adds_all_expected_columns(clean_df):
    """engineer_features should add all 9 expected derived features."""
    result = engineer_features(clean_df)
    expected_new_cols = [
        "weight_gain_kg",
        "parity",
        "is_multiple_birth",
        "lmp_known",
        "birth_month_sin",
        "birth_month_cos",
        "mother_age_group",
        "gestation_preterm",
        "gestation_post_term",
    ]
    for col in expected_new_cols:
        assert col in result.columns, f"Expected engineered feature '{col}' not found"
