"""Data validation tests for the STP Natality processed dataset.

These tests validate data quality constraints on the processed Parquet splits
produced by notebooks/03_data_preparation.ipynb. Run after the preprocessing
pipeline has been executed.

Tests are grouped by:
- Schema constraints (columns, dtypes)
- Value range constraints (clinical validity)
- Integrity constraints (no duplicates, no temporal overlap)
- Leakage checks (excluded columns)

Usage:
    # Run only if processed data exists:
    pytest tests/data_validation/ -v

    # Skip if processed data not found (CI-safe):
    pytest tests/data_validation/ -v -m "not requires_processed_data"
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

# ── Clinical constants ─────────────────────────────────────────────────────────
WEIGHT_GRAMS_MIN = 300.0  # ~0.66 lbs — clinical lower bound
WEIGHT_GRAMS_MAX = 7000.0  # ~15.43 lbs — clinical upper bound
GESTATION_WEEKS_MIN = 20
GESTATION_WEEKS_MAX = 44
MOTHER_AGE_MIN = 10
MOTHER_AGE_MAX = 60

# ── Columns that MUST NOT appear in processed data ─────────────────────────────
LEAKAGE_COLUMNS = ["apgar_1min", "apgar_5min"]

# ── Required columns in processed data ────────────────────────────────────────
REQUIRED_COLUMNS = [
    "weight_grams",
    "weight_pounds",
    "gestation_weeks",
    "mother_age",
    "is_male",
    "plurality",
    "year_fix",
    "month",
    "source_year",
    "source_file",
]

# ── Split year thresholds (must match split.py) ────────────────────────────────
VAL_START_YEAR = 2010
TEST_START_YEAR = 2016


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _load_split(path: Path) -> pd.DataFrame:
    """Load a single parquet split; skip test if file doesn't exist."""
    if not path.exists():
        pytest.skip(
            f"Processed data not found at {path}. "
            "Run notebooks/03_data_preparation.ipynb first."
        )
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return _load_split(TRAIN_PATH)


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    return _load_split(VAL_PATH)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return _load_split(TEST_PATH)


@pytest.fixture(scope="module")
def all_splits(
    train_df, val_df, test_df
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return train_df, val_df, test_df


# ── Schema Tests ───────────────────────────────────────────────────────────────


class TestSchema:
    """Verify all required columns are present in every split."""

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_required_column_in_train(self, train_df, col):
        assert (
            col in train_df.columns
        ), f"Missing required column '{col}' in train split"

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_required_column_in_val(self, val_df, col):
        assert col in val_df.columns, f"Missing required column '{col}' in val split"

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_required_column_in_test(self, test_df, col):
        assert col in test_df.columns, f"Missing required column '{col}' in test split"

    @pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
    def test_no_leakage_column_in_train(self, train_df, col):
        assert (
            col not in train_df.columns
        ), f"Leakage column '{col}' found in train split!"

    @pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
    def test_no_leakage_column_in_val(self, val_df, col):
        assert col not in val_df.columns, f"Leakage column '{col}' found in val split!"

    @pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
    def test_no_leakage_column_in_test(self, test_df, col):
        assert (
            col not in test_df.columns
        ), f"Leakage column '{col}' found in test split!"

    def test_all_splits_have_same_columns(self, train_df, val_df, test_df):
        """All three splits must have identical column sets."""
        assert set(train_df.columns) == set(val_df.columns) == set(test_df.columns), (
            "Column mismatch across splits!\n"
            f"  train: {sorted(train_df.columns)}\n"
            f"  val:   {sorted(val_df.columns)}\n"
            f"  test:  {sorted(test_df.columns)}"
        )


# ── Target Variable Tests ──────────────────────────────────────────────────────


class TestTargetVariable:
    """Validate the primary target variable: weight_grams."""

    @pytest.mark.parametrize(
        "split_name,fixture_name",
        [("train", "train_df"), ("val", "val_df"), ("test", "test_df")],
    )
    def test_no_null_weight_grams(self, request, split_name, fixture_name):
        df = request.getfixturevalue(fixture_name)
        null_count = df["weight_grams"].isna().sum()
        assert (
            null_count == 0
        ), f"{null_count} null weight_grams values in {split_name} split"

    def test_weight_grams_min_train(self, train_df):
        assert (
            train_df["weight_grams"].min() >= WEIGHT_GRAMS_MIN
        ), f"weight_grams below minimum {WEIGHT_GRAMS_MIN}g in train"

    def test_weight_grams_max_train(self, train_df):
        assert (
            train_df["weight_grams"].max() <= WEIGHT_GRAMS_MAX
        ), f"weight_grams above maximum {WEIGHT_GRAMS_MAX}g in train"

    def test_weight_grams_min_val(self, val_df):
        assert val_df["weight_grams"].min() >= WEIGHT_GRAMS_MIN

    def test_weight_grams_max_val(self, val_df):
        assert val_df["weight_grams"].max() <= WEIGHT_GRAMS_MAX

    def test_weight_grams_min_test(self, test_df):
        assert test_df["weight_grams"].min() >= WEIGHT_GRAMS_MIN

    def test_weight_grams_max_test(self, test_df):
        assert test_df["weight_grams"].max() <= WEIGHT_GRAMS_MAX

    def test_weight_grams_reasonable_mean(self, train_df):
        """Mean birth weight should be in a plausible clinical range (2800–3800g)."""
        mean_g = train_df["weight_grams"].mean()
        assert (
            2800 <= mean_g <= 3800
        ), f"Unexpected mean weight_grams: {mean_g:.0f}g (expected 2800–3800g)"

    def test_lbw_rate_plausible(self, train_df):
        """LBW rate should be between 5% and 20% (historical US range)."""
        lbw_rate = (train_df["weight_grams"] < 2500).mean() * 100
        assert (
            3.0 <= lbw_rate <= 20.0
        ), f"Implausible LBW rate: {lbw_rate:.2f}% (expected 3–20%)"


# ── Clinical Feature Range Tests ───────────────────────────────────────────────


class TestFeatureRanges:
    """Validate that clinical features fall within valid ranges."""

    def test_gestation_weeks_range_train(self, train_df):
        col = train_df["gestation_weeks"].dropna()
        assert col.min() >= GESTATION_WEEKS_MIN
        assert col.max() <= GESTATION_WEEKS_MAX

    def test_mother_age_range_train(self, train_df):
        col = train_df["mother_age"].dropna()
        assert col.min() >= MOTHER_AGE_MIN
        assert col.max() <= MOTHER_AGE_MAX

    def test_is_male_values(self, train_df):
        """is_male must be 0 or 1 (or NaN)."""
        valid = {0, 1}
        actual = set(train_df["is_male"].dropna().astype(int).unique())
        assert actual.issubset(valid), f"Unexpected is_male values: {actual - valid}"

    def test_plurality_positive(self, train_df):
        """Plurality must be >= 1 (no zero or negative births)."""
        non_null = train_df["plurality"].dropna()
        assert (non_null >= 1).all(), "plurality values < 1 found in train"

    def test_plurality_max(self, train_df):
        """Plurality > 9 is implausible."""
        non_null = train_df["plurality"].dropna()
        assert (non_null <= 9).all(), "plurality > 9 found in train"

    def test_mother_married_values(self, train_df):
        """mother_married must be 0 or 1 (or NaN)."""
        if "mother_married" in train_df.columns:
            valid = {0, 1}
            actual = set(train_df["mother_married"].dropna().astype(int).unique())
            assert actual.issubset(valid)

    def test_engineered_flags_binary(self, train_df):
        """Engineered flag columns must be 0 or 1."""
        flag_cols = [
            c
            for c in [
                "gestation_preterm",
                "gestation_post_term",
                "is_multiple_birth",
                "lmp_known",
            ]
            if c in train_df.columns
        ]
        for col in flag_cols:
            valid = {0, 1}
            actual = set(train_df[col].dropna().astype(int).unique())
            assert actual.issubset(
                valid
            ), f"{col} has non-binary values: {actual - valid}"

    def test_mother_age_group_range(self, train_df):
        """mother_age_group must be 0–3."""
        if "mother_age_group" in train_df.columns:
            valid = {0, 1, 2, 3}
            actual = set(train_df["mother_age_group"].dropna().astype(int).unique())
            assert actual.issubset(valid)

    def test_cyclical_month_range(self, train_df):
        """sin/cos cyclical month encodings must be in [-1, 1]."""
        for col in ["birth_month_sin", "birth_month_cos"]:
            if col in train_df.columns:
                vals = train_df[col].dropna()
                assert vals.between(
                    -1.0, 1.0
                ).all(), f"{col} has values outside [-1, 1]"


# ── Temporal Integrity Tests ───────────────────────────────────────────────────


class TestTemporalIntegrity:
    """Verify no temporal overlap between splits and correct year boundaries."""

    def test_train_max_year_before_val(self, train_df, val_df):
        assert train_df["year_fix"].max() < VAL_START_YEAR, (
            f"Train contains records from year {train_df['year_fix'].max()} "
            f"which overlaps with val start ({VAL_START_YEAR})"
        )

    def test_val_year_range(self, val_df):
        assert val_df["year_fix"].min() >= VAL_START_YEAR
        assert val_df["year_fix"].max() < TEST_START_YEAR

    def test_test_year_min(self, test_df):
        assert test_df["year_fix"].min() >= TEST_START_YEAR, (
            f"Test set contains records from {test_df['year_fix'].min()}, "
            f"expected >= {TEST_START_YEAR}"
        )

    def test_all_splits_cover_all_rows(self, train_df, val_df, test_df):
        """Sum of all split rows should match the feature-engineered total.
        (We can't easily check against df_feat here, but we can verify no
        split is suspiciously empty relative to others.)"""
        total = len(train_df) + len(val_df) + len(test_df)
        train_pct = len(train_df) / total * 100
        # Train should be at least 50% of data (historical ~70%)
        assert (
            train_pct >= 50.0
        ), f"Train split is suspiciously small: {train_pct:.1f}% of total"

    def test_no_split_is_empty(self, train_df, val_df, test_df):
        assert len(train_df) > 0, "Train split is empty!"
        assert len(val_df) > 0, "Val split is empty!"
        assert len(test_df) > 0, "Test split is empty!"


# ── Consistency Tests ──────────────────────────────────────────────────────────


class TestConsistency:
    """Cross-column consistency checks."""

    def test_weight_grams_consistent_with_pounds(self, train_df):
        """weight_grams should be approximately weight_pounds × 453.592."""
        LBS_TO_G = 453.592
        sample = (
            train_df[["weight_grams", "weight_pounds"]]
            .dropna()
            .sample(min(10_000, len(train_df)), random_state=42)
        )
        expected = sample["weight_pounds"] * LBS_TO_G
        diff = (sample["weight_grams"] - expected).abs()
        assert (
            diff.max() < 1.0
        ), f"weight_grams ≠ weight_pounds × 453.592 (max diff: {diff.max():.3f}g)"

    def test_parity_non_negative(self, train_df):
        if "parity" in train_df.columns:
            non_null = train_df["parity"].dropna()
            assert (non_null >= 0).all(), "Negative parity values found"

    def test_weight_gain_kg_non_negative(self, train_df):
        if "weight_gain_kg" in train_df.columns:
            non_null = train_df["weight_gain_kg"].dropna()
            # Some records may have 0 weight gain; negative is invalid
            assert (non_null >= 0).all(), "Negative weight_gain_kg found"

    def test_gestation_flags_consistency(self, train_df):
        """gestation_preterm and gestation_post_term must not both be 1."""
        if all(
            c in train_df.columns for c in ["gestation_preterm", "gestation_post_term"]
        ):
            both_set = (
                (train_df["gestation_preterm"] == 1)
                & (train_df["gestation_post_term"] == 1)
            ).sum()
            assert both_set == 0, (
                f"{both_set} records have both gestation_preterm=1"
                " and gestation_post_term=1"
            )
