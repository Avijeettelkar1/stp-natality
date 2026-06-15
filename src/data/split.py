"""Data splitting module for the STP Natality project.

Implements a temporal (year-based) train/validation/test split strategy,
which is the appropriate choice for multi-year time-series data. Using a
random split would allow data leakage from future years into training.

Split strategy (based on ``year_fix``):
    - Train set:  years < 2010  (~1986–2009, approximately 70% of data)
    - Val set:    2010 ≤ year < 2016  (approximately 15% of data)
    - Test set:   year ≥ 2016  (approximately 15% of data, most recent)

This mirrors a realistic deployment scenario: the model is trained on
historical data and evaluated on data from years it has never seen.

Usage:
    from src.data.split import temporal_split, save_splits

    train, val, test = temporal_split(df)
    save_splits(train, val, test, out_dir="data/processed/")
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Split Thresholds ───────────────────────────────────────────────────────────

# Year thresholds for temporal split
VAL_START_YEAR = 2010  # validation set starts from this year (inclusive)
TEST_START_YEAR = 2016  # test set starts from this year (inclusive)

# Output file names (Parquet format for efficiency with large data)
TRAIN_FILENAME = "train.parquet"
VAL_FILENAME = "val.parquet"
TEST_FILENAME = "test.parquet"


# ── Split Functions ────────────────────────────────────────────────────────────


def temporal_split(
    df: pd.DataFrame,
    val_start: int = VAL_START_YEAR,
    test_start: int = TEST_START_YEAR,
    year_col: str = "year_fix",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/validation/test by year.

    Args:
        df: Feature-engineered natality DataFrame with a year column.
        val_start: First year to include in validation set (default: 2010).
        test_start: First year to include in test set (default: 2016).
        year_col: Name of the year column to split on (default: ``year_fix``).

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        ValueError: If ``year_col`` is not present in the DataFrame.
    """
    if year_col not in df.columns:
        raise ValueError(
            f"Year column '{year_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    train = df[df[year_col] < val_start].copy()
    val = df[(df[year_col] >= val_start) & (df[year_col] < test_start)].copy()
    test = df[df[year_col] >= test_start].copy()

    total = len(df)
    _log_split_info(train, val, test, total)

    return train, val, test


def _log_split_info(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    total: int,
) -> None:
    """Log split size summary and year range information.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        total: Total number of rows before splitting.
    """
    splits = [("Train", train), ("Val", val), ("Test", test)]
    for name, split_df in splits:
        if len(split_df) == 0:
            logger.warning("%s split is empty!", name)
            continue
        year_min = (
            int(split_df["year_fix"].min()) if "year_fix" in split_df.columns else "?"
        )
        year_max = (
            int(split_df["year_fix"].max()) if "year_fix" in split_df.columns else "?"
        )
        pct = 100 * len(split_df) / max(total, 1)
        logger.info(
            "  %s: %d rows (%.1f%%) | years %s–%s",
            name,
            len(split_df),
            pct,
            year_min,
            year_max,
        )


# ── Save / Load Functions ──────────────────────────────────────────────────────


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str | Path = "data/processed",
) -> None:
    """Persist train/val/test splits to Parquet files.

    Parquet is used instead of CSV for three reasons:
    1. ~5–10× smaller file size due to columnar compression.
    2. Preserves column dtypes (including nullable Int8, Int16, etc.).
    3. Much faster I/O for large DataFrames.

    Args:
        train: Training DataFrame.
        val: Validation DataFrame.
        test: Test DataFrame.
        out_dir: Directory where Parquet files will be written.
            Created if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        (train, TRAIN_FILENAME),
        (val, VAL_FILENAME),
        (test, TEST_FILENAME),
    ]

    for split_df, filename in splits:
        out_path = out_dir / filename
        split_df.to_parquet(out_path, index=False, engine="pyarrow")
        size_mb = out_path.stat().st_size / 1_048_576
        logger.info("Saved %s → %s (%.1f MB)", filename, out_path, size_mb)

    logger.info("All splits saved to: %s", out_dir)


def load_splits(
    data_dir: str | Path = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train/val/test splits from Parquet files.

    Args:
        data_dir: Directory containing the Parquet split files.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        FileNotFoundError: If any of the expected Parquet files are missing.
    """
    data_dir = Path(data_dir)
    result = []

    for filename in [TRAIN_FILENAME, VAL_FILENAME, TEST_FILENAME]:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. "
                "Run the preprocessing and split pipeline first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %s: %d rows × %d cols", filename, len(df), len(df.columns))
        result.append(df)

    return tuple(result)  # type: ignore[return-value]


# ── Feature / Target Extraction ────────────────────────────────────────────────

# Columns that are targets or identifiers — not model input features
NON_FEATURE_COLUMNS = [
    "weight_pounds",  # original target (before conversion)
    "weight_grams",  # primary target — held out from X
    "outlier_weight",  # quality flag, not a feature
    "source_year",  # traceability metadata
    "source_file",  # traceability metadata
    "lmp",  # raw LMP string — lmp_known flag is the usable form
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of model input feature columns.

    Excludes target variables, metadata, and raw source columns.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        List of column names to use as model features (X).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def get_X_y(
    df: pd.DataFrame,
    target_col: str = "weight_grams",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target series y.

    Args:
        df: DataFrame after feature engineering (train, val, or test split).
        target_col: Name of the target column (default: ``weight_grams``).

    Returns:
        Tuple (X, y) where X is the feature DataFrame and y is the target Series.

    Raises:
        ValueError: If ``target_col`` is not in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    feature_cols = get_feature_columns(df)
    # Ensure target is not accidentally included in features
    feature_cols = [c for c in feature_cols if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    logger.info(
        "get_X_y: X shape=%s, y shape=%s (target='%s')",
        X.shape,
        y.shape,
        target_col,
    )
    return X, y
