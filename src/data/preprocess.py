"""Data preprocessing module for the STP Natality project.

Applies a sequential, reproducible cleaning and transformation pipeline
to the raw natality DataFrame produced by ``src.data.ingest``.

Key responsibilities:
- Remove data-leakage columns (Apgar scores)
- Cast boolean columns from string representations to integers
- Encode unknown/sentinel values as NaN
- Clip outliers to clinically valid ranges
- Convert birth weight from pounds to grams (adds ``weight_grams`` column)
- Drop rows with missing target variable

Usage:
    from src.data.ingest import load_all_files
    from src.data.preprocess import run_preprocessing

    df_raw = load_all_files("newborn_data/", nrows_per_file=10_000)
    df_clean = run_preprocessing(df_raw)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Columns to unconditionally remove (data leakage or no clinical value)
LEAKAGE_COLUMNS = ["apgar_1min", "apgar_5min", "day"]

# Boolean columns stored as "true"/"false" strings (or True/False objects)
BOOL_COLUMNS = ["is_male", "mother_married", "cigarette_use", "alcohol_use"]

# Clinical/data constraints
GESTATION_WEEKS_MIN = 20
GESTATION_WEEKS_MAX = 44  # Values > 44 are almost certainly data errors
MOTHER_AGE_MIN = 10
MOTHER_AGE_MAX = 60
WEIGHT_POUNDS_MIN = 0.66  # ~300g — clinical minimum viable birth weight
WEIGHT_POUNDS_MAX = 15.43  # ~7,000g — clinical maximum (macrosomia upper bound)

# Pound → gram conversion
LBS_TO_GRAMS = 453.592


# ── Individual Transformation Steps ───────────────────────────────────────────


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove post-birth measurement columns and clinically irrelevant columns.

    Removes: ``apgar_1min``, ``apgar_5min``, ``day``.

    Args:
        df: Raw natality DataFrame.

    Returns:
        DataFrame with leakage columns removed.
    """
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    logger.info("Dropping leakage/irrelevant columns: %s", cols_to_drop)
    return df.drop(columns=cols_to_drop)


def cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean string columns (\"true\"/\"false\") to integer (1/0).

    Handles: ``is_male``, ``mother_married``, ``cigarette_use``,
    ``alcohol_use``.

    Args:
        df: DataFrame with boolean columns as strings or Python bools.

    Returns:
        DataFrame with boolean columns cast to nullable integer dtype (pd.Int8Dtype).
    """
    df = df.copy()
    for col in BOOL_COLUMNS:
        if col not in df.columns:
            continue

        col_dtype = df[col].dtype
        # pandas 2.x/3.x may infer StringDtype or object for string columns;
        # also handle Python bool columns.
        is_string_like = (
            col_dtype == object
            or isinstance(col_dtype, pd.StringDtype)
            or str(col_dtype).startswith("string")
        )
        if is_string_like:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {"true": 1, "false": 0, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA}
                )
                .astype(pd.Int8Dtype())
            )
        elif col_dtype == bool or col_dtype == "bool":
            df[col] = df[col].astype(pd.Int8Dtype())
        # If already numeric, leave as-is

    logger.info(
        "Boolean columns cast to Int8: %s", [c for c in BOOL_COLUMNS if c in df.columns]
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel/unknown values with NaN.

    Encodings handled:
    - ``father_age == 99``  → NaN  (99 = unknown)
    - ``lmp`` containing \"99\" or \"9999\"  → NaN
    - ``gestation_weeks == 99``  → NaN
    - ``plurality == 0`` → NaN  (invalid)

    Args:
        df: DataFrame after boolean casting.

    Returns:
        DataFrame with sentinels replaced by NaN.
    """
    df = df.copy()

    if "father_age" in df.columns:
        n = (df["father_age"] == 99).sum()
        df.loc[df["father_age"] == 99, "father_age"] = np.nan
        logger.info("father_age: replaced %d sentinel values (99) with NaN", n)

    if "gestation_weeks" in df.columns:
        n = (df["gestation_weeks"] == 99).sum()
        df.loc[df["gestation_weeks"] == 99, "gestation_weeks"] = np.nan
        logger.info("gestation_weeks: replaced %d sentinel values (99) with NaN", n)

    if "lmp" in df.columns:
        # Mark unknown LMP as NaN (sentinel: "99" or "9999" or similar patterns)
        lmp_str = df["lmp"].astype(str).str.strip()
        unknown_mask = lmp_str.isin(["99", "9999", "nan", ""]) | lmp_str.str.match(
            r"^9+$"
        )
        n = unknown_mask.sum()
        df.loc[unknown_mask, "lmp"] = np.nan
        logger.info("lmp: replaced %d unknown LMP values with NaN", n)

    if "plurality" in df.columns:
        n = (df["plurality"] == 0).sum()
        df.loc[df["plurality"] == 0, "plurality"] = np.nan
        logger.info("plurality: replaced %d zero values with NaN", n)

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap/clip feature values to clinically valid ranges.

    Ranges applied:
    - ``gestation_weeks``: clipped to [20, 44]
    - ``mother_age``: clipped to [10, 60]
    - ``weight_pounds`` (target): rows outside [0.66, 15.43] are flagged
      with an ``outlier_weight`` boolean column (not removed here).

    Args:
        df: DataFrame after missing value handling.

    Returns:
        DataFrame with out-of-range values capped and an ``outlier_weight``
        flag column added.
    """
    df = df.copy()

    if "gestation_weeks" in df.columns:
        out_of_range = (
            (df["gestation_weeks"] < GESTATION_WEEKS_MIN)
            | (df["gestation_weeks"] > GESTATION_WEEKS_MAX)
        ) & df["gestation_weeks"].notna()
        n = out_of_range.sum()
        df["gestation_weeks"] = df["gestation_weeks"].clip(
            lower=GESTATION_WEEKS_MIN, upper=GESTATION_WEEKS_MAX
        )
        logger.info(
            "gestation_weeks: capped %d out-of-range values to [%d, %d]",
            n,
            GESTATION_WEEKS_MIN,
            GESTATION_WEEKS_MAX,
        )

    if "mother_age" in df.columns:
        out_of_range = (
            (df["mother_age"] < MOTHER_AGE_MIN) | (df["mother_age"] > MOTHER_AGE_MAX)
        ) & df["mother_age"].notna()
        n = out_of_range.sum()
        df["mother_age"] = df["mother_age"].clip(
            lower=MOTHER_AGE_MIN, upper=MOTHER_AGE_MAX
        )
        logger.info(
            "mother_age: capped %d out-of-range values to [%d, %d]",
            n,
            MOTHER_AGE_MIN,
            MOTHER_AGE_MAX,
        )

    if "weight_pounds" in df.columns:
        # Flag — do NOT remove (that happens in drop_null_targets)
        df["outlier_weight"] = (
            (df["weight_pounds"] < WEIGHT_POUNDS_MIN)
            | (df["weight_pounds"] > WEIGHT_POUNDS_MAX)
        ).astype(pd.Int8Dtype())
        n = df["outlier_weight"].sum()
        logger.info(
            "weight_pounds: flagged %d clinical outliers in 'outlier_weight' column", n
        )

    return df


def convert_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``weight_grams`` column (birth weight in grams).

    Converts ``weight_pounds`` to grams using: weight_grams = weight_pounds × 453.592.
    This is the primary target variable for modeling (grams are the clinical standard).

    Args:
        df: DataFrame after outlier capping.

    Returns:
        DataFrame with ``weight_grams`` column added.
    """
    if "weight_pounds" not in df.columns:
        logger.warning("'weight_pounds' column not found — skipping target conversion.")
        return df

    df = df.copy()
    df["weight_grams"] = (df["weight_pounds"] * LBS_TO_GRAMS).round(1)
    logger.info(
        "Added 'weight_grams' column: mean=%.1fg, std=%.1fg",
        df["weight_grams"].mean(),
        df["weight_grams"].std(),
    )
    return df


def drop_null_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the birth weight target is null or clinically invalid.

    Drops rows where ``weight_pounds`` is NaN OR outside the clinical valid
    range [0.66 lbs, 15.43 lbs] (equivalent to [300g, 7000g]).

    Args:
        df: DataFrame after target conversion.

    Returns:
        DataFrame with invalid target rows removed.

    Note:
        This must be applied AFTER ``cap_outliers`` so the ``outlier_weight``
        flag is available for logging.
    """
    df = df.copy()
    n_before = len(df)

    # Drop null targets
    null_mask = df["weight_pounds"].isna()
    n_null = null_mask.sum()

    # Drop extreme clinical outliers (likely data errors)
    extreme_mask = (df["weight_pounds"] < WEIGHT_POUNDS_MIN) | (
        df["weight_pounds"] > WEIGHT_POUNDS_MAX
    )
    n_extreme = extreme_mask.sum()

    df = df[~(null_mask | extreme_mask)].copy()
    n_after = len(df)

    logger.info(
        "drop_null_targets: removed %d null + %d extreme outlier rows "
        "(%d → %d rows, %.2f%% removed)",
        n_null,
        n_extreme,
        n_before,
        n_after,
        100 * (n_before - n_after) / max(n_before, 1),
    )
    return df


# ── Full Pipeline ──────────────────────────────────────────────────────────────


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full preprocessing pipeline in the correct order.

    Pipeline steps (applied sequentially):
    1. ``drop_leakage_columns`` — remove Apgar scores and ``day``
    2. ``cast_booleans`` — convert string booleans to Int8
    3. ``handle_missing_values`` — replace sentinel values with NaN
    4. ``cap_outliers`` — clip features to valid ranges; flag weight outliers
    5. ``convert_target`` — add ``weight_grams`` column
    6. ``drop_null_targets`` — remove rows with invalid birth weight

    Args:
        df: Raw natality DataFrame from ``src.data.ingest.load_all_files``.

    Returns:
        Cleaned DataFrame ready for feature engineering and modeling.
    """
    logger.info("=== Starting preprocessing pipeline (input: %d rows) ===", len(df))

    df = drop_leakage_columns(df)
    df = cast_booleans(df)
    df = handle_missing_values(df)
    df = cap_outliers(df)
    df = convert_target(df)
    df = drop_null_targets(df)

    logger.info("=== Preprocessing complete (output: %d rows) ===", len(df))
    return df
