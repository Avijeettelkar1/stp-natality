"""Feature engineering module for the STP Natality project.

Creates derived, domain-informed features from the cleaned natality DataFrame
(output of ``src.data.preprocess.run_preprocessing``).

New features produced:
- ``weight_gain_kg``          — maternal weight gain converted to kg
- ``parity``                  — total prior births
  (born_alive_alive + born_alive_dead + born_dead)
- ``is_multiple_birth``       — 1 if plurality > 1 (twins, triplets, etc.)
- ``lmp_known``               — 1 if last menstrual period date is not null
- ``birth_month_sin``         — sine encoding of month (cyclical)
- ``birth_month_cos``         — cosine encoding of month (cyclical)
- ``mother_age_group``        — ordinal age bucket: 0=teen, 1=20s, 2=30s, 3=40+
- ``gestation_preterm``       — 1 if gestation_weeks < 37 (preterm birth)
- ``gestation_post_term``     — 1 if gestation_weeks > 41 (post-term)

Usage:
    from src.data.preprocess import run_preprocessing
    from src.data.feature_engineering import engineer_features

    df_clean = run_preprocessing(df_raw)
    df_features = engineer_features(df_clean)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Clinical preterm / post-term thresholds
PRETERM_WEEKS = 37
POST_TERM_WEEKS = 41

# Maternal age group bins and labels
MOTHER_AGE_BINS = [0, 19, 29, 39, 120]
MOTHER_AGE_LABELS = [0, 1, 2, 3]  # 0=teen (≤19), 1=20s, 2=30s, 3=40+

# Pounds → kg
LBS_TO_KG = 0.453592


# ── Feature Functions ──────────────────────────────────────────────────────────


def add_weight_gain_kg(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``weight_gain_pounds`` to kilograms.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with ``weight_gain_kg`` column added.
    """
    if "weight_gain_pounds" not in df.columns:
        return df
    df = df.copy()
    df["weight_gain_kg"] = (df["weight_gain_pounds"] * LBS_TO_KG).round(2)
    logger.debug("Added 'weight_gain_kg' feature.")
    return df


def add_parity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total parity (number of prior pregnancies/births).

    parity = born_alive_alive + born_alive_dead + born_dead

    This is a known clinical predictor: subsequent children tend to be heavier.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with ``parity`` column added.
    """
    required = ["born_alive_alive", "born_alive_dead", "born_dead"]
    if not all(c in df.columns for c in required):
        logger.warning(
            "Cannot compute 'parity' — missing columns: %s",
            [c for c in required if c not in df.columns],
        )
        return df

    df = df.copy()
    df["parity"] = (
        df["born_alive_alive"].fillna(0)
        + df["born_alive_dead"].fillna(0)
        + df["born_dead"].fillna(0)
    ).astype(pd.Int16Dtype())
    logger.debug("Added 'parity' feature (median=%.0f).", df["parity"].median())
    return df


def add_multiple_birth_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag multiple births (twins, triplets, etc.).

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with ``is_multiple_birth`` (Int8) column added.
    """
    if "plurality" not in df.columns:
        return df
    df = df.copy()
    df["is_multiple_birth"] = (df["plurality"] > 1).astype(pd.Int8Dtype())
    n_multiple = df["is_multiple_birth"].sum()
    logger.debug("Added 'is_multiple_birth': %d multiple-birth records.", n_multiple)
    return df


def add_lmp_known_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Indicate whether the last menstrual period (LMP) date is known.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with ``lmp_known`` (Int8) flag column added.
    """
    if "lmp" not in df.columns:
        return df
    df = df.copy()
    df["lmp_known"] = df["lmp"].notna().astype(pd.Int8Dtype())
    pct_known = df["lmp_known"].mean() * 100
    logger.debug("Added 'lmp_known': %.1f%% of records have known LMP.", pct_known)
    return df


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sine/cosine cyclical encoding to the birth month.

    Months are cyclic (December → January), so standard ordinal encoding
    would imply a false distance between month 1 and month 12.

    Args:
        df: Cleaned DataFrame with a ``month`` column (1–12).

    Returns:
        DataFrame with ``birth_month_sin`` and ``birth_month_cos`` columns added.
    """
    if "month" not in df.columns:
        return df
    df = df.copy()
    df["birth_month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(6)
    df["birth_month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(6)
    logger.debug("Added cyclical month encoding: 'birth_month_sin', 'birth_month_cos'.")
    return df


def add_mother_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin maternal age into ordinal clinical categories.

    Groups:
    - 0: Teen (≤ 19)
    - 1: Twenties (20–29)
    - 2: Thirties (30–39)
    - 3: Forties+ (≥ 40)

    Args:
        df: Cleaned DataFrame with a ``mother_age`` column.

    Returns:
        DataFrame with ``mother_age_group`` (Int8) column added.
    """
    if "mother_age" not in df.columns:
        return df
    df = df.copy()
    df["mother_age_group"] = pd.cut(
        df["mother_age"],
        bins=MOTHER_AGE_BINS,
        labels=MOTHER_AGE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(pd.Int8Dtype())
    logger.debug("Added 'mother_age_group' feature (ordinal 0–3).")
    return df


def add_gestation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag preterm and post-term pregnancies.

    - ``gestation_preterm`` = 1 if gestation_weeks < 37
    - ``gestation_post_term`` = 1 if gestation_weeks > 41

    These are clinically significant thresholds directly associated with
    birth weight abnormalities.

    Args:
        df: Cleaned DataFrame with ``gestation_weeks`` column.

    Returns:
        DataFrame with ``gestation_preterm`` and ``gestation_post_term``
        flag columns added.
    """
    if "gestation_weeks" not in df.columns:
        return df
    df = df.copy()
    df["gestation_preterm"] = (df["gestation_weeks"] < PRETERM_WEEKS).astype(
        pd.Int8Dtype()
    )
    df["gestation_post_term"] = (df["gestation_weeks"] > POST_TERM_WEEKS).astype(
        pd.Int8Dtype()
    )
    n_preterm = df["gestation_preterm"].sum()
    n_post = df["gestation_post_term"].sum()
    logger.debug(
        "Added gestation flags: %d preterm (<37wk), %d post-term (>41wk).",
        n_preterm,
        n_post,
    )
    return df


# ── Full Feature Engineering Pipeline ─────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature engineering pipeline.

    Applies all derived feature transformations in order:
    1. ``add_weight_gain_kg``       — kg conversion
    2. ``add_parity``               — prior pregnancy count
    3. ``add_multiple_birth_flag``  — twins/triplets flag
    4. ``add_lmp_known_flag``       — LMP data quality flag
    5. ``add_cyclical_month``       — sin/cos month encoding
    6. ``add_mother_age_group``     — ordinal age bins
    7. ``add_gestation_flags``      — preterm / post-term flags

    Args:
        df: Cleaned DataFrame from ``src.data.preprocess.run_preprocessing``.

    Returns:
        DataFrame enriched with all engineered features.
    """
    logger.info(
        "=== Starting feature engineering (input: %d rows, %d cols) ===",
        len(df),
        len(df.columns),
    )

    df = add_weight_gain_kg(df)
    df = add_parity(df)
    df = add_multiple_birth_flag(df)
    df = add_lmp_known_flag(df)
    df = add_cyclical_month(df)
    df = add_mother_age_group(df)
    df = add_gestation_flags(df)

    new_features = [
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
    existing_new = [f for f in new_features if f in df.columns]
    logger.info(
        "=== Feature engineering complete (output: %d rows, %d cols) — added: %s ===",
        len(df),
        len(df.columns),
        existing_new,
    )
    return df
