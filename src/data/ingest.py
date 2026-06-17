"""Data ingestion module for the STP Natality project.

Handles loading and combining the 5 raw US Natality CSV files
(1986, 1994, 2002, 2010, 2018 era files).

Usage:
    from src.data.ingest import load_all_files

    df = load_all_files("newborn_data/")
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Default raw data directory (relative to project root)
DEFAULT_RAW_DATA_DIR = "newborn_data"

# All 5 raw natality files and their era labels
NATALITY_FILES = {
    "newborn_1986": 1986,
    "newborn_1994": 1994,
    "newborn_2002": 2002,
    "newborn_2010": 2010,
    "newborn_2018": 2018,
}

# The 26 columns in the shared schema (in order)
COLUMN_NAMES = [
    "source_year",  # year group label (added by us; not in raw file)
    "year_fix",  # actual 4-digit year of birth
    "month",  # month of birth (1–12)
    "day",  # day of birth (1–31)
    "weight_pounds",  # TARGET: birth weight in pounds
    "plurality",  # number of children born (1=singleton, 2=twins …)
    "is_male",  # bool: TRUE if child is male
    "apgar_1min",  # LEAKAGE — Apgar score at 1 minute (post-birth)
    "apgar_5min",  # LEAKAGE — Apgar score at 5 minutes (post-birth)
    "state",  # state of birth (postal code)
    "mother_residence_state",  # mother's state of residence
    "mother_birth_state",  # mother's state of birth
    "mother_age",  # mother's age at birth
    "mother_married",  # bool: TRUE if married
    "gestation_weeks",  # pregnancy duration in weeks
    "lmp",  # last menstrual period (MMDDYYYY; 99/9999=unknown)
    "weight_gain_pounds",  # maternal weight gain during pregnancy
    "cigarette_use",  # bool: smoked during pregnancy (2020+ only)
    "cigarettes_per_day",  # cigarettes/day (2020+ only)
    "alcohol_use",  # bool: alcohol use (2006+ only)
    "drinks_per_week",  # drinks/week (2006+ only)
    "born_alive_alive",  # prior children still living
    "born_alive_dead",  # prior children who have died
    "born_dead",  # prior stillbirths
    "ever_born",  # total children ever born (including current)
    "father_age",  # father's age at birth
    "record_weight",  # 1=full-reporting; 2=50% sample
]

# Actual raw columns (the file doesn't include 'source_year', we add it)
RAW_COLUMN_NAMES = COLUMN_NAMES[1:]  # everything except our added 'source_year'

# Columns in the files that we want to keep (same as RAW_COLUMN_NAMES, used
# for filtering after reading with header=0 which gives us the actual column
# names from the file).
# Note: actual file header has 26 columns in a *different* order than COLUMN_NAMES.
# We read with header=0 to get the file's own column names, then select/reorder.
FILE_COLUMNS = [
    "year_fix",
    "month",
    "day",
    "weight_pounds",
    "plurality",
    "is_male",
    "apgar_1min",
    "apgar_5min",
    "state",
    "mother_residence_state",
    "mother_birth_state",
    "mother_age",
    "mother_married",
    "gestation_weeks",
    "lmp",
    "weight_gain_pounds",
    "cigarette_use",
    "cigarettes_per_day",
    "alcohol_use",
    "drinks_per_week",
    "born_alive_alive",
    "born_alive_dead",
    "born_dead",
    "ever_born",
    "father_age",
    "record_weight",
]


# ── Core Functions ─────────────────────────────────────────────────────────────


def load_single_file(
    filepath: str | Path,
    era_year: Optional[int] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a single raw natality CSV file into a Pandas DataFrame.

    The raw files are CSVs **with a header row** whose column names match the
    natality schema but may be in a different order than ``COLUMN_NAMES``.
    This function reads the file with ``header=0``, selects the 26 known columns,
    and adds a ``source_year`` column for traceability.

    Args:
        filepath: Absolute or relative path to the raw natality file.
        era_year: Integer year label for this file (e.g. 1986). If None,
            inferred from the filename suffix.
        nrows: If set, only load the first ``nrows`` rows (useful for quick
            development iteration without loading 3–6M rows).

    Returns:
        DataFrame with 26 schema columns plus ``source_year``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    # Infer era year from filename if not provided
    if era_year is None:
        for name, year in NATALITY_FILES.items():
            if name in filepath.name:
                era_year = year
                break

    logger.info("Loading file: %s (era=%s, nrows=%s)", filepath.name, era_year, nrows)

    # The files have a header row — read it as-is, then select known columns.
    df = pd.read_csv(
        filepath,
        header=0,
        nrows=nrows,
        low_memory=False,
        sep=",",
        on_bad_lines="warn",
    )

    # Select only the 26 known schema columns (guards against unexpected extra cols)
    available = [c for c in FILE_COLUMNS if c in df.columns]
    missing = [c for c in FILE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Columns not found in %s: %s", filepath.name, missing)
    df = df[available].copy()

    # Add traceability column
    df.insert(0, "source_year", era_year)

    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    logger.info(
        "  → Loaded %d rows × %d cols (%.1f MB)", len(df), len(df.columns), mem_mb
    )

    return df


def load_all_files(
    data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    file_filter: Optional[list[str]] = None,
    nrows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    """Load and concatenate all 5 raw natality files into a single DataFrame.

    Args:
        data_dir: Directory containing the raw natality files. Defaults to
            ``newborn_data/`` relative to the current working directory.
        file_filter: Optional list of filenames to load (e.g.
            ``["newborn_2010", "newborn_2018"]``). If None, all 5 files are
            loaded.
        nrows_per_file: If set, loads only the first N rows from each file.
            Useful for fast iteration during development.

    Returns:
        Concatenated DataFrame from all specified files. A ``source_file``
        string column is added for traceability.

    Raises:
        ValueError: If no files are found in ``data_dir``.
    """
    data_dir = Path(data_dir)
    frames = []
    files_to_load = file_filter or list(NATALITY_FILES.keys())

    logger.info("Loading %d natality file(s) from: %s", len(files_to_load), data_dir)

    for filename in files_to_load:
        filepath = data_dir / filename
        era_year = NATALITY_FILES.get(filename)

        try:
            df_single = load_single_file(
                filepath, era_year=era_year, nrows=nrows_per_file
            )
            df_single["source_file"] = filename
            frames.append(df_single)
        except FileNotFoundError as exc:
            logger.warning("Skipping missing file: %s — %s", filepath, exc)

    if not frames:
        raise ValueError(
            f"No natality files could be loaded from '{data_dir}'. "
            f"Expected files: {files_to_load}"
        )

    logger.info("Concatenating %d file(s)...", len(frames))
    combined = pd.concat(frames, ignore_index=True)

    total_rows = len(combined)
    total_mb = combined.memory_usage(deep=True).sum() / 1_048_576
    logger.info(
        "Combined dataset: %d rows × %d cols (%.1f MB)",
        total_rows,
        len(combined.columns),
        total_mb,
    )

    return combined


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict with basic dataset statistics.

    Args:
        df: DataFrame returned by ``load_all_files`` or ``load_single_file``.

    Returns:
        Dictionary with keys: n_rows, n_cols, files_included, year_range,
        missing_pct_per_col, memory_mb.
    """
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "files_included": (
            df["source_file"].unique().tolist() if "source_file" in df.columns else []
        ),
        "year_range": {
            "min": int(df["year_fix"].min()) if "year_fix" in df.columns else None,
            "max": int(df["year_fix"].max()) if "year_fix" in df.columns else None,
        },
        "missing_pct_per_col": (df.isnull().mean() * 100).round(2).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 1),
    }
