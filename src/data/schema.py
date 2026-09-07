"""Canonical data contract for the provided natality files."""

RAW_COLUMNS = [
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

TRACE_COLUMNS = ["source_year", "source_file"]
TARGET_COLUMN = "weight_grams"
LEAKAGE_COLUMNS = ["apgar_1min", "apgar_5min"]


def validate_raw_columns(columns: list[str]) -> None:
    """Raise when a raw file does not implement the agreed 26-column schema."""
    missing = sorted(set(RAW_COLUMNS) - set(columns))
    if missing:
        raise ValueError(
            "Natality file does not match natality_schema.txt. "
            f"Missing columns: {missing}"
        )
