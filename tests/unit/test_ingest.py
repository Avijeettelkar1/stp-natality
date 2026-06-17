"""Unit tests for src/data/ingest.py."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest import (
    FILE_COLUMNS,
    get_data_summary,
    load_all_files,
    load_single_file,
)

# ── CSV fixture helper ─────────────────────────────────────────────────────────

_CSV_HEADER = ",".join(FILE_COLUMNS)

# One representative data row — order matches FILE_COLUMNS exactly
_ROW_VALUES = [
    "1990",
    "6",
    "15",
    "7.5",
    "1",
    "true",
    "8",
    "9",
    "CA",
    "CA",
    "CA",
    "28",
    "true",
    "39",
    "01012000",
    "25.0",
    "",
    "",
    "",
    "",
    "1",
    "0",
    "0",
    "2",
    "30",
    "1",
]


def _write_csv(path: Path, n_rows: int = 3, year_start: int = 1990) -> None:
    """Write a minimal valid natality CSV with header to path."""
    rows = []
    for i in range(n_rows):
        row = _ROW_VALUES[:]
        row[0] = str(year_start + i)
        rows.append(",".join(row))
    path.write_text(_CSV_HEADER + "\n" + "\n".join(rows), encoding="utf-8")


# ── Tests: load_single_file ────────────────────────────────────────────────────


def test_load_single_file_returns_dataframe(tmp_path):
    f = tmp_path / "sample.csv"
    _write_csv(f, n_rows=3)
    result = load_single_file(f, era_year=1990)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_load_single_file_adds_source_year(tmp_path):
    f = tmp_path / "sample.csv"
    _write_csv(f)
    result = load_single_file(f, era_year=2002)
    assert "source_year" in result.columns
    assert (result["source_year"] == 2002).all()


def test_load_single_file_infers_era_from_filename(tmp_path):
    """era_year=None should be inferred from the filename."""
    f = tmp_path / "newborn_1994"
    _write_csv(f)
    result = load_single_file(f, era_year=None)
    assert (result["source_year"] == 1994).all()


def test_load_single_file_respects_nrows(tmp_path):
    f = tmp_path / "sample.csv"
    _write_csv(f, n_rows=10)
    result = load_single_file(f, era_year=2000, nrows=4)
    assert len(result) == 4


def test_load_single_file_drops_unknown_extra_columns(tmp_path):
    """Columns not in FILE_COLUMNS must not appear in the output."""
    f = tmp_path / "sample.csv"
    extra_header = _CSV_HEADER + ",mystery_col"
    row = ",".join(_ROW_VALUES) + ",extra_value"
    f.write_text(extra_header + "\n" + row, encoding="utf-8")
    result = load_single_file(f, era_year=2000)
    assert "mystery_col" not in result.columns


def test_load_single_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_single_file(tmp_path / "nonexistent.csv", era_year=2000)


def test_load_single_file_contains_expected_columns(tmp_path):
    f = tmp_path / "sample.csv"
    _write_csv(f)
    result = load_single_file(f, era_year=2000)
    assert "source_year" in result.columns
    assert "year_fix" in result.columns
    assert "weight_pounds" in result.columns


# ── Tests: load_all_files ──────────────────────────────────────────────────────


def test_load_all_files_concatenates_multiple_files(tmp_path):
    _write_csv(tmp_path / "newborn_2010", n_rows=3, year_start=2010)
    _write_csv(tmp_path / "newborn_2018", n_rows=2, year_start=2018)
    result = load_all_files(
        data_dir=tmp_path,
        file_filter=["newborn_2010", "newborn_2018"],
    )
    assert len(result) == 5


def test_load_all_files_adds_source_file_column(tmp_path):
    _write_csv(tmp_path / "newborn_2010", n_rows=2)
    result = load_all_files(data_dir=tmp_path, file_filter=["newborn_2010"])
    assert "source_file" in result.columns
    assert (result["source_file"] == "newborn_2010").all()


def test_load_all_files_respects_nrows_per_file(tmp_path):
    _write_csv(tmp_path / "newborn_2010", n_rows=10)
    _write_csv(tmp_path / "newborn_2018", n_rows=10)
    result = load_all_files(
        data_dir=tmp_path,
        file_filter=["newborn_2010", "newborn_2018"],
        nrows_per_file=3,
    )
    assert len(result) == 6


def test_load_all_files_skips_missing_files(tmp_path):
    """A missing file should be skipped (not raise); others still load."""
    _write_csv(tmp_path / "newborn_2018", n_rows=2)
    result = load_all_files(
        data_dir=tmp_path,
        file_filter=["newborn_2010", "newborn_2018"],  # 2010 missing
    )
    assert len(result) == 2


def test_load_all_files_raises_when_no_files_found(tmp_path):
    with pytest.raises(ValueError, match="No natality files"):
        load_all_files(data_dir=tmp_path, file_filter=["newborn_2010"])


# ── Tests: get_data_summary ────────────────────────────────────────────────────


def test_get_data_summary_expected_keys(tmp_path):
    f = tmp_path / "newborn_2018"
    _write_csv(f, n_rows=3)
    df = load_single_file(f, era_year=2018)
    summary = get_data_summary(df)
    expected = {
        "n_rows",
        "n_cols",
        "files_included",
        "year_range",
        "missing_pct_per_col",
        "memory_mb",
    }
    assert expected.issubset(summary.keys())


def test_get_data_summary_row_count(tmp_path):
    f = tmp_path / "newborn_2018"
    _write_csv(f, n_rows=5)
    df = load_single_file(f, era_year=2018)
    assert get_data_summary(df)["n_rows"] == 5


def test_get_data_summary_year_range(tmp_path):
    f = tmp_path / "newborn_2018"
    _write_csv(f, n_rows=3, year_start=2015)
    df = load_single_file(f, era_year=2018)
    yr = get_data_summary(df)["year_range"]
    assert yr["min"] == 2015
    assert yr["max"] == 2017
