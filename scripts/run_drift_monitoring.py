"""Evidently drift-monitoring job: reference vs recent production inputs.

Reads the model's reference/training data and the recent production prediction
inputs from Postgres (``monitoring.predictions.inputs``), runs Evidently, and
writes an HTML report plus a small JSON summary to ``reports/drift/``.

This is a standalone monitoring job (run on a schedule, e.g. GitHub Actions);
it does not run inside the prediction request. The database connection comes
from the ``DATABASE_URL`` environment variable — nothing is hardcoded.

Usage:
    DATABASE_URL=... uv run python scripts/run_drift_monitoring.py
    uv run python scripts/run_drift_monitoring.py --current-parquet FILE.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.evidently_drift import (  # noqa: E402
    DEFAULT_MIN_ROWS,
    MONITORED_FEATURES,
    drift_row_from_summary,
    extract_inputs,
    load_reference,
    run_drift,
    write_outputs,
)

DRIFT_COLUMNS = [
    "generated_at",
    "status",
    "reference_rows",
    "current_rows",
    "dataset_drift",
    "drifted_features",
    "total_features",
    "drift_share",
    "features",
]


def fetch_production_inputs(database_url: str, days: int, limit: int | None) -> list:
    """Return recent ``inputs`` JSON payloads from monitoring.predictions."""
    import psycopg

    query = (
        "SELECT inputs FROM monitoring.predictions "
        "WHERE created_at >= now() - make_interval(days => %(days)s) "
        "ORDER BY created_at DESC"
    )
    params: dict = {"days": days}
    if limit:
        query += " LIMIT %(limit)s"
        params["limit"] = limit

    with psycopg.connect(database_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]


def save_drift_run(database_url: str, row: dict) -> None:
    """Insert one drift-run summary into monitoring.drift (best-effort)."""
    import psycopg
    from psycopg.types.json import Jsonb

    values = [row.get(col) for col in DRIFT_COLUMNS]
    values[-1] = Jsonb(row["features"]) if row.get("features") is not None else None
    placeholders = ", ".join(["%s"] * len(DRIFT_COLUMNS))
    query = (
        f"INSERT INTO monitoring.drift ({', '.join(DRIFT_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    with psycopg.connect(database_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)


def main(args: argparse.Namespace) -> int:
    database_url = args.database_url or os.getenv("DATABASE_URL")
    reference = load_reference(args.reference)

    if args.current_parquet:
        current = pd.read_parquet(args.current_parquet, columns=MONITORED_FEATURES)
        current = extract_inputs(current.to_dict("records"))
    else:
        if not database_url:
            print("DATABASE_URL is not set; pass --current-parquet for an offline run.")
            return 1
        rows = fetch_production_inputs(database_url, args.days, args.limit)
        current = extract_inputs(rows)

    result, summary = run_drift(reference, current, min_rows=args.min_rows)
    write_outputs(result, summary, args.html, args.json)

    # Persist the summary so Grafana can trend drift over time (best-effort).
    if database_url:
        try:
            save_drift_run(database_url, drift_row_from_summary(summary))
            print("Saved drift summary to monitoring.drift.")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not write monitoring.drift row: {exc}")

    print(json.dumps(summary, indent=2))
    if summary.get("status") == "insufficient_data":
        print(
            f"\nInsufficient production data: {summary['current_rows']} rows "
            f"(need {summary['required_rows']}). No drift report produced; job OK."
        )
    else:
        print(f"\nHTML report: {args.html}")
        state = "DETECTED" if summary["dataset_drift"] else "not detected"
        print(
            f"Dataset drift {state}: {summary['drifted_features']}/"
            f"{summary['total_features']} features ({summary['drift_share']:.0%})."
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train.parquet",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back this many days of production predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of recent predictions.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Below this many production rows, report insufficient_data.",
    )
    parser.add_argument(
        "--current-parquet",
        type=Path,
        default=None,
        help="Offline mode: use a parquet instead of querying the database.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drift" / "latest.html",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drift" / "latest.json",
    )
    sys.exit(main(parser.parse_args()))
