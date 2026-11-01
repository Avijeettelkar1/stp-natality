"""Generate an Evidently data-drift report for the model features.

Compares a reference split (training data by default) with a current split
(the test split by default, or any processed parquet) and writes an
interactive HTML report plus a JSON summary to ``reports/``.

Usage:
    uv run python scripts/drift_evidently.py
    uv run python scripts/drift_evidently.py --current data/processed/val.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import MODEL_FEATURES  # noqa: E402
from src.monitoring.evidently_drift import build_report, write_outputs  # noqa: E402


def main(reference_path: Path, current_path: Path, html_path: Path, json_path: Path):
    reference = pd.read_parquet(reference_path, columns=MODEL_FEATURES)
    current = pd.read_parquet(current_path, columns=MODEL_FEATURES)
    report = build_report(reference, current, MODEL_FEATURES)
    summary = write_outputs(report, html_path, json_path)
    print(json.dumps(summary, indent=2))
    print(f"\nHTML report: {html_path}")
    if summary["dataset_drift"]:
        print(
            f"Dataset drift detected: {summary['n_drifted']}/{summary['n_features']} "
            f"features drifted ({summary['share_drifted']:.0%})."
        )
    else:
        print("No dataset-level drift detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train.parquet",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.parquet",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drift_evidently.html",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drift_evidently.json",
    )
    args = parser.parse_args()
    main(args.reference, args.current, args.html, args.json)
