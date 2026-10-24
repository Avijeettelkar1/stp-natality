"""Compare training and recent temporal splits for input drift."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import MODEL_FEATURES  # noqa: E402
from src.monitoring.drift import compare_batches, write_report  # noqa: E402


def main(reference_path: Path, current_path: Path, output_path: Path) -> None:
    reference = pd.read_parquet(reference_path, columns=MODEL_FEATURES)
    current = pd.read_parquet(current_path, columns=MODEL_FEATURES)
    report = compare_batches(reference, current, MODEL_FEATURES)
    write_report(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drift_report.json",
    )
    args = parser.parse_args()
    main(args.reference, args.current, args.output)
