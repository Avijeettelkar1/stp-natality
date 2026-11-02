"""Send real births from the held-out test split to the live API.

Simulates usage for monitoring demos: each row becomes a prediction request,
and for a share of them the actual birth weight is reported through /feedback
(ground truth arriving later). With --drift, the sample is restricted to
preterm births or older mothers, so the inputs differ clearly from the
training data and the Evidently drift job has something to detect.

Usage:
    uv run python scripts/replay_traffic.py --n 150
    uv run python scripts/replay_traffic.py --n 120 --drift
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ["gestation_weeks", "mother_age", "year_fix", "month", "is_male"]
OPTIONAL_FLOAT = ["weight_gain_pounds", "father_age"]
COUNTS = ["born_alive_alive", "born_alive_dead", "born_dead"]

# Ranges accepted by the API (src/api/schemas.py); rows outside are skipped.
RANGES = {
    "gestation_weeks": (20, 44),
    "mother_age": (10, 60),
    "weight_gain_pounds": (0, 150),
    "father_age": (10, 100),
    "plurality": (1, 9),
    "year_fix": (1986, 2100),
    "month": (1, 12),
}


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def to_request(row: pd.Series) -> dict | None:
    """Map one processed row to a /predict payload, or None if unusable."""
    if any(_missing(row.get(name)) for name in REQUIRED):
        return None
    payload = {
        "gestation_weeks": float(row["gestation_weeks"]),
        "mother_age": float(row["mother_age"]),
        "year_fix": int(row["year_fix"]),
        "month": int(row["month"]),
        "is_male": bool(row["is_male"]),
        "plurality": 1 if _missing(row.get("plurality")) else int(row["plurality"]),
        "lmp_known": True if _missing(row.get("lmp_known")) else bool(row["lmp_known"]),
    }
    for name in OPTIONAL_FLOAT:
        value = row.get(name)
        payload[name] = None if _missing(value) else float(value)
    for name in COUNTS:
        value = row.get(name)
        payload[name] = 0 if _missing(value) else int(min(max(value, 0), 30))
    payload["ever_born"] = max(1, sum(payload[name] for name in COUNTS) + 1)
    married = row.get("mother_married")
    payload["mother_married"] = None if _missing(married) else bool(married)

    for name, (low, high) in RANGES.items():
        value = payload.get(name)
        if value is not None and not low <= value <= high:
            return None
    return payload


def send_feedback(client, base: str, prediction_id: str, actual: float) -> bool:
    """Report the actual weight; retry briefly because the API logs each
    prediction just after answering, so an immediate feedback can arrive
    before the prediction row exists (404)."""
    for _ in range(5):
        response = client.post(
            f"{base}/feedback",
            json={"prediction_id": prediction_id, "actual_weight_g": actual},
        )
        if response.status_code == 200:
            return True
        if response.status_code != 404:
            return False
        time.sleep(0.5)
    return False


def load_rows(path: Path, n: int, drift: bool, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if drift:
        df = df[(df["gestation_weeks"] < 37) | (df["mother_age"] >= 35)]
    return df.sample(n=min(n * 2, len(df)), random_state=seed)


def main(args: argparse.Namespace) -> int:
    rows = load_rows(args.data, args.n, args.drift, args.seed)
    rng = random.Random(args.seed)
    sent = feedback = errors = 0
    base = args.url.rstrip("/")

    with httpx.Client(timeout=90) as client:
        # Wake up a sleeping free-tier instance before sending traffic.
        client.get(f"{base}/health")
        for _, row in rows.iterrows():
            if sent >= args.n:
                break
            payload = to_request(row)
            if payload is None:
                continue
            response = client.post(f"{base}/predict", json=payload)
            if response.status_code != 200:
                errors += 1
                print(f"predict failed ({response.status_code}): {response.text[:120]}")
                continue
            sent += 1
            prediction_id = response.json()["prediction_id"]
            actual = row.get("weight_grams")
            if not _missing(actual) and rng.random() < args.feedback_share:
                feedback += send_feedback(client, base, prediction_id, float(actual))
            if sent % 25 == 0:
                print(f"{sent} predictions sent, {feedback} with feedback")
            time.sleep(args.delay)

    print(
        f"Done: {sent} predictions, {feedback} with feedback, {errors} errors "
        f"({'drifted' if args.drift else 'representative'} sample)."
    )
    return 0 if sent else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="https://stp-natality.onrender.com")
    parser.add_argument("--n", type=int, default=150, help="Number of predictions.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.parquet",
    )
    parser.add_argument(
        "--feedback-share",
        type=float,
        default=0.5,
        help="Share of predictions that get the actual weight as feedback.",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Send only preterm births or mothers aged 35+ (simulated drift).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Seconds between requests."
    )
    parser.add_argument("--seed", type=int, default=42)
    sys.exit(main(parser.parse_args()))
