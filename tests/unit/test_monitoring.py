import numpy as np
import pandas as pd

from src.monitoring.drift import compare_batches


def test_drift_report_detects_large_distribution_shift():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"feature": rng.normal(0, 1, 1000)})
    current = pd.DataFrame({"feature": rng.normal(4, 1, 1000)})
    report = compare_batches(reference, current, ["feature"])
    assert report["drift_detected"] is True
    assert report["drifted_features"] == ["feature"]


def test_drift_report_handles_empty_feature():
    reference = pd.DataFrame({"feature": [None, None]})
    current = pd.DataFrame({"feature": [1.0, 2.0]})
    report = compare_batches(reference, current, ["feature"])
    assert report["features"]["feature"]["status"] == "insufficient_data"
