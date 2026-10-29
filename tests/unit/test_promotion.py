"""Unit tests for src/models/promotion.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.pipeline import MODEL_FEATURES, build_pipeline
from src.models.promotion import PromotionRules, decide, scorecard


def _card(test_mae, subgroups=None):
    return {
        "test_mae": test_mae,
        "test_r2": 0.4,
        "subgroups": subgroups or {"sex/male": {"n": 5000, "mae": test_mae}},
    }


def test_better_challenger_is_promoted():
    decision = decide(_card(350.0), _card(362.8), baseline_improvement_pct=16.0)
    assert decision.promote
    assert any("12.8 g better" in reason for reason in decision.reasons)


def test_worse_challenger_is_rejected():
    decision = decide(_card(371.0), _card(362.8), baseline_improvement_pct=15.0)
    assert not decision.promote


def test_equal_challenger_is_rejected():
    decision = decide(_card(362.8), _card(362.8), baseline_improvement_pct=16.0)
    assert not decision.promote


def test_subgroup_regression_blocks_promotion():
    champion = _card(362.8, {"gestation_band/preterm": {"n": 14000, "mae": 451.0}})
    challenger = _card(355.0, {"gestation_band/preterm": {"n": 14000, "mae": 490.0}})
    decision = decide(challenger, champion, baseline_improvement_pct=16.0)
    assert not decision.promote
    assert any("preterm" in reason for reason in decision.reasons)


def test_small_subgroups_are_ignored():
    champion = _card(362.8, {"gestation_band/tiny": {"n": 50, "mae": 300.0}})
    challenger = _card(355.0, {"gestation_band/tiny": {"n": 50, "mae": 600.0}})
    decision = decide(challenger, champion, baseline_improvement_pct=16.0)
    assert decision.promote


def test_first_deployment_needs_only_the_baseline_rule():
    assert decide(_card(362.8), None, baseline_improvement_pct=16.0).promote
    assert not decide(_card(362.8), None, baseline_improvement_pct=4.0).promote


def test_weak_baseline_improvement_blocks_promotion():
    decision = decide(_card(350.0), _card(362.8), baseline_improvement_pct=5.0)
    assert not decision.promote


def test_rules_can_be_tightened():
    rules = PromotionRules(min_mae_improvement_g=20.0)
    decision = decide(_card(350.0), _card(362.8), 16.0, rules=rules)
    assert not decision.promote


def test_scorecard_reports_overall_and_subgroups():
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    df["gestation_weeks"] = rng.integers(30, 43, n)
    df["is_male"] = rng.integers(0, 2, n).astype(float)
    df["year_fix"] = 2017
    df["weight_grams"] = 3300 + 150 * (df["gestation_weeks"] - 39)
    pipeline = build_pipeline(Ridge())
    pipeline.fit(df[MODEL_FEATURES], df["weight_grams"])

    card = scorecard(pipeline, df)

    assert card["test_mae"] == pytest.approx(0.0, abs=1.0)
    assert "sex/male" in card["subgroups"]
    assert card["subgroups"]["sex/male"]["n"] > 0
