"""Promotion gate: decide whether a newly trained model replaces the live one.

The challenger is the best candidate from the latest training run. The
champion is the model currently in production. Both are evaluated on the same
held-out test split, and the challenger is promoted only if every rule passes:

1. It beats the mean-prediction baseline on validation by a minimum margin.
2. Its test MAE is lower than the champion's by a minimum number of grams.
3. No sufficiently large subgroup gets noticeably worse than with the champion.

Without a champion (first deployment), only rule 1 applies.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.models.evaluate import evaluate_split, evaluate_subgroups
from src.models.pipeline import get_X_y


@dataclass(frozen=True)
class PromotionRules:
    """Thresholds for promoting a challenger model."""

    min_baseline_improvement_pct: float = 10.0
    min_mae_improvement_g: float = 1.0
    max_subgroup_regression_pct: float = 5.0
    min_subgroup_rows: int = 500


@dataclass
class PromotionDecision:
    """Outcome of the promotion gate, with the reason behind each rule."""

    promote: bool
    reasons: list[str] = field(default_factory=list)
    champion: dict | None = None
    challenger: dict | None = None
    rules: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def scorecard(pipeline, test_df: pd.DataFrame) -> dict:
    """Evaluate a fitted pipeline on the test split.

    Returns overall test MAE and R² plus MAE and size per subgroup, keyed as
    ``"grouping/group"`` (for example ``"gestation_band/preterm"``).
    """
    X, y = get_X_y(test_df)
    overall = evaluate_split(pipeline, X, y, split_name="test")
    subgroups = {
        f"{grouping}/{group}": {"n": values["n"], "mae": values["mae"]}
        for grouping, groups in evaluate_subgroups(pipeline, X, y).items()
        for group, values in groups.items()
    }
    return {
        "test_mae": overall["test_mae"],
        "test_r2": overall["test_r2"],
        "subgroups": subgroups,
    }


def decide(
    challenger: dict,
    champion: dict | None,
    baseline_improvement_pct: float,
    rules: PromotionRules = PromotionRules(),
) -> PromotionDecision:
    """Apply the promotion rules to two scorecards.

    Args:
        challenger: Scorecard of the newly trained model.
        champion: Scorecard of the live model, or None if there is none.
        baseline_improvement_pct: Challenger's validation improvement over the
            mean-prediction baseline, in percent.
        rules: Promotion thresholds.
    """
    reasons = []
    passed = True

    if baseline_improvement_pct >= rules.min_baseline_improvement_pct:
        reasons.append(
            f"Beats the mean baseline by {baseline_improvement_pct:.1f}% on "
            f"validation (required: {rules.min_baseline_improvement_pct:.0f}%)."
        )
    else:
        passed = False
        reasons.append(
            f"Beats the mean baseline by only {baseline_improvement_pct:.1f}% on "
            f"validation (required: {rules.min_baseline_improvement_pct:.0f}%)."
        )

    if champion is None:
        reasons.append("No live model to compare with; first deployment.")
    else:
        gain = champion["test_mae"] - challenger["test_mae"]
        if gain >= rules.min_mae_improvement_g:
            reasons.append(
                f"Test MAE {challenger['test_mae']:.1f} g vs. live "
                f"{champion['test_mae']:.1f} g: {gain:.1f} g better."
            )
        else:
            passed = False
            reasons.append(
                f"Test MAE {challenger['test_mae']:.1f} g vs. live "
                f"{champion['test_mae']:.1f} g: not at least "
                f"{rules.min_mae_improvement_g:.0f} g better."
            )

        for name, live in champion["subgroups"].items():
            new = challenger["subgroups"].get(name)
            if new is None or live["n"] < rules.min_subgroup_rows:
                continue
            change_pct = 100 * (new["mae"] - live["mae"]) / live["mae"]
            if change_pct > rules.max_subgroup_regression_pct:
                passed = False
                reasons.append(
                    f"Subgroup {name} gets worse: MAE {new['mae']:.1f} g vs. "
                    f"{live['mae']:.1f} g (+{change_pct:.1f}%, allowed "
                    f"{rules.max_subgroup_regression_pct:.0f}%)."
                )

    return PromotionDecision(
        promote=passed,
        reasons=reasons,
        champion=champion,
        challenger=challenger,
        rules=asdict(rules),
    )
