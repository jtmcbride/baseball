"""Model #6's counterfactual and feature contracts."""

from __future__ import annotations

import numpy as np
import polars as pl

from bbml.features.run_value import RunValue
from bbml.features.swing_decision import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET_SWING
from bbml.models.swing_decision import decision_value_by_batter, score_decisions


class _SwingValues:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict_swing_value(self, _df):
        return self.values


class _StrikeProbabilities:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, _df):
        return self.values


def _rv() -> RunValue:
    # A tiny, deliberately monotone count table; enough to make taking a
    # called strike worse than taking a ball at 0-0.
    return RunValue(
        count_re={"0-0": 0.0, "1-0": 0.04, "0-1": -0.03},
        event_value={"walk": 0.30, "strikeout": -0.20},
    )


def test_feature_contract_has_no_batter_identity_and_one_categorical():
    assert "batter" not in FEATURE_NAMES
    assert CATEGORICAL_FEATURES == ["pitch_type"]


def test_score_uses_better_counterfactual_and_never_awards_regret():
    df = pl.DataFrame(
        {
            "batter": [1, 1],
            "season": [2026, 2026],
            "balls": [0, 0],
            "strikes": [0, 0],
            TARGET_SWING: [True, False],
            "pitch_type": ["FF", "FF"],
        }
    )
    # first pitch: swinging is worth 0.10, so the swing is optimal. Second:
    # swinging is -0.10 while a likely ball makes taking better.
    scored = score_decisions(
        df, _SwingValues([0.10, -0.10]), _StrikeProbabilities([0.5, 0.1]), _rv()
    )
    assert scored["decision_value"][0] == 0.0
    assert scored["decision_value"][1] == 0.0


def test_wrong_choice_has_negative_value_and_aggregates_per_100():
    df = pl.DataFrame(
        {
            "batter": [1, 1],
            "season": [2026, 2026],
            "balls": [0, 0],
            "strikes": [0, 0],
            TARGET_SWING: [True, False],
            "pitch_type": ["FF", "FF"],
        }
    )
    # Swinging for -0.10 is worse than a 90%-ball take; taking when swing is
    # 0.10 is likewise worse. Both rows must be regret, never positive credit.
    scored = score_decisions(
        df, _SwingValues([-0.10, 0.10]), _StrikeProbabilities([0.1, 0.9]), _rv()
    )
    assert (scored["decision_value"] < 0).all()
    board = decision_value_by_batter(scored, min_pitches=2)
    assert board.height == 1
    assert board["decision_runs"][0] < 0
    assert board["decision_value_per_100"][0] < 0
