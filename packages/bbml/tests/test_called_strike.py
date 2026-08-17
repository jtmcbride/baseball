"""Called-strike model tests.

`framing_runs` is a sum of residuals, so the properties worth pinning are the
ones a metric-only check would miss: that catcher/umpire never leak into the
feature set (the entire premise of the residual), that `marginal_strike_value`
routes terminal counts to the right table, and that the ECE calibration gate
actually catches a miscalibrated model rather than just a low-AUC one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import get_settings
from bbml.features.called_strike import (
    CATCHER_COLUMN,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TAKE_DESCRIPTIONS,
    TARGET_CALLED_STRIKE,
    UMPIRE_COLUMN,
    add_called_strike_features,
    build_called_strike_frame,
)
from bbml.features.run_value import RunValue
from bbml.models.called_strike import (
    CalledStrikeModel,
    binary_ece,
    evaluate,
    framing_runs,
    umpire_zone_rate,
)


def _lake_available() -> bool:
    return any((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet"))


# --- feature contract (no data needed) ---------------------------------------


class TestFeatureContract:
    def test_catcher_and_umpire_never_leak_into_the_feature_set(self):
        """The whole premise of `framing_runs` is that the model does not
        already know who is behind the plate."""
        assert CATCHER_COLUMN not in FEATURE_NAMES
        assert UMPIRE_COLUMN not in FEATURE_NAMES

    def test_hit_by_pitch_is_not_a_take(self):
        assert "hit_by_pitch" not in TAKE_DESCRIPTIONS

    def test_no_duplicate_features(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_pitch_type_is_the_only_categorical(self):
        assert CATEGORICAL_FEATURES == ["pitch_type"]


class TestHandednessMirroring:
    def test_outside_is_positive_for_both_batter_hands(self):
        df = pl.DataFrame(
            {
                "stand": ["R", "L"],
                "plate_x": [0.8, -0.8],
                TARGET_CALLED_STRIKE: [True, True],
            }
        )
        out = add_called_strike_features(df)
        assert (out["plate_x_out"] > 0).all()


# --- marginal strike value -----------------------------------------------------

_EVENT_RV = {"strikeout": -0.20, "walk": 0.30, "single": 0.45}


def _pa(game_pk, at_bat, pitches, *, xwoba=None):
    return [
        {
            "game_pk": game_pk,
            "at_bat_number": at_bat,
            "pitch_number": i + 1,
            "balls": b,
            "strikes": s,
            "description": d,
            "events": e,
            "delta_run_exp": _EVENT_RV.get(e or "", 0.0),
            "estimated_woba_using_speedangle": xwoba if d == "hit_into_play" else None,
        }
        for i, (b, s, d, e) in enumerate(pitches)
    ]


@pytest.fixture(scope="module")
def synthetic_rv() -> RunValue:
    rows = []
    for i in range(200):
        rows += _pa(
            1,
            i * 5 + 1,
            [
                (0, 0, "called_strike", None),
                (0, 1, "called_strike", None),
                (0, 2, "swinging_strike", "strikeout"),
            ],
        )
        rows += _pa(
            1,
            i * 5 + 2,
            [*[(b, 0, "ball", None) for b in range(3)], (3, 0, "ball", "walk")],
        )
        rows += _pa(
            1,
            i * 5 + 3,
            [(0, 0, "ball", None), (1, 0, "hit_into_play", "single")],
            xwoba=0.3 + 0.003 * i,
        )
        # Long counts, to populate the 1-1/2-1/3-1/3-2 corner of the table.
        rows += _pa(
            1,
            i * 5 + 4,
            [
                (0, 0, "ball", None),
                (1, 0, "called_strike", None),
                (1, 1, "ball", None),
                (2, 1, "ball", None),
                (3, 1, "called_strike", None),
                (3, 2, "ball" if i % 2 else "called_strike", "walk" if i % 2 else "strikeout"),
            ],
        )
        # And the 1-2/2-2 corner.
        rows += _pa(
            1,
            i * 5 + 5,
            [
                (0, 0, "ball", None),
                (1, 0, "called_strike", None),
                (1, 1, "called_strike", None),
                (1, 2, "ball", None),
                (2, 2, "called_strike", "strikeout"),
            ],
        )
    return RunValue.fit(pl.DataFrame(rows))


class TestMarginalStrikeValue:
    def test_full_count_routes_to_terminal_events_not_a_missing_re_entry(self, synthetic_rv):
        """A ball at 3-2 is a walk and a strike is a strikeout — neither is a
        reachable in-progress count, so this must not silently come back null."""
        got = synthetic_rv.marginal_strike_value(pl.Series([3]), pl.Series([2]))
        expected = synthetic_rv.event_value["walk"] - synthetic_rv.event_value["strikeout"]
        assert got[0] == pytest.approx(expected)

    def test_a_strike_call_is_never_better_for_the_batter_than_a_ball_call(self, synthetic_rv):
        balls = pl.Series([0, 1, 2, 3, 0, 1, 2])
        strikes = pl.Series([0, 0, 0, 0, 1, 1, 2])
        got = synthetic_rv.marginal_strike_value(balls, strikes)
        assert (got > 0).all()

    def test_zero_zero_count_matches_the_re_table_directly(self, synthetic_rv):
        got = synthetic_rv.marginal_strike_value(pl.Series([0]), pl.Series([0]))
        expected = synthetic_rv.count_re["1-0"] - synthetic_rv.count_re["0-1"]
        assert got[0] == pytest.approx(expected)


# --- binary ECE / AUC ----------------------------------------------------------


class TestBinaryECE:
    def test_perfectly_calibrated_predictions_score_near_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20_000)
        y = (rng.uniform(0, 1, 20_000) < p).astype(float)
        assert binary_ece(y, p) < 0.02

    def test_a_constant_overconfidence_offset_is_caught(self):
        rng = np.random.default_rng(0)
        true_p = rng.uniform(0.3, 0.7, 20_000)
        y = (rng.uniform(0, 1, 20_000) < true_p).astype(float)
        overconfident = np.clip(true_p + 0.2, 0, 1)
        assert binary_ece(y, overconfident) > 0.15

    def test_empty_input_is_zero_not_an_error(self):
        assert binary_ece(np.array([]), np.array([])) == 0.0


class TestAUC:
    def test_perfect_separation_is_one(self):
        from bbml.models.called_strike import _auc

        y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _auc(y, p) == pytest.approx(1.0)

    def test_inverted_ranking_is_zero(self):
        from bbml.models.called_strike import _auc

        y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _auc(y, p) == pytest.approx(0.0)


# --- framing runs ----------------------------------------------------------------


class _FixedProbaModel:
    """Test double: `predict_proba` returns a fixed array regardless of input."""

    def __init__(self, p: np.ndarray) -> None:
        self._p = p

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        return self._p


class TestFramingRuns:
    def test_more_strikes_than_expected_is_positive_credit(self, synthetic_rv):
        df = pl.DataFrame(
            {
                CATCHER_COLUMN: [1] * 600,
                "balls": [0] * 600,
                "strikes": [0] * 600,
                TARGET_CALLED_STRIKE: [1] * 600,  # every taken pitch called a strike
            }
        )
        model = _FixedProbaModel(np.full(600, 0.5))  # model expected 50/50
        out = framing_runs(df, model, synthetic_rv, min_pitches=500)
        assert out.height == 1
        assert out["framing_runs"][0] > 0

    def test_fewer_strikes_than_expected_is_negative_credit(self, synthetic_rv):
        df = pl.DataFrame(
            {
                CATCHER_COLUMN: [1] * 600,
                "balls": [0] * 600,
                "strikes": [0] * 600,
                TARGET_CALLED_STRIKE: [0] * 600,
            }
        )
        model = _FixedProbaModel(np.full(600, 0.5))
        out = framing_runs(df, model, synthetic_rv, min_pitches=500)
        assert out["framing_runs"][0] < 0

    def test_below_the_pitch_minimum_is_dropped(self, synthetic_rv):
        df = pl.DataFrame(
            {
                CATCHER_COLUMN: [1] * 10,
                "balls": [0] * 10,
                "strikes": [0] * 10,
                TARGET_CALLED_STRIKE: [1] * 10,
            }
        )
        model = _FixedProbaModel(np.full(10, 0.5))
        out = framing_runs(df, model, synthetic_rv, min_pitches=500)
        assert out.height == 0


class TestUmpireZoneRate:
    def test_borderline_only_umpire_who_calls_more_strikes_shows_positive_edge(self):
        n = 600
        df = pl.DataFrame(
            {
                UMPIRE_COLUMN: [1] * n,
                TARGET_CALLED_STRIKE: [1] * n,  # always a strike
            }
        )
        model = _FixedProbaModel(np.full(n, 0.5))  # expected 50/50
        out = umpire_zone_rate(df, model, min_pitches=500)
        assert out.height == 1
        assert out["edge"][0] == pytest.approx(0.5, abs=1e-9)

    def test_pitches_far_from_the_edge_are_excluded(self):
        n = 600
        df = pl.DataFrame(
            {
                UMPIRE_COLUMN: [1] * n,
                TARGET_CALLED_STRIKE: [1] * n,
            }
        )
        model = _FixedProbaModel(np.full(n, 0.99))  # nowhere near borderline
        out = umpire_zone_rate(df, model, min_pitches=500)
        assert out.height == 0


# --- model against real data ---------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    if not _lake_available():
        pytest.skip("no lake built")
    frame = build_called_strike_frame()
    if frame.height < 50_000:
        pytest.skip("not enough taken pitches for a meaningful test")
    train = frame.filter(pl.col("season") < frame["season"].max())
    test = frame.filter(pl.col("season") == frame["season"].max())
    if train.height < 10_000 or test.height < 10_000:
        pytest.skip("lake has only one season")
    model = CalledStrikeModel().fit(train, num_boost_round=400)
    return model, train, test


class TestCalledStrikeModel:
    def test_beats_a_coin_flip_by_a_lot(self, trained):
        model, _, test = trained
        ev = evaluate(model, test)
        assert ev.auc > 0.85

    def test_is_calibrated_on_real_held_out_data(self, trained):
        """The hard gate — see the module docstring on why AUC alone is not
        enough to trust this model for framing runs."""
        model, _, test = trained
        ev = evaluate(model, test)
        assert ev.ece < 0.03

    def test_round_trips_through_disk(self, trained, tmp_path):
        model, _, test = trained
        loaded = CalledStrikeModel.load(model.save(tmp_path / "called_strike"))
        sample = test.head(500)
        np.testing.assert_allclose(loaded.predict_proba(sample), model.predict_proba(sample))

    def test_scoring_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            CalledStrikeModel().predict_proba(pl.DataFrame({"plate_x_out": [0.0]}))

    def test_a_mismatched_feature_list_refuses_to_load(self, trained, tmp_path, monkeypatch):
        import bbml.models.called_strike as cs

        model, _, _ = trained
        directory = model.save(tmp_path / "called_strike")
        monkeypatch.setattr(cs, "FEATURE_NAMES", FEATURE_NAMES[:-1])
        with pytest.raises(ValueError, match="does not match"):
            CalledStrikeModel.load(directory)
