"""Stuff+ / Location+ / Pitching+ tests.

The properties pinned here are the ones that fail silently. A run value that
quietly keeps its base-out context, a stuff feature set that quietly acquires a
location column, or a handedness mirror applied to one hand and not the other
all produce a model that trains fine, scores fine, and means something other
than what it claims to.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import get_settings
from bbml.features import stuff as sf
from bbml.features.run_value import RunValue
from bbml.features.stuff import (
    LOCATION,
    STUFF,
    TARGET_RUN_VALUE,
    add_pitch_quality_features,
    assert_sets_are_disjoint,
    build_pitch_quality_frame,
    categorical_features,
    feature_names,
    primary_fastball,
)
from bbml.models.pitch_quality import (
    CALIBRATION_MIN_PITCHES,
    PitchQualityModel,
    _spearman,
    aggregate_correlation,
    evaluate,
    stability,
)


def _lake_available() -> bool:
    return any((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet"))


# --- feature contract (no data needed) ---------------------------------------


class TestFeatureSets:
    def test_stuff_and_location_are_disjoint(self):
        assert_sets_are_disjoint()

    def test_a_location_column_in_the_stuff_set_is_caught(self, monkeypatch):
        polluted = [*STUFF, LOCATION[0]]
        monkeypatch.setattr(sf, "STUFF", polluted)
        with pytest.raises(AssertionError, match="disjoint"):
            sf.assert_sets_are_disjoint()

    def test_pitching_is_the_union(self):
        assert set(feature_names("pitching")) == set(feature_names("stuff")) | set(
            feature_names("location")
        )

    def test_no_duplicate_feature_names(self):
        for role in sf.ROLES:
            names = feature_names(role)
            assert len(names) == len(set(names))

    def test_stuff_carries_no_categoricals(self):
        # Measured to not help, and documented as such — see features/stuff.py.
        assert categorical_features("stuff") == []

    def test_unknown_role_is_an_error(self):
        with pytest.raises(KeyError):
            feature_names("command")


# --- run value ----------------------------------------------------------------


_EVENT_RV = {"strikeout": -0.2, "walk": 0.3, "single": 0.45}


def _pa(
    game_pk: int,
    at_bat: int,
    pitches: list[tuple[int, int, str, str | None]],
    *,
    xwoba: float | None = None,
):
    """(balls, strikes, description, events) -> rows for one plate appearance."""
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
def synthetic_rv() -> pl.DataFrame:
    """A tiny league: every PA is a strikeout, a walk, or a single.

    The singles carry a spread of xwOBA values because the contact de-noiser is
    a regression — a league where every ball in play looks identical has nothing
    to fit.
    """
    rows = []
    for i in range(200):
        rows += _pa(
            1,
            i * 3 + 1,
            [
                (0, 0, "called_strike", None),
                (0, 1, "called_strike", None),
                (0, 2, "swinging_strike", "strikeout"),
            ],
        )
        rows += _pa(
            1,
            i * 3 + 2,
            [*[(b, 0, "ball", None) for b in range(3)], (3, 0, "ball", "walk")],
        )
        rows += _pa(
            1,
            i * 3 + 3,
            [(0, 0, "ball", None), (1, 0, "hit_into_play", "single")],
            xwoba=0.3 + 0.003 * i,
        )
    return pl.DataFrame(rows)


class TestRunValue:
    def test_mean_run_value_is_zero_in_every_count(self, synthetic_rv):
        """The defining property: the count itself carries no run value."""
        rv = RunValue.fit(synthetic_rv)
        out = rv.attach(synthetic_rv)
        by_count = out.group_by(["balls", "strikes"]).agg(
            pl.col(TARGET_RUN_VALUE).mean().alias("rv")
        )
        assert by_count["rv"].abs().max() < 1e-9

    def test_a_two_strike_foul_is_worth_exactly_zero(self, synthetic_rv):
        """Documented consequence of the count-based framing, not a bug."""
        rv = RunValue.fit(synthetic_rv)
        foul = pl.DataFrame(
            _pa(2, 1, [(0, 2, "foul", None), (0, 2, "swinging_strike", "strikeout")])
        )
        out = rv.attach(pl.concat([synthetic_rv, foul]))
        got = out.filter((pl.col("game_pk") == 2) & (pl.col("pitch_number") == 1))
        assert got[TARGET_RUN_VALUE][0] == pytest.approx(0.0, abs=1e-12)

    def test_pitcher_sign_convention(self, synthetic_rv):
        """Higher is better for the pitcher — the opposite of Savant's."""
        rv = RunValue.fit(synthetic_rv)
        out = rv.attach(synthetic_rv)
        by_desc = dict(
            out.group_by("description").agg(pl.col(TARGET_RUN_VALUE).mean()).iter_rows()  # type: ignore[arg-type]
        )
        assert by_desc["called_strike"] > 0 > by_desc["ball"]

    def test_count_run_expectancy_favours_the_batter_as_balls_accumulate(self, synthetic_rv):
        rv = RunValue.fit(synthetic_rv)
        assert rv.count_re["3-0"] > rv.count_re["0-0"] > rv.count_re["0-2"]

    def test_round_trips_through_disk(self, synthetic_rv, tmp_path):
        rv = RunValue.fit(synthetic_rv)
        loaded = RunValue.load(rv.save(tmp_path / "run_value.json"))
        assert loaded.count_re == rv.count_re
        assert loaded.bip_slope == rv.bip_slope

    def test_missing_columns_fail_loudly(self):
        with pytest.raises(ValueError, match="Run value needs columns"):
            RunValue.fit(pl.DataFrame({"game_pk": [1]}))


# --- handedness normalization -------------------------------------------------


class TestHandednessMirroring:
    def _mirrored_pair(self) -> pl.DataFrame:
        """The same pitch thrown by a RHP and its exact left-handed mirror."""
        base = {
            "pitcher": 1,
            "season": 2025,
            "pitch_type": "FF",
            "release_speed": 95.0,
            "ivb_in": 15.0,
            "hb_arm_in": 8.0,
            "release_pos_z": 6.0,
            "is_platoon_same": True,
            "plate_z": 2.5,
            "plate_z_norm": 0.5,
        }
        return pl.DataFrame(
            [
                {
                    **base,
                    "p_throws": "R",
                    "stand": "R",
                    "release_pos_x": -2.0,
                    "spin_axis": 210.0,
                    "plate_x": 0.6,
                },
                {
                    **base,
                    "pitcher": 2,
                    "p_throws": "L",
                    "stand": "L",
                    "release_pos_x": 2.0,
                    "spin_axis": 150.0,
                    "plate_x": -0.6,
                },
            ]
        )

    def test_mirrored_pitches_land_on_identical_features(self):
        out = add_pitch_quality_features(self._mirrored_pair())
        for col in ("release_pos_x_arm", "spin_axis_sin", "spin_axis_cos", "plate_x_out"):
            assert out[col][0] == pytest.approx(out[col][1], abs=1e-9), col

    def test_arm_side_release_is_positive_for_both_hands(self):
        out = add_pitch_quality_features(self._mirrored_pair())
        assert (out["release_pos_x_arm"] > 0).all()

    def test_outside_is_positive_for_both_batter_hands(self):
        out = add_pitch_quality_features(self._mirrored_pair())
        assert (out["plate_x_out"] > 0).all()

    def test_spin_axis_wraps_rather_than_jumps(self):
        """359 and 1 degrees must be neighbours, which raw degrees would not be."""
        df = pl.DataFrame(
            [
                {"p_throws": "R", "spin_axis": 359.0},
                {"p_throws": "R", "spin_axis": 1.0},
            ]
        ).with_columns(
            pl.lit("R").alias("stand"),
            pl.lit(0.0).alias("release_pos_x"),
            pl.lit(0.0).alias("plate_x"),
            pl.lit(True).alias("is_platoon_same"),
            pl.lit(1).alias("pitcher"),
            pl.lit(2025).alias("season"),
            pl.lit("FF").alias("pitch_type"),
            pl.lit(95.0).alias("release_speed"),
            pl.lit(15.0).alias("ivb_in"),
            pl.lit(8.0).alias("hb_arm_in"),
        )
        out = add_pitch_quality_features(df)
        gap = np.hypot(
            out["spin_axis_sin"][0] - out["spin_axis_sin"][1],
            out["spin_axis_cos"][0] - out["spin_axis_cos"][1],
        )
        assert gap < 0.05


class TestPrimaryFastball:
    def _arsenal(self, kinds: dict[str, int]) -> pl.DataFrame:
        rows = []
        for pt, n in kinds.items():
            rows += [
                {
                    "pitcher": 1,
                    "season": 2025,
                    "pitch_type": pt,
                    "release_speed": {"FF": 95.0, "SI": 94.0, "FC": 89.0}[pt],
                    "ivb_in": 15.0,
                    "hb_arm_in": 8.0,
                }
            ] * n
        return pl.DataFrame(rows)

    def test_a_cutter_never_outranks_a_true_fastball(self):
        """Anchoring on FC reports a sinker as +7mph of "offspeed separation"."""
        fb = primary_fastball(self._arsenal({"FC": 900, "SI": 100}))
        assert fb["fb_velo"][0] == pytest.approx(94.0)

    def test_a_cutter_is_used_when_there_is_no_true_fastball(self):
        fb = primary_fastball(self._arsenal({"FC": 900}))
        assert fb["fb_velo"][0] == pytest.approx(89.0)

    def test_a_handful_of_pitches_is_not_an_arsenal(self):
        assert primary_fastball(self._arsenal({"FF": 3})).height == 0


# --- model --------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    if not _lake_available():
        pytest.skip("no lake built")
    frame = build_pitch_quality_frame()
    if frame.height < 50_000:
        pytest.skip("not enough data for a meaningful pitch quality test")
    train = frame.filter(pl.col("season") < frame["season"].max())
    test = frame.filter(pl.col("season") == frame["season"].max())
    if train.height < 10_000 or test.height < 10_000:
        pytest.skip("lake has only one season")
    rv = RunValue.fit(train)
    train, test = rv.attach(train), rv.attach(test)
    model = PitchQualityModel(role="stuff").fit(train, num_boost_round=60)
    return model, train, test


class TestPitchQualityModel:
    def test_plus_is_centred_on_100_over_the_reference_population(self, trained):
        model, train, _ = trained
        groups = (
            train.with_columns(pl.Series("plus", model.plus(train)))
            .group_by(["pitcher", "season", "pitch_type"])
            .agg(pl.col("plus").mean(), pl.len().alias("n"))
            .filter(pl.col("n") >= CALIBRATION_MIN_PITCHES)
        )
        assert groups["plus"].mean() == pytest.approx(100.0, abs=0.5)
        assert groups["plus"].std() == pytest.approx(10.0, abs=0.5)

    def test_the_grade_tracks_observed_run_value_out_of_sample(self, trained):
        """Weak but real and positive. If this goes negative the model is inverted."""
        model, _, test = trained
        corr, n_groups = aggregate_correlation(model, test)
        assert n_groups > 50
        assert corr > 0.05

    def test_the_grade_is_more_stable_than_the_results_it_grades(self, trained):
        """The whole reason a stuff metric exists."""
        model, train, _ = trained
        yoy = stability(model, train)
        if yoy.height == 0:
            pytest.skip("need two consecutive seasons in train")
        assert (yoy["stuff_yoy"] > yoy["own_rv_yoy"]).all()

    def test_round_trips_through_disk(self, trained, tmp_path):
        model, _, test = trained
        loaded = PitchQualityModel.load(model.save(tmp_path / "stuff"))
        sample = test.head(500)
        np.testing.assert_allclose(loaded.plus(sample), model.plus(sample))

    def test_a_mismatched_feature_list_refuses_to_load(self, trained, tmp_path, monkeypatch):
        model, _, _ = trained
        directory = model.save(tmp_path / "stuff")
        monkeypatch.setitem(sf.FEATURE_SETS, "stuff", STUFF[:-1])
        with pytest.raises(ValueError, match="does not match"):
            PitchQualityModel.load(directory)

    def test_scoring_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            PitchQualityModel(role="stuff").predict_rv(pl.DataFrame({"release_speed": [95.0]}))

    def test_r2_is_near_zero_and_that_is_expected(self, trained):
        """Pinned so nobody reads a 0.001 as a broken model and 'fixes' it."""
        model, _, test = trained
        ev = evaluate(model, test)
        assert -0.01 < ev.r2 < 0.05


class TestSpearman:
    def test_perfect_monotone_agreement(self):
        assert _spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])) == pytest.approx(1)

    def test_perfect_disagreement(self):
        assert _spearman(np.array([1.0, 2, 3, 4]), np.array([40.0, 30, 20, 10])) == pytest.approx(
            -1
        )

    def test_ties_share_a_rank(self):
        assert _spearman(np.array([1.0, 1, 2, 2]), np.array([5.0, 5, 9, 9])) == pytest.approx(1)
