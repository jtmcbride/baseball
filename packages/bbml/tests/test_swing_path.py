"""Swing-path tests.

The interaction in `TestTheFinding` is the reason this model exists, so it is
pinned against real data rather than trusted. If a coordinate mirror, a bad VAA
reconstruction, or a reference-point drift ever flattens it, that test fails
where a metric-only check would keep reporting a plausible-looking AUC.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import get_settings
from bbml.features.swing import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    PITCH,
    PLANE_FEATURE,
    SWING,
    add_swing_features,
    build_swing_frame,
)
from bbml.models.swing_path import ROLES, SwingPathModel, evaluate, rows_for, target_for


def _lake_has_approach_angles() -> bool:
    files = sorted((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet"))
    if not files:
        return False
    return "vaa_deg" in pl.scan_parquet(files[-1]).collect_schema().names()


class TestFeatureContract:
    def test_plane_feature_is_actually_a_feature(self):
        """The counterfactual perturbs this column; if it is not in the model's
        inputs, `plane_value` silently returns zero for every swing."""
        assert PLANE_FEATURE in FEATURE_NAMES

    def test_swing_and_pitch_sets_are_disjoint(self):
        assert not ({f.name for f in SWING} & {f.name for f in PITCH})

    def test_location_and_pitch_type_are_present_as_controls(self):
        """Without these the headline interaction is confounded with 'uppercuts
        miss low breaking balls' — see the module docstring."""
        for control in ("plate_z_norm", "pitch_type", "vaa_deg"):
            assert control in FEATURE_NAMES

    def test_no_duplicate_features(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_unknown_role_is_an_error(self):
        with pytest.raises(KeyError):
            target_for("barrel")


class TestHandednessMirroring:
    def _mirrored_pair(self) -> pl.DataFrame:
        """One swing and its exact left-handed mirror."""
        base = {
            "attack_angle": 12.0,
            "swing_path_tilt": 32.0,
            "bat_speed": 72.0,
            "swing_length": 7.2,
            "is_whiff": False,
            "intercept_ball_minus_batter_pos_y_inches": 30.0,
        }
        return pl.DataFrame(
            [
                {
                    **base,
                    "stand": "R",
                    "plate_x": -0.8,
                    "attack_direction": -10.0,
                    "intercept_ball_minus_batter_pos_x_inches": -6.0,
                },
                {
                    **base,
                    "stand": "L",
                    "plate_x": 0.8,
                    "attack_direction": 10.0,
                    "intercept_ball_minus_batter_pos_x_inches": 6.0,
                },
            ]
        )

    def test_mirrored_swings_land_on_identical_features(self):
        out = add_swing_features(self._mirrored_pair())
        for col in ("plate_x_pull", "attack_direction_pull", "intercept_x_pull"):
            assert out[col][0] == pytest.approx(out[col][1]), col

    def test_pull_side_is_positive_for_both_hands(self):
        out = add_swing_features(self._mirrored_pair())
        assert (out["plate_x_pull"] > 0).all()
        assert (out["attack_direction_pull"] > 0).all()


class TestRowSelection:
    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "is_in_play": [True, False, True],
                "is_whiff": [0, 1, 0],
                "estimated_woba_using_speedangle": [0.5, None, None],
            }
        )

    def test_whiff_head_uses_every_swing(self):
        assert rows_for("whiff", self._frame()).height == 3

    def test_contact_head_is_only_defined_where_contact_happened(self):
        """A whiff has no xwOBA, and treating its null as a zero would tell the
        model that missing the ball is merely very weak contact."""
        assert rows_for("contact", self._frame()).height == 1


@pytest.fixture(scope="module")
def swings() -> pl.DataFrame:
    if not _lake_has_approach_angles():
        pytest.skip("lake has no vaa_deg — run `bb build pitches`")
    frame = build_swing_frame()
    if frame.height < 50_000:
        pytest.skip("not enough tracked swings for a meaningful test")
    return frame


class TestTheFinding:
    """The interaction the whole model rests on, asserted against real data."""

    def test_swing_plane_is_good_or_bad_only_relative_to_the_pitch(self, swings):
        steep_swing = swings.filter(pl.col("attack_angle") > 20)
        flat_swing = swings.filter(pl.col("attack_angle") < 0)
        steep_pitch = pl.col("vaa_deg") < -8
        flat_pitch = pl.col("vaa_deg") > -5.5

        steep_vs_steep = steep_swing.filter(steep_pitch)["is_whiff"].mean()
        steep_vs_flat = steep_swing.filter(flat_pitch)["is_whiff"].mean()
        flat_vs_steep = flat_swing.filter(steep_pitch)["is_whiff"].mean()
        flat_vs_flat = flat_swing.filter(flat_pitch)["is_whiff"].mean()

        # An uppercut is punished by steep pitches and rewarded by flat ones.
        assert steep_vs_steep > steep_vs_flat * 2
        # A flat swing runs the other way — which is what makes this an
        # interaction rather than "steep pitches are just harder to hit".
        assert flat_vs_flat > flat_vs_steep

    def test_approach_angle_orders_pitch_types_correctly(self, swings):
        by_type = (
            swings.group_by("pitch_type")
            .agg(pl.col("vaa_deg").mean().alias("vaa"), pl.len().alias("n"))
            .filter(pl.col("n") > 2000)
        )
        vaa = dict(zip(by_type["pitch_type"].to_list(), by_type["vaa"].to_list(), strict=True))
        # Curveballs arrive steeply, four-seamers flat. If the y=50 reference is
        # ever mistaken for release, this ordering collapses toward equality.
        assert vaa["CU"] < vaa["SL"] < vaa["FF"]
        assert vaa["FF"] > -6.0


@pytest.fixture(scope="module")
def trained(swings):
    train = swings.filter(pl.col("season") < swings["season"].max())
    test = swings.filter(pl.col("season") == swings["season"].max())
    if min(train.height, test.height) < 10_000:
        pytest.skip("need two seasons of swing tracking")
    # 120 rounds was enough back when this fixture only fed AUC/shape checks.
    # It is not enough to resolve the attack_angle x vaa_deg interaction that
    # test_plane_value_is_positive_when_the_plane_helps checks against real
    # data (measured: mean plane_value on the favourable slice is still the
    # wrong sign at 120 and 300 rounds, stable and positive by 600).
    return SwingPathModel(role="whiff").fit(train, num_boost_round=600), train, test


class TestSwingPathModel:
    def test_whiff_head_beats_a_coin_flip_by_a_lot(self, trained):
        model, _, test = trained
        ev = evaluate(model, test)
        assert ev["auc"] > 0.70

    def test_plane_value_is_zero_when_the_swing_is_league_average(self, trained):
        """The counterfactual is the model scored against itself here, so any
        non-zero result means the perturbation is not reaching the model.

        The self-consistent input is the *matched* neutral swing (attack_angle
        plus the correlated bat speed/tilt/contact point, all at their
        league-median-attack-angle values) — not the hitter's real swing with
        attack_angle alone swapped in, which is a different, out-of-distribution
        point since matched_neutral fixed exactly that gap."""
        model, _, test = trained
        overrides = {PLANE_FEATURE: model.league_plane, **model.matched_neutral}
        neutral = test.head(2000).with_columns(
            [pl.lit(v).alias(k) for k, v in overrides.items()]
        )
        np.testing.assert_allclose(model.plane_value(neutral), 0.0, atol=1e-12)

    def test_plane_value_is_positive_when_the_plane_helps(self, trained):
        """Sign convention: positive is better for the batter in both heads.

        A flat swing against flat pitches beats a league-median plane there, so
        it must score positive — the direction is easy to invert silently.
        """
        model, _, test = trained
        favourable = test.filter((pl.col("vaa_deg") > -5.0) & (pl.col("attack_angle") > 20))
        if favourable.height < 500:
            pytest.skip("not enough uppercuts against flat pitches")
        assert model.plane_value(favourable).mean() > 0

    def test_round_trips_through_disk(self, trained, tmp_path):
        model, _, test = trained
        loaded = SwingPathModel.load(model.save(tmp_path / "swing"))
        sample = test.head(500)
        np.testing.assert_allclose(loaded.predict(sample), model.predict(sample))
        assert loaded.league_plane == model.league_plane
        assert loaded.matched_neutral == model.matched_neutral

    def test_scoring_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            SwingPathModel(role="whiff").predict(pl.DataFrame({"attack_angle": [10.0]}))

    def test_every_role_declares_a_target(self):
        assert {target_for(r) for r in ROLES} == {"is_whiff", "estimated_woba_using_speedangle"}

    def test_categoricals_are_encoded_not_dropped(self, trained):
        model, _, _ = trained
        assert set(model.cat_maps) == set(CATEGORICAL_FEATURES)
        assert model.cat_maps["pitch_type"]
