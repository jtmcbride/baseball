"""Unit tests for the derived columns. These are the transforms that fail silently."""

from __future__ import annotations

import polars as pl
import pytest

from bbetl.transforms.statcast import enrich


def _pitch(**overrides) -> dict:
    base = {
        "game_date": "2025-06-14",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitch_type": "FF",
        "description": "called_strike",
        "p_throws": "R",
        "stand": "R",
        "pfx_x": -0.5,
        "pfx_z": 1.3,
        "api_break_x_arm": 0.5,
        "api_break_z_with_gravity": 1.2,
        "plate_x": 0.0,
        "plate_z": 2.5,
        "release_speed": 94.0,
        # A generic four-seam physics fit, valid at the y=50 reference.
        "vx0": 5.0,
        "vy0": -135.0,
        "vz0": -5.0,
        "ax": -10.0,
        "ay": 28.0,
        "az": -15.0,
        "release_pos_x": -1.9,
        "release_pos_z": 5.8,
        "sz_top": 3.4,
        "sz_bot": 1.6,
        "balls": 1,
        "strikes": 2,
        "outs_when_up": 1,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "hc_x": None,
        "hc_y": None,
    }
    base.update(overrides)
    return base


def frame(*rows: dict) -> pl.DataFrame:
    return enrich(pl.DataFrame(list(rows)))


class TestArmSideNormalization:
    """The handedness flip. If it inverts, every movement plot is wrong and
    nothing raises — so it gets the most direct test in the suite."""

    def test_rhp_arm_side_is_positive(self):
        # A RHP sinker runs to the arm side, which is negative pfx_x.
        df = frame(_pitch(p_throws="R", pfx_x=-1.25, api_break_x_arm=1.25))
        assert df["hb_arm_in"][0] == pytest.approx(15.0)

    def test_lhp_arm_side_is_also_positive(self):
        # Same pitch shape from the other side: positive pfx_x, same arm-side sign.
        df = frame(_pitch(p_throws="L", pfx_x=1.25, api_break_x_arm=1.25))
        assert df["hb_arm_in"][0] == pytest.approx(15.0)

    def test_both_hands_agree_after_normalization(self):
        df = frame(
            _pitch(p_throws="R", pfx_x=-1.25, api_break_x_arm=1.25),
            _pitch(p_throws="L", pfx_x=1.25, api_break_x_arm=1.25, pitch_number=2),
        )
        assert df["hb_arm_in"][0] == pytest.approx(df["hb_arm_in"][1])
        # ...while the raw values still disagree, which is the whole point.
        assert df["hb_in"][0] == pytest.approx(-df["hb_in"][1])

    def test_falls_back_when_savant_column_is_null(self):
        """Pre-2015 or partial rows lack api_break_x_arm; the flip must still apply."""
        rhp = frame(_pitch(p_throws="R", pfx_x=-1.0, api_break_x_arm=None))
        lhp = frame(_pitch(p_throws="L", pfx_x=1.0, api_break_x_arm=None))
        assert rhp["hb_arm_in"][0] == pytest.approx(12.0)
        assert lhp["hb_arm_in"][0] == pytest.approx(12.0)


class TestZoneNormalization:
    def test_top_of_zone_is_one(self):
        df = frame(_pitch(plate_z=3.4, sz_top=3.4, sz_bot=1.6))
        assert df["plate_z_norm"][0] == pytest.approx(1.0)

    def test_bottom_of_zone_is_zero(self):
        df = frame(_pitch(plate_z=1.6, sz_top=3.4, sz_bot=1.6))
        assert df["plate_z_norm"][0] == pytest.approx(0.0)

    def test_normalization_makes_tall_and_short_hitters_comparable(self):
        """The reason this column exists: identical relative location, different
        absolute height."""
        tall = frame(_pitch(plate_z=3.3, sz_top=3.6, sz_bot=1.8))
        short = frame(_pitch(plate_z=2.75, sz_top=3.0, sz_bot=1.5))
        assert tall["plate_z_norm"][0] == pytest.approx(short["plate_z_norm"][0], abs=0.02)

    def test_degenerate_zone_yields_null_not_infinity(self):
        df = frame(_pitch(sz_top=2.0, sz_bot=2.0))
        assert df["plate_z_norm"][0] is None


class TestOutcomeFlags:
    def test_foul_tip_is_contact_not_a_whiff(self):
        """Counting foul tips as whiffs inflates whiff rate ~1pp. Common error."""
        df = frame(_pitch(description="foul_tip"))
        assert df["is_swing"][0] is True
        assert df["is_whiff"][0] is False

    @pytest.mark.parametrize("desc", ["swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    def test_whiffs(self, desc):
        df = frame(_pitch(description=desc))
        assert df["is_whiff"][0] is True
        assert df["is_swing"][0] is True

    def test_csw_combines_called_strikes_and_whiffs(self):
        assert frame(_pitch(description="called_strike"))["is_csw"][0] is True
        assert frame(_pitch(description="swinging_strike"))["is_csw"][0] is True
        assert frame(_pitch(description="ball"))["is_csw"][0] is False
        assert frame(_pitch(description="foul"))["is_csw"][0] is False

    @pytest.mark.parametrize("desc", ["automatic_ball", "automatic_strike"])
    def test_automatic_calls_are_not_tracked_pitches(self, desc):
        """Pitch-clock/ABS calls carry no tracking data and must not reach a
        model or a movement plot."""
        df = frame(_pitch(description=desc))
        assert df["is_tracked_pitch"][0] is False

    @pytest.mark.parametrize("desc", ["pitchout", "intent_ball"])
    def test_uncompetitive_pitches_are_tracked_but_not_competitive(self, desc):
        """An intentional ball is a real, measured pitch and not an attempt to
        get anyone out. It was missing from the exclusion set until 2026-08-16,
        which put ~3,700 deliberate 4-11ft misses into every command metric."""
        df = frame(_pitch(description=desc))
        assert df["is_tracked_pitch"][0] is True
        assert df["is_competitive"][0] is False


class TestImpossibleTracking:
    """Quarantine of corrupt tracking records.

    Bounds reject impossibility, not unusualness — the tests pin both sides,
    because a guard tight enough to catch a real lob would be worse than none.
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("release_speed", 21.7),  # slower than a ball can reach the plate
            ("release_speed", 120.0),  # faster than anyone has thrown
            ("plate_x", 35.0),  # in the stands
            ("plate_z", -57.6),  # underground
            ("release_pos_z", -3.2),  # released below ground level
            ("release_pos_z", 11.4),  # released 11ft in the air
            ("release_pos_x", 18.0),  # released off the mound entirely
        ],
    )
    def test_impossible_measurements_are_not_tracked(self, field, value):
        df = frame(_pitch(**{field: value}))
        assert df["is_tracked_pitch"][0] is False

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("plate_x", 11.3),  # a real intentional ball
            ("release_speed", 30.0),  # a real position-player lob
            ("plate_z", -1.0),  # a real pitch spiked in front of the plate
        ],
    )
    def test_extreme_but_real_measurements_survive(self, field, value):
        df = frame(_pitch(**{field: value}))
        assert df["is_tracked_pitch"][0] is True

    def test_a_missing_measurement_is_not_impossible(self):
        """Older seasons have whole columns unpopulated; null is absence."""
        df = frame(_pitch(release_speed=None, release_pos_x=None, release_pos_z=None))
        assert df["is_tracked_pitch"][0] is True

    def test_a_quarantined_pitch_still_occupies_its_row(self):
        """It moved the count, so deleting it would corrupt the sequence."""
        df = frame(_pitch(plate_z=-57.6, description="ball"))
        assert df.height == 1
        assert df["is_tracked_pitch"][0] is False


class TestZoneAndChase:
    def test_pitch_down_the_middle_is_in_zone(self):
        df = frame(_pitch(plate_x=0.0, plate_z=2.5))
        assert df["is_in_zone"][0] is True

    def test_pitch_off_the_plate_is_out_of_zone(self):
        df = frame(_pitch(plate_x=1.5, plate_z=2.5))
        assert df["is_in_zone"][0] is False

    def test_chase_requires_swing_outside_zone(self):
        chased = frame(_pitch(plate_x=1.5, description="swinging_strike"))
        took = frame(_pitch(plate_x=1.5, description="ball"))
        in_zone_swing = frame(_pitch(plate_x=0.0, description="swinging_strike"))
        assert chased["is_chase"][0] is True
        assert took["is_chase"][0] is False
        assert in_zone_swing["is_chase"][0] is False


class TestGameState:
    def test_base_state_is_a_three_bit_code(self):
        empty = frame(_pitch())
        first = frame(_pitch(on_1b=123))
        loaded = frame(_pitch(on_1b=1, on_2b=2, on_3b=3))
        assert empty["base_state"][0] == 0
        assert first["base_state"][0] == 1
        assert loaded["base_state"][0] == 7

    def test_base_out_state_spans_24_values(self):
        loaded_two_out = frame(_pitch(on_1b=1, on_2b=2, on_3b=3, outs_when_up=2))
        bases_empty_no_outs = frame(_pitch(outs_when_up=0))
        assert loaded_two_out["base_out_state"][0] == 23
        assert bases_empty_no_outs["base_out_state"][0] == 0

    def test_count_state(self):
        assert frame(_pitch(balls=3, strikes=2))["count_state"][0] == "3-2"

    def test_platoon_flag(self):
        assert frame(_pitch(stand="R", p_throws="R"))["is_platoon_same"][0] is True
        assert frame(_pitch(stand="L", p_throws="R"))["is_platoon_same"][0] is False


def test_enrich_is_a_noop_on_empty_input():
    assert enrich(pl.DataFrame()).height == 0


class TestHitCoordinates:
    """hc_x/hc_y -> feet-from-home-plate, viz #8's spray chart. Constants are
    measured, not the published defaults -- see `statcast.py`'s own comment
    for the fit against real `hit_distance_sc` data (origin confirmed within a
    foot of the community-published one; scale corrected from 2.495 to
    2.339)."""

    def test_ball_hit_straight_up_the_middle_lands_on_the_x_axis(self):
        # hc_x == HC_X0 (home plate's own x) means dead centre: x_ft == 0.
        df = frame(_pitch(hc_x=125.91, hc_y=38.0))
        assert df["x_ft"][0] == pytest.approx(0.0, abs=1e-6)
        assert df["spray_angle_deg"][0] == pytest.approx(0.0, abs=1e-6)

    def test_derived_distance_matches_a_known_deep_center_field_shot(self):
        # A ~378ft blast dead centre: y_ft = (199.54-38.0)*2.339 = 377.87ft.
        df = frame(_pitch(hc_x=125.91, hc_y=38.0))
        assert df["hit_distance_derived_ft"][0] == pytest.approx(377.87, abs=0.5)

    def test_third_base_side_is_negative_x_first_base_side_is_positive(self):
        # Gameday's pixel x grows rightward (toward 1B from the plate's view);
        # hc_x < home plate's own x is therefore the 3B/LF side.
        pulled_left = frame(_pitch(hc_x=90.0, hc_y=120.0))
        pulled_right = frame(_pitch(hc_x=160.0, hc_y=120.0))
        assert pulled_left["x_ft"][0] < 0
        assert pulled_right["x_ft"][0] > 0

    def test_spray_angle_sign_follows_x_ft(self):
        left = frame(_pitch(hc_x=90.0, hc_y=120.0))
        right = frame(_pitch(hc_x=160.0, hc_y=120.0))
        assert left["spray_angle_deg"][0] < 0
        assert right["spray_angle_deg"][0] > 0

    def test_null_hit_coordinates_yield_null_derived_columns(self):
        df = frame(_pitch(hc_x=None, hc_y=None))
        assert df["x_ft"][0] is None
        assert df["hit_distance_derived_ft"][0] is None


class TestApproachAngles:
    """VAA/HAA are reconstructed, not shipped by Savant, and feed the swing-path
    model. They are also the third home of the y=50 / y=17/12 constants, so the
    ordering assertions here are what catch a reference-point drift."""

    def test_four_seam_vaa_is_shallow_and_negative(self):
        # Descending, but the flattest thing anyone throws.
        df = frame(_pitch())
        assert df["vaa_deg"][0] == pytest.approx(-4.87, abs=0.05)

    def test_a_curveball_approaches_more_steeply_than_a_fastball(self):
        """The single most useful sanity check on this column. If the y=50
        reference is ever mistaken for the release point, every pitch flattens
        toward the same angle and this ordering collapses."""
        fastball = frame(_pitch())["vaa_deg"][0]
        curve = frame(_pitch(vy0=-100.0, vz0=-3.0, az=-28.0, ay=22.0, release_speed=79.0))[
            "vaa_deg"
        ][0]
        assert curve < fastball
        assert curve < -8.0

    def test_the_angle_is_taken_at_the_plate_not_at_the_reference(self):
        """atan2(vz0, |vy0|) at y=50 is the shortcut this must not be. Gravity
        has 48 more feet to act, so the true angle is meaningfully steeper."""
        import math

        naive = math.degrees(math.atan2(-5.0, 135.0))
        assert frame(_pitch())["vaa_deg"][0] < naive - 2.0

    def test_haa_sign_follows_horizontal_velocity(self):
        assert frame(_pitch(vx0=5.0))["haa_deg"][0] > 0
        assert frame(_pitch(vx0=-5.0))["haa_deg"][0] < 0

    def test_a_corrupt_fit_yields_null_rather_than_infinity(self):
        """ay=0 divides by zero; a trajectory that never reaches the plate has a
        negative discriminant. Neither may produce an inf that poisons a mean."""
        assert frame(_pitch(ay=0.0))["vaa_deg"][0] is None
        assert frame(_pitch(vy0=-1.0, ay=28.0))["vaa_deg"][0] is None
