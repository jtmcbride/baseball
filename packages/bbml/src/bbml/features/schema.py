"""The feature contract — the single declaration both callers agree on.

Everything downstream (batch training, live inference, the parity test, the model
registry) reads the feature list from here rather than from a hardcoded list at
each site. A feature added in one place and forgotten in another is exactly the
drift this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind
    doc: str


# --- targets ----------------------------------------------------------------

# The global pitch-type vocabulary. Predictions are MASKED to each pitcher's own
# arsenal at inference time (see `arsenal.py`); this list only fixes the label
# encoding so a model trained today can be loaded tomorrow.
PITCH_TYPES: list[str] = [
    "FF",
    "SI",
    "FC",
    "SL",
    "ST",
    "CU",
    "KC",
    "SV",
    "CH",
    "FS",
    "KN",
    "EP",
    "FA",
    "CS",
    "SC",
]

TARGET_PITCH_TYPE = "target_pitch_type"
TARGET_LOCATION = "target_location"

# Location target: a 5x5 grid over the zone and its immediate surroundings, plus
# one "far miss" class = 26 classes.
#
# The grid deliberately spans wider than the rulebook zone. A pure in-zone grid
# would collapse every ball into one bucket and throw away the direction the
# pitcher missed in — which is most of the signal in an 0-2 count, where nobody
# is trying to hit the zone.
LOC_GRID_N = 5
LOC_X_MIN, LOC_X_MAX = -1.5, 1.5
LOC_Z_MIN, LOC_Z_MAX = -0.5, 1.5
LOC_FAR_MISS_CLASS = LOC_GRID_N * LOC_GRID_N  # == 25
N_LOCATION_CLASSES = LOC_GRID_N * LOC_GRID_N + 1  # == 26


# --- features ---------------------------------------------------------------
#
# Every feature here must be knowable BEFORE the pitch is released. Anything
# describing the pitch itself (velocity, movement, result, location) is a label
# or a leak, never an input.

GAME_STATE: list[Feature] = [
    Feature("balls", "numeric", "Balls in the count."),
    Feature("strikes", "numeric", "Strikes in the count."),
    Feature("outs_when_up", "numeric", "Outs in the half-inning."),
    Feature("inning", "numeric", "Inning number."),
    Feature("base_state", "numeric", "3-bit runner occupancy, 0-7."),
    Feature("base_out_state", "categorical", "Base-out state, 0-23."),
    Feature("score_diff", "numeric", "Batting team score minus fielding team."),
    Feature("is_platoon_same", "numeric", "Batter and pitcher share handedness."),
    Feature("stand", "categorical", "Batter handedness."),
    Feature("p_throws", "categorical", "Pitcher handedness."),
]

FATIGUE: list[Feature] = [
    Feature("pitch_count_in_game", "numeric", "Pitches thrown by this pitcher BEFORE this one."),
    Feature("n_thruorder_pitcher", "numeric", "Times through the order."),
    Feature("pitcher_days_since_prev_game", "numeric", "Days of rest."),
    Feature("pitch_in_ab", "numeric", "Pitch number within the plate appearance."),
]

# Sequence features look strictly backward. `prev_*_1` is the pitch immediately
# before this one in the same plate appearance; null on the first pitch.
SEQUENCE: list[Feature] = [
    Feature("prev_pitch_type_1", "categorical", "Previous pitch type in this PA."),
    Feature("prev_pitch_type_2", "categorical", "Two pitches ago in this PA."),
    Feature("prev_pitch_type_3", "categorical", "Three pitches ago in this PA."),
    Feature("prev_plate_x_1", "numeric", "Previous pitch horizontal location."),
    Feature("prev_plate_z_norm_1", "numeric", "Previous pitch normalized height."),
    Feature("prev_was_swing_1", "numeric", "Batter swung at the previous pitch."),
    Feature("prev_was_whiff_1", "numeric", "Batter missed the previous pitch."),
    Feature("prev_was_in_zone_1", "numeric", "Previous pitch was in the zone."),
    Feature("prev_velo_1", "numeric", "Previous pitch velocity."),
    Feature("last_pitch_type_vs_batter", "categorical", "Last pitch this batter saw this game."),
]

# Priors are expanding-window: computed from pitches strictly BEFORE the current
# one. A full-season usage rate would encode the future into an April pitch.
# The pitch types common enough to deserve their own expanding-window rate.
PRIOR_PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"]

# Per-pitch-type priors matter more than they look. Collapsing an arsenal into
# three family rates throws away most of what identifies a pitcher -- a slider/
# sweeper pitcher and a curveball pitcher have identical "breaking %". The
# count-bucket baseline gets the full per-pitcher distribution, so a model given
# only family rates is strictly less informed than the thing it must beat, and
# will lose to it no matter how good the booster is.
#
# `pitcher` itself is deliberately NOT a feature: with hundreds of ids it would
# memorize rather than generalize, and would be useless for a debut pitcher.
# These rates carry the same information in a form that transfers.
PRIORS: list[Feature] = [
    *[
        Feature(f"prior_usage_{pt.lower()}", "numeric", f"{pt} rate so far this season.")
        for pt in PRIOR_PITCH_TYPES
    ],
    Feature("prior_usage_ff_family", "numeric", "Fastball-family rate so far this season."),
    Feature("prior_usage_breaking", "numeric", "Breaking-ball rate so far this season."),
    Feature("prior_usage_offspeed", "numeric", "Offspeed rate so far this season."),
    Feature("prior_zone_rate", "numeric", "Pitcher's zone rate so far this season."),
    Feature("prior_pitches_seen", "numeric", "Sample size behind the priors."),
    Feature("prior_usage_this_count", "numeric", "Sample size in this count bucket."),
]

CONTEXT: list[Feature] = [
    Feature("home_team", "categorical", "Park proxy."),
    Feature("is_home_pitching", "numeric", "Pitching team is the home team."),
]

ALL_FEATURES: list[Feature] = GAME_STATE + FATIGUE + SEQUENCE + PRIORS + CONTEXT

FEATURE_NAMES: list[str] = [f.name for f in ALL_FEATURES]
CATEGORICAL_FEATURES: list[str] = [f.name for f in ALL_FEATURES if f.kind == "categorical"]
NUMERIC_FEATURES: list[str] = [f.name for f in ALL_FEATURES if f.kind == "numeric"]


# --- leakage guard ----------------------------------------------------------

# Columns that describe the pitch being predicted. If any of these ever appears
# in FEATURE_NAMES the model is reading its own answer; `datasets` asserts on it.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "pitch_type",
        "pitch_name",
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "release_pos_x",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
        "ivb_in",
        "hb_in",
        "hb_arm_in",
        "plate_x",
        "plate_z",
        "plate_z_norm",
        "zone",
        "spin_axis",
        "arm_angle",
        "description",
        "events",
        "type",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "estimated_woba_using_speedangle",
        "estimated_ba_using_speedangle",
        "woba_value",
        "delta_run_exp",
        "delta_home_win_exp",
        "delta_pitcher_run_exp",
        "is_swing",
        "is_whiff",
        "is_called_strike",
        "is_in_play",
        "is_in_zone",
        "is_chase",
        "is_csw",
        "bat_speed",
        "swing_length",
        "attack_angle",
        "vb_gravity_in",
    }
)


def assert_no_leakage(names: list[str]) -> None:
    """Raise if any feature name describes the pitch being predicted."""
    bad = sorted(set(names) & FORBIDDEN_FEATURE_COLUMNS)
    if bad:
        raise ValueError(
            f"Leaking feature(s): {bad}. These describe the pitch being predicted "
            "and cannot be model inputs."
        )
