"""Swing-path features: the bat's plane, the pitch's plane, and the gap between them.

THE FINDING THIS IS BUILT ON
----------------------------
Measured on 1.04M tracked swings (2023H2-2026), whiff rate by swing plane
(rows) against pitch descent angle (columns):

                flat pitch    mid     steep pitch
    aa  -7 deg     24.2%      13.4%      18.7%
    aa  +3 deg     22.0%      10.2%      15.0%
    aa  +9 deg     19.3%       9.6%      16.9%
    aa +15 deg     15.7%      12.3%      27.1%
    aa +24 deg     11.6%      29.4%      57.8%

That is an interaction, not a main effect. A steep uppercut whiffs 11.6% against
flat pitches and 57.8% against steep ones — a 5x swing — while a flat swing does
the opposite. **A swing plane is not good or bad in itself; it is good or bad
against a particular pitch.** The same interaction shows up in contact quality
(the +24 deg swing runs .412 xwOBA against flat pitches and .327 against steep)
but far more weakly, so most of the effect is in whether contact happens at all.

Two things follow for the feature design.

WHY THE PITCH PLANE HAS TO BE RECONSTRUCTED, NOT LOOKED UP
-----------------------------------------------------------
Savant ships no approach angle. `vaa_deg`/`haa_deg` come out of the transform
layer (`bbetl.transforms.statcast`), solved from the 9-parameter fit at the
plate rather than read off the y=50 reference — see that module for why the
shortcut flattens every pitch.

WHY LOCATION AND PITCH TYPE ARE IN THE FEATURE SET
---------------------------------------------------
They are controls, not signal, and leaving them out would have made the headline
result an artifact. VAA correlates hard with both: a steep approach angle usually
*is* a low curveball. Without `plate_z_norm` and `pitch_type` in the model, "steep
uppercuts miss steep pitches" would be indistinguishable from "uppercuts miss low
breaking balls", which is a statement about location and pitch selection rather
than about swing geometry. `swing_path.SwingPathModel.plane_value` then holds
those controls fixed and varies only the attack angle, which is what isolates the
geometry the metric claims to measure.
"""

from __future__ import annotations

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbml.features.schema import Feature

log = get_logger(__name__)

TARGET_WHIFF = "is_whiff"
TARGET_CONTACT = "estimated_woba_using_speedangle"

# Swing tracking rolled out mid-2023 (zero before July, 62% that month, ~95%
# after) and the distributions have been stable since — attack angle mean
# 8.2-8.5 deg, bat speed 70.9-71.3mph across all four seasons. So 2023 is usable
# despite `STATUS.md` having long claimed swing path was 2025+; Savant backfilled
# it. That correction is worth ~4x the training data.
FIRST_SWING_TRACKING_SEASON = 2023

LAKE_COLUMNS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "pitch_type",
    "balls",
    "strikes",
    "release_speed",
    "vaa_deg",
    "haa_deg",
    "plate_x",
    "plate_z",
    "plate_z_norm",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "bat_speed",
    "swing_length",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "is_swing",
    "is_whiff",
    "is_in_play",
    "estimated_woba_using_speedangle",
]

# What the batter did. `attack_angle` is the one the counterfactual varies.
SWING: list[Feature] = [
    Feature("attack_angle", "numeric", "Swing plane, degrees, positive is upward."),
    Feature("swing_path_tilt", "numeric", "Tilt of the swing plane, degrees."),
    Feature("attack_direction_pull", "numeric", "Swing direction, positive = pull side."),
    Feature("bat_speed", "numeric", "Bat speed at contact point, mph."),
    Feature("swing_length", "numeric", "Path length of the barrel, feet."),
    Feature("intercept_x_pull", "numeric", "Contact point across the plate, pull-positive."),
    Feature("intercept_y_in", "numeric", "Contact point depth relative to the batter, inches."),
]

# What was thrown. Controls for the confound described in the module docstring.
PITCH: list[Feature] = [
    Feature("vaa_deg", "numeric", "Pitch vertical approach angle at the plate."),
    Feature("haa_deg", "numeric", "Pitch horizontal approach angle at the plate."),
    Feature("release_speed", "numeric", "Velocity out of hand, mph."),
    Feature("plate_x_pull", "numeric", "Horizontal location, pull-side positive."),
    Feature("plate_z_norm", "numeric", "Height scaled to this batter's own zone."),
    Feature("pitch_type", "categorical", "Which pitch was thrown."),
    Feature("balls", "numeric", "Balls in the count."),
    Feature("strikes", "numeric", "Strikes in the count."),
]

ALL_FEATURES: list[Feature] = SWING + PITCH
FEATURE_NAMES: list[str] = [f.name for f in ALL_FEATURES]
CATEGORICAL_FEATURES: list[str] = [f.name for f in ALL_FEATURES if f.kind == "categorical"]

# The feature the counterfactual perturbs. Named rather than inlined so the model
# and the metric cannot drift apart about which column "swing plane" means.
PLANE_FEATURE = "attack_angle"

# The rest of the swing-mechanic block. A real hitter cannot dial attack_angle
# down to league-median while keeping his own bat speed, tilt and contact point
# fixed — that combination barely exists in the data, so a counterfactual that
# freezes them there is extrapolating into empty space and its sign is not to be
# trusted (confirmed on real data: freezing gives plane_value < 0 on a slice
# where the raw whiff-rate gap says it should be strongly positive). SwingPathModel
# instead re-matches these to their typical value at the reference attack_angle.
CORRELATED_SWING_FEATURES: list[str] = [f.name for f in SWING if f.name != PLANE_FEATURE]


def load_swing_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Tracked, competitive, regular-season SWINGS with bat tracking attached."""
    s = settings or get_settings()
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch")
        & pl.col("is_competitive")
        & (pl.col("game_type") == "R")
        & pl.col("is_swing")
        & pl.col("attack_angle").is_not_null()
        & pl.col("vaa_deg").is_not_null()
    )
    lf = lf.filter(
        pl.col("season").is_in(seasons)
        if seasons
        else pl.col("season") >= FIRST_SWING_TRACKING_SEASON
    )
    return lf.select(LAKE_COLUMNS).collect()


def add_swing_features(df: pl.DataFrame) -> pl.DataFrame:
    """Mirror every left/right quantity onto the batter's own frame.

    Pull and oppo are the units a swing is actually described in; left and right
    are not. Without this a left-handed pull is filed alongside a right-handed
    oppo, and the model spends its capacity rediscovering that hitters come in
    two mirror images.
    """
    pull = pl.when(pl.col("stand") == "R").then(-1.0).otherwise(1.0)
    return df.with_columns(
        (pl.col("plate_x") * pull).alias("plate_x_pull"),
        (pl.col("attack_direction") * pull).alias("attack_direction_pull"),
        (pl.col("intercept_ball_minus_batter_pos_x_inches") * pull).alias("intercept_x_pull"),
        pl.col("intercept_ball_minus_batter_pos_y_inches").alias("intercept_y_in"),
        pl.col("is_whiff").cast(pl.Int8),
    )


def build_swing_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    df = load_swing_frame(seasons=seasons, settings=settings)
    log.info("loaded %d tracked swings", df.height)
    if df.height == 0:
        raise ValueError(
            "No swings with bat tracking. Either the lake predates 2023H2 or "
            "`vaa_deg` is missing — rebuild with `bb build pitches`."
        )
    return add_swing_features(df)
