"""Features for model #6: whether swinging was the value-maximising choice.

The model sees only information available as the pitch reaches the plate.  It
learns the expected batting run value of a *swing* from observed swings; the
alternative, taking, is supplied by the calibrated called-strike model.  Batter
identity is intentionally absent: a decision grade must not teach the model to
expect a particular hitter to make his usual choices.
"""

from __future__ import annotations

import polars as pl

from bbcore.config import Settings, get_settings
from bbml.features.schema import Feature
from bbml.features.stuff import load_pitch_frame

TARGET_SWING = "is_swing"
TARGET_VALUE = "rv_batter"

FEATURES: list[Feature] = [
    Feature("plate_x_out", "numeric", "Horizontal location, positive = away from batter."),
    Feature("plate_z", "numeric", "Height at the plate, feet."),
    Feature("plate_z_norm", "numeric", "Height relative to batter zone."),
    Feature("balls", "numeric", "Balls in count."),
    Feature("strikes", "numeric", "Strikes in count."),
    Feature("pitch_type", "categorical", "Pitch type."),
    Feature("release_speed", "numeric", "Velocity, mph."),
    Feature("ivb_in", "numeric", "Induced vertical break, inches."),
    Feature("hb_arm_in", "numeric", "Arm-side break, inches."),
]
FEATURE_NAMES = [f.name for f in FEATURES]
CATEGORICAL_FEATURES = [f.name for f in FEATURES if f.kind == "categorical"]


def add_swing_decision_features(df: pl.DataFrame) -> pl.DataFrame:
    mirror = pl.when(pl.col("stand") == "R").then(1.0).otherwise(-1.0)
    return df.with_columns((pl.col("plate_x") * mirror).alias("plate_x_out"))


def build_swing_decision_frame(
    *, seasons: list[int] | None = None, settings: Settings | None = None
) -> pl.DataFrame:
    """All competitive pitches, not just swings: every row is a decision."""
    # `load_pitch_frame` already carries both the RunValue columns and the
    # pre-decision pitch controls. `is_swing` is not part of its public frame,
    # so read it from the lake and join on the natural pitch key.
    s = settings or get_settings()
    base = load_pitch_frame(seasons=seasons, settings=s)
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    choices = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch") & pl.col("is_competitive") & (pl.col("game_type") == "R")
    )
    if seasons:
        choices = choices.filter(pl.col("season").is_in(seasons))
    choices = choices.select(
        "game_pk", "at_bat_number", "pitch_number", pl.col("is_swing").cast(pl.Boolean)
    )
    return add_swing_decision_features(
        base.join(choices.collect(), on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    )
