"""Called-strike features: what the umpire saw, deliberately without who was watching.

TARGET AND POPULATION
----------------------
`is_called_strike` on TAKES — pitches with no swing offered, so the outcome is
entirely the umpire's call. `TAKE_DESCRIPTIONS` is `ball` / `called_strike` /
`blocked_ball`; a hit-by-pitch is deliberately excluded even though it is not a
swing, because the pitch hit the batter rather than being judged ball or
strike — it is not a take in the sense this model cares about.

WHY CATCHER AND UMPIRE ARE NOT FEATURES
-----------------------------------------
Same principle as keeping `pitcher` out of the next-pitch model (`schema.py`)
and pitch type/velocity out of Stuff+'s stuff feature set (`stuff.py`): the
entity a downstream metric is meant to grade cannot be an input to the model
doing the grading, or the model absorbs the very effect the metric exists to
isolate. If catcher were a feature, a good framer's receiving would train the
model to expect more strikes from him specifically, and `framing_runs`
(`models/called_strike.py`) would then compare his outcomes to a baseline that
already contains his own skill — his measured framing value would collapse
toward zero exactly because he is good at it. `CATCHER_COLUMN` and
`UMPIRE_COLUMN` are carried on the frame as passthrough columns, grouped on
*after* scoring, never fed to the booster.

FEATURES
--------
The physical pitch as it crossed the plate, mirrored the same way `stuff.py`
mirrors: `plate_x_out` positive = away from the batter, `plate_z`/
`plate_z_norm` for absolute and batter-zone-relative height, movement
(`ivb_in`/`hb_arm_in`), velocity, and count. `pitch_type` rides along as a bare
label rather than a stuff signal — the same reasoning `stuff.py` gives for
putting it in the Location+ set: a breaking ball's late, deceptive path is part
of what the umpire had to judge at a given location, a fastball's is not.
"""

from __future__ import annotations

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbml.features.schema import Feature

log = get_logger(__name__)

TARGET_CALLED_STRIKE = "is_called_strike"

TAKE_DESCRIPTIONS: frozenset[str] = frozenset({"ball", "called_strike", "blocked_ball"})

# Never fed to the booster — see the module docstring. `fielder_2` is
# Statcast's catcher column; `umpire_hp` is joined in from `dim_official`,
# since Statcast's own `umpire` column is empty.
CATCHER_COLUMN = "fielder_2"
UMPIRE_COLUMN = "umpire_hp"

LAKE_COLUMNS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "pitcher",
    "batter",
    CATCHER_COLUMN,
    "stand",
    "p_throws",
    "pitch_type",
    "balls",
    "strikes",
    "release_speed",
    "ivb_in",
    "hb_arm_in",
    "plate_x",
    "plate_z",
    "plate_z_norm",
    TARGET_CALLED_STRIKE,
]

FEATURES: list[Feature] = [
    Feature("plate_x_out", "numeric", "Horizontal location, positive = away from the batter."),
    Feature("plate_z", "numeric", "Height at the plate, feet."),
    Feature("plate_z_norm", "numeric", "Height scaled to this batter's own zone."),
    Feature("balls", "numeric", "Balls in the count."),
    Feature("strikes", "numeric", "Strikes in the count."),
    Feature("pitch_type", "categorical", "Which pitch was thrown."),
    Feature("release_speed", "numeric", "Velocity out of hand, mph."),
    Feature("ivb_in", "numeric", "Induced vertical break, inches."),
    Feature("hb_arm_in", "numeric", "Horizontal break, arm-side positive, inches."),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.kind == "categorical"]


def load_called_strike_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Tracked, regular-season TAKES, with the home-plate umpire joined on.

    No `is_competitive` filter: `TAKE_DESCRIPTIONS` already excludes pitchouts
    and intentional balls (their description is `pitchout`/`intent_ball`, never
    `ball`/`called_strike`/`blocked_ball`), so the filter would be redundant.
    """
    s = settings or get_settings()
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch")
        & (pl.col("game_type") == "R")
        & pl.col("description").is_in(list(TAKE_DESCRIPTIONS))
    )
    if seasons:
        lf = lf.filter(pl.col("season").is_in(seasons))
    df = lf.select(LAKE_COLUMNS).collect()

    officials_path = s.lake_dir / "dim_official" / "part_0.parquet"
    if officials_path.exists():
        officials = pl.read_parquet(officials_path).select(
            "game_pk", pl.col("umpire_home_plate_id").alias(UMPIRE_COLUMN)
        )
        df = df.join(officials, on="game_pk", how="left")
    else:
        log.warning(
            "dim_official not built — %s will be null. Run `bb build officials`.",
            UMPIRE_COLUMN,
        )
        df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias(UMPIRE_COLUMN))
    return df


def add_called_strike_features(df: pl.DataFrame) -> pl.DataFrame:
    mirror_batter = pl.when(pl.col("stand") == "R").then(1.0).otherwise(-1.0)
    return df.with_columns(
        (pl.col("plate_x") * mirror_batter).alias("plate_x_out"),
        pl.col(TARGET_CALLED_STRIKE).cast(pl.Int8),
    )


def build_called_strike_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    df = load_called_strike_frame(seasons=seasons, settings=settings)
    log.info("loaded %d taken pitches", df.height)
    if df.height == 0:
        raise ValueError("No taken pitches found — rebuild the lake with `bb build pitches`.")
    return add_called_strike_features(df)
