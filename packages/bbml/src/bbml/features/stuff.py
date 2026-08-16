"""Feature sets for Stuff+ / Location+ / Pitching+ — and the wall between them.

Three models, one target (`run_value.rv_pitcher`), three input sets:

| model      | inputs                                    | question answered                       |
|------------|-------------------------------------------|-----------------------------------------|
| Stuff+     | physical characteristics, no location     | how good is the pitch as an object?     |
| Location+  | location + count, no physical             | was it put in a good place?             |
| Pitching+  | both                                      | how good was the pitch, all in?         |

The entire value of the triple comes from the first two sets being *disjoint*. If
a location column leaks into the stuff set, Stuff+ silently becomes a worse
Pitching+ and the comparison everyone actually wants — "elite shape, poor
command" — stops meaning anything. `assert_sets_are_disjoint` enforces that as a
check rather than as a comment, in the same spirit as `schema.assert_no_leakage`.

Note the leakage rule here is the OPPOSITE of the next-pitch model's. There,
anything describing the pitch itself is forbidden. Here it is the entire point:
the pitch has been thrown and we are grading it. `schema.FORBIDDEN_FEATURE_
COLUMNS` must not be applied to these sets.

WHY `pitch_type` IS IN THE LOCATION SET AND NOT THE STUFF SET
--------------------------------------------------------------
It looks asymmetric and it was measured both ways (train <=2016 / val 2017 /
test 2025, correlation of the grade with observed run value over pitcher x
pitch_type groups of 100+ test pitches):

    stuff                 0.173   ->  stuff + pitch_type      0.164
    location              0.130   ->  location + pitch_type   0.175

Adding it to the stuff set does nothing, which is the expected result: the
physical measurements already say everything the label does, and more precisely.
Adding it to the location set is a real gain, because "well located" is not a
question you can answer without knowing what was being located — a curveball at
the bottom of the zone and a fastball at the bottom of the zone are different
pitches at the same coordinates.

It is not a stuff leak in disguise. `pitch_type` is a bare label: every slider
shares it whether the slider is elite or unusable, so it carries no information
about pitch *quality* for the location model to smuggle in. It says which pitch,
not how good.

HANDEDNESS NORMALIZATION
------------------------
Every directional input is mirrored so that left- and right-handed pitchers (and
batters) land in the same frame. Without it the model spends its capacity
learning that lefties are mirror images instead of learning what a good pitch is,
and every LHP grade is quietly fit on a fifth as much data as it looks like.

  * `hb_arm_in` is already arm-side-positive out of the transform layer.
  * `release_pos_x_arm` gets the same flip — RHP release x averages -1.9, LHP
    +2.1, so raw values put the two hands on opposite ends of one axis.
  * `spin_axis` is a compass bearing, so it is mirrored about 12 o'clock for LHP
    and then encoded as sin/cos. Feeding degrees straight in tells a tree that
    359 and 1 are as far apart as 359 and 1 are close.
  * `plate_x_out` is mirrored on the BATTER's hand, positive = away from him.
    Inside and outside are the units command is actually measured in; left and
    right are not.

FASTBALL DIFFERENTIALS
----------------------
A changeup is graded by its separation from the heater, not by absolute speed —
this is the input that makes stuff models work at all. The primary-fastball
ranking rule is deliberately identical to `sql/marts/mart_pitcher_arsenal.sql`
(FF/SI/FA first, FC only when there is no true fastball), because two different
answers to "what is this pitcher's fastball" in two places is a bug waiting to be
discovered from a graph that disagrees with a table.

The baseline is a whole-season aggregate, which is future information relative to
an April pitch. That is fine and is not the leak it looks like: it is an average
of the pitcher's OWN physical characteristics and contains no outcome, so it
cannot leak the target. It would be a leak in the next-pitch model, which is
forecasting; this one is describing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbml.features.schema import Feature

log = get_logger(__name__)

TARGET_RUN_VALUE = "rv_pitcher"

# Lake columns read for pitch quality work: the physics, the location, the count,
# and everything `run_value` needs to build the target.
LAKE_COLUMNS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
    "balls",
    "strikes",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "ivb_in",
    "hb_arm_in",
    "arm_angle",
    "spin_axis",
    "plate_x",
    "plate_z",
    "plate_z_norm",
    "is_platoon_same",
    "events",
    "description",
    "delta_run_exp",
    "estimated_woba_using_speedangle",
]

STUFF: list[Feature] = [
    Feature("release_speed", "numeric", "Velocity out of hand, mph."),
    Feature("ivb_in", "numeric", "Induced vertical break, inches."),
    Feature("hb_arm_in", "numeric", "Horizontal break, arm-side positive, inches."),
    Feature("release_spin_rate", "numeric", "Spin rate, rpm."),
    Feature("spin_axis_sin", "numeric", "Spin axis, mirrored for LHP, sine component."),
    Feature("spin_axis_cos", "numeric", "Spin axis, mirrored for LHP, cosine component."),
    Feature("release_extension", "numeric", "Feet toward the plate at release."),
    Feature("release_pos_x_arm", "numeric", "Release point, arm-side positive, feet."),
    Feature("release_pos_z", "numeric", "Release height, feet."),
    Feature("arm_angle", "numeric", "Arm slot, degrees. 2025-only in most of the lake."),
    Feature("velo_diff_fb", "numeric", "Velocity minus the pitcher's primary fastball."),
    Feature("ivb_diff_fb", "numeric", "IVB minus the pitcher's primary fastball."),
    Feature("hb_diff_fb", "numeric", "Arm-side break minus the pitcher's primary fastball."),
    Feature("is_platoon_same", "numeric", "Batter and pitcher share handedness."),
]

LOCATION: list[Feature] = [
    Feature("plate_x_out", "numeric", "Horizontal location, positive = away from the batter."),
    Feature("plate_z", "numeric", "Height at the plate, feet."),
    Feature("plate_z_norm", "numeric", "Height scaled to this batter's own zone."),
    Feature("balls", "numeric", "Balls in the count."),
    Feature("strikes", "numeric", "Strikes in the count."),
    Feature("pitch_type", "categorical", "Which pitch is being located."),
    Feature("is_platoon_same", "numeric", "Batter and pitcher share handedness."),
]

# `is_platoon_same` sits in both sets on purpose: arm-side run plays differently
# against an opposite-handed batter, and so does a pitch on the outer edge. It is
# a matchup fact, not a location and not a physical characteristic, so excluding
# it from either model would handicap that model rather than purify it.
_SHARED = {"is_platoon_same"}

FEATURE_SETS: dict[str, list[Feature]] = {
    "stuff": STUFF,
    "location": LOCATION,
    "pitching": STUFF + [f for f in LOCATION if f.name not in {s.name for s in STUFF}],
}

ROLES: tuple[str, ...] = ("stuff", "location", "pitching")


def feature_names(role: str) -> list[str]:
    if role not in FEATURE_SETS:
        raise KeyError(f"Unknown pitch-quality role {role!r}; expected one of {ROLES}.")
    return [f.name for f in FEATURE_SETS[role]]


def categorical_features(role: str) -> list[str]:
    if role not in FEATURE_SETS:
        raise KeyError(f"Unknown pitch-quality role {role!r}; expected one of {ROLES}.")
    return [f.name for f in FEATURE_SETS[role] if f.kind == "categorical"]


def assert_sets_are_disjoint() -> None:
    """Stuff and Location may share only the declared matchup columns."""
    overlap = ({f.name for f in STUFF} & {f.name for f in LOCATION}) - _SHARED
    if overlap:
        raise AssertionError(
            f"{sorted(overlap)} appears in both the stuff and location feature sets. "
            "Stuff+ and Location+ are only meaningful while their inputs are disjoint."
        )


# --- loading -----------------------------------------------------------------


def load_pitch_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Tracked, competitive, regular-season pitches with the physics attached.

    Untracked pitches (pitch-clock and ABS automatic calls) carry no measurements
    at all, and pitchouts/intentional balls are not attempts to get anyone out.
    """
    s = settings or get_settings()
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch") & pl.col("is_competitive") & (pl.col("game_type") == "R")
    )
    if seasons:
        lf = lf.filter(pl.col("season").is_in(seasons))
    return lf.select(LAKE_COLUMNS).collect()


# --- derivation --------------------------------------------------------------

# Ranked exactly as in sql/marts/mart_pitcher_arsenal.sql — see the module
# docstring on why FC is the last resort rather than a fastball.
TRUE_FASTBALLS: list[str] = ["FF", "SI", "FA"]
FASTBALL_CANDIDATES: list[str] = [*TRUE_FASTBALLS, "FC"]

# Below this many pitches a "primary fastball" is one bad afternoon, and the
# differentials it anchors are noise dressed up as a feature.
MIN_FASTBALL_PITCHES = 25


def primary_fastball(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (pitcher, season): the shape of his primary fastball."""
    return (
        df.filter(pl.col("pitch_type").is_in(FASTBALL_CANDIDATES))
        .group_by(["pitcher", "season", "pitch_type"])
        .agg(
            pl.len().alias("n"),
            pl.col("release_speed").mean().alias("fb_velo"),
            pl.col("ivb_in").mean().alias("fb_ivb"),
            pl.col("hb_arm_in").mean().alias("fb_hb"),
        )
        .filter(pl.col("n") >= MIN_FASTBALL_PITCHES)
        .with_columns(pl.col("pitch_type").is_in(TRUE_FASTBALLS).not_().alias("_is_cutter"))
        .sort(["pitcher", "season", "_is_cutter", "n"], descending=[False, False, False, True])
        .unique(subset=["pitcher", "season"], keep="first", maintain_order=True)
        .select("pitcher", "season", "fb_velo", "fb_ivb", "fb_hb")
    )


def add_pitch_quality_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive every column in `FEATURE_SETS` from a raw lake frame."""
    mirror_pitcher = pl.when(pl.col("p_throws") == "R").then(-1.0).otherwise(1.0)
    mirror_batter = pl.when(pl.col("stand") == "R").then(1.0).otherwise(-1.0)

    # Mirror the compass bearing for LHP, then put it on the unit circle so that
    # 359 degrees and 1 degree are neighbours rather than opposites.
    axis = pl.when(pl.col("p_throws") == "R").then(pl.col("spin_axis")).otherwise(
        (360.0 - pl.col("spin_axis")) % 360.0
    ) * (np.pi / 180.0)

    out = df.with_columns(
        (pl.col("release_pos_x") * mirror_pitcher).alias("release_pos_x_arm"),
        axis.sin().alias("spin_axis_sin"),
        axis.cos().alias("spin_axis_cos"),
        (pl.col("plate_x") * mirror_batter).alias("plate_x_out"),
        pl.col("is_platoon_same").cast(pl.Int8),
    )

    fb = primary_fastball(df)
    return (
        out.join(fb, on=["pitcher", "season"], how="left")
        .with_columns(
            (pl.col("release_speed") - pl.col("fb_velo")).alias("velo_diff_fb"),
            (pl.col("ivb_in") - pl.col("fb_ivb")).alias("ivb_diff_fb"),
            (pl.col("hb_arm_in") - pl.col("fb_hb")).alias("hb_diff_fb"),
        )
        .drop("fb_velo", "fb_ivb", "fb_hb")
    )


def build_pitch_quality_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Load + derive, without the target — `RunValue.attach` supplies that."""
    assert_sets_are_disjoint()
    df = load_pitch_frame(seasons=seasons, settings=settings)
    log.info("loaded %d pitches for pitch quality", df.height)
    return add_pitch_quality_features(df)
