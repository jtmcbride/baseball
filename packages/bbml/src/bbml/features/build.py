"""The feature builder. ONE implementation, two callers.

HOW THE SHARING ACTUALLY WORKS
------------------------------
The naive approach — a scalar function for live inference and a vectorized one for
training — guarantees drift. The two implementations start identical and diverge
on the first bug fix applied to only one of them, and the symptom is a model that
quietly degrades in production with no failing test anywhere.

So there is one function, `build_features`, and it takes a *chronologically
ordered frame of pitches*. The difference between the callers is only which
frame they hand it:

  batch : the whole lake, millions of rows -> features for every row
  live  : this pitcher's pitches so far, plus one synthetic row describing the
          state the next pitch will be thrown into -> `.tail(1)`

Window functions (sequence, priors, fatigue) therefore run through the exact same
polars expressions in both paths. Live frames are a few hundred rows, so the
window work is sub-millisecond — far inside a 5s poll budget.

WHY THE SYNTHETIC ROW PROVES THE NO-LEAKAGE PROPERTY
----------------------------------------------------
The live caller cannot fill in the current pitch's own columns — velocity,
location, result do not exist until the pitch is thrown, so they are null. Every
one of those columns is used *only* through `.shift()`, to describe PREVIOUS
pitches. If a feature ever started reading the current row's pitch data, it would
come out null in the live path and the parity test would fail immediately. The
architecture makes leakage a test failure rather than a silent accuracy illusion.
"""

from __future__ import annotations

import polars as pl

from bbml.features.schema import (
    FEATURE_NAMES,
    LOC_FAR_MISS_CLASS,
    LOC_GRID_N,
    LOC_X_MAX,
    LOC_X_MIN,
    LOC_Z_MAX,
    LOC_Z_MIN,
    PITCH_TYPES,
    PRIOR_PITCH_TYPES,
    TARGET_LOCATION,
    TARGET_PITCH_TYPE,
    assert_no_leakage,
)

# Chronological order. Every window function below depends on the frame being
# sorted this way, so `build_features` sorts rather than trusting the caller.
ORDER_COLS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# Columns the builder reads. The pitch-describing ones are used ONLY via shift().
REQUIRED_COLUMNS = [
    *ORDER_COLS,
    "season",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "base_state",
    "bat_score",
    "fld_score",
    "stand",
    "p_throws",
    "home_team",
    "inning_topbot",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    # backward-looking only:
    "pitch_type",
    "plate_x",
    "plate_z_norm",
    "release_speed",
    "is_swing",
    "is_whiff",
    "is_in_zone",
]

FASTBALLS = ["FF", "SI", "FC", "FA"]
BREAKING = ["SL", "ST", "CU", "KC", "SV", "CS", "SC"]
OFFSPEED = ["CH", "FS", "KN", "EP"]

_PA = ["game_pk", "at_bat_number"]
_PITCHER_GAME = ["game_pk", "pitcher"]
_PITCHER_SEASON = ["pitcher", "season"]
_PITCHER_SEASON_COUNT = ["pitcher", "season", "balls", "strikes"]
_BATTER_GAME = ["game_pk", "batter"]


def _prior_sum(expr: pl.Expr, over: list[str]) -> pl.Expr:
    """Cumulative sum of everything STRICTLY BEFORE the current row.

    `cum_sum() - current` rather than `cum_sum().shift(1)`: it needs no null
    handling at group boundaries and states the intent — "the total before this
    pitch" — directly. Getting this wrong by one row is the single easiest way to
    leak the label into an expanding-window feature.

    The `fill_null(0)` is load-bearing, not defensive. Without it the subtraction
    returns null whenever the CURRENT row's indicator is null — which is always
    true in live inference, where the pending pitch has no pitch type yet. That
    made every prior null at serve time while batch computed them fine: a
    train/serve split invisible to any metric, caught only by the parity test.
    A count of prior events cannot depend on the current row's value.
    """
    counted = expr.fill_null(0)
    return (counted.cum_sum() - counted).over(over)


def build_features(df: pl.DataFrame, *, with_targets: bool = True) -> pl.DataFrame:
    """Compute model features for every row of a chronologically ordered frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"build_features missing required columns: {missing}")
    if df.height == 0:
        return df

    df = df.sort(ORDER_COLS)

    # --- family indicators, used by the expanding-window priors ---
    df = df.with_columns(
        pl.col("pitch_type").is_in(FASTBALLS).cast(pl.Int32).alias("_is_ff"),
        pl.col("pitch_type").is_in(BREAKING).cast(pl.Int32).alias("_is_br"),
        pl.col("pitch_type").is_in(OFFSPEED).cast(pl.Int32).alias("_is_off"),
        pl.col("is_in_zone").cast(pl.Int32).alias("_in_zone"),
        pl.lit(1, dtype=pl.Int32).alias("_one"),
    )

    df = df.with_columns(
        # --- game state ---
        (pl.col("bat_score") - pl.col("fld_score")).alias("score_diff"),
        (pl.col("stand") == pl.col("p_throws")).cast(pl.Int8).alias("is_platoon_same"),
        (pl.col("base_state") * 3 + pl.col("outs_when_up")).cast(pl.Int16).alias("base_out_state"),
        (pl.col("inning_topbot") == "Top").cast(pl.Int8).alias("is_home_pitching"),
        pl.col("pitch_number").alias("pitch_in_ab"),
        # --- fatigue: pitches thrown by this pitcher earlier in this game ---
        pl.int_range(pl.len()).over(_PITCHER_GAME).cast(pl.Int32).alias("pitch_count_in_game"),
        # --- sequence within the plate appearance ---
        pl.col("pitch_type").shift(1).over(_PA).alias("prev_pitch_type_1"),
        pl.col("pitch_type").shift(2).over(_PA).alias("prev_pitch_type_2"),
        pl.col("pitch_type").shift(3).over(_PA).alias("prev_pitch_type_3"),
        pl.col("plate_x").shift(1).over(_PA).alias("prev_plate_x_1"),
        pl.col("plate_z_norm").shift(1).over(_PA).alias("prev_plate_z_norm_1"),
        pl.col("release_speed").shift(1).over(_PA).alias("prev_velo_1"),
        pl.col("is_swing").shift(1).over(_PA).cast(pl.Int8).alias("prev_was_swing_1"),
        pl.col("is_whiff").shift(1).over(_PA).cast(pl.Int8).alias("prev_was_whiff_1"),
        pl.col("is_in_zone").shift(1).over(_PA).cast(pl.Int8).alias("prev_was_in_zone_1"),
        # Crosses plate appearances: what this batter last saw from anyone today.
        pl.col("pitch_type").shift(1).over(_BATTER_GAME).alias("last_pitch_type_vs_batter"),
        # --- expanding-window priors, strictly backward ---
        _prior_sum(pl.col("_one"), _PITCHER_SEASON).alias("prior_pitches_seen"),
        _prior_sum(pl.col("_is_ff"), _PITCHER_SEASON).alias("_prior_ff"),
        _prior_sum(pl.col("_is_br"), _PITCHER_SEASON).alias("_prior_br"),
        _prior_sum(pl.col("_is_off"), _PITCHER_SEASON).alias("_prior_off"),
        _prior_sum(pl.col("_in_zone"), _PITCHER_SEASON).alias("_prior_zone"),
        # Per-pitch-type rates, and the same rates conditioned on the count
        # bucket. Without these the model is strictly less informed about who is
        # pitching than the count-bucket baseline it has to beat.
        *[
            _prior_sum((pl.col("pitch_type") == pt).cast(pl.Int32), _PITCHER_SEASON).alias(
                f"_prior_pt_{pt.lower()}"
            )
            for pt in PRIOR_PITCH_TYPES
        ],
        _prior_sum(pl.col("_one"), _PITCHER_SEASON_COUNT).alias("prior_usage_this_count"),
    )

    # Rates are null (not zero) until there is history. Zero would assert "this
    # pitcher throws no fastballs" on his first pitch of the season, which is a
    # different and false claim; the model handles nulls natively.
    denom = pl.when(pl.col("prior_pitches_seen") > 0).then(pl.col("prior_pitches_seen"))
    df = df.with_columns(
        (pl.col("_prior_ff") / denom).alias("prior_usage_ff_family"),
        (pl.col("_prior_br") / denom).alias("prior_usage_breaking"),
        (pl.col("_prior_off") / denom).alias("prior_usage_offspeed"),
        (pl.col("_prior_zone") / denom).alias("prior_zone_rate"),
        *[
            (pl.col(f"_prior_pt_{pt.lower()}") / denom).alias(f"prior_usage_{pt.lower()}")
            for pt in PRIOR_PITCH_TYPES
        ],
    )

    if with_targets:
        df = df.with_columns(
            pl.col("pitch_type").alias(TARGET_PITCH_TYPE),
            location_class_expr().alias(TARGET_LOCATION),
        )

    return df.drop([c for c in df.columns if c.startswith("_")])


def location_class_expr() -> pl.Expr:
    """Bucket (plate_x, plate_z_norm) into the 26-class location target.

    Classes 0-24 are a 5x5 grid over the zone plus its immediate surroundings;
    class 25 is everything further out. Keeping the grid wider than the rulebook
    zone preserves *which way* a pitcher missed, which is most of the signal in
    a two-strike count.
    """
    col = (
        ((pl.col("plate_x") - LOC_X_MIN) / (LOC_X_MAX - LOC_X_MIN) * LOC_GRID_N)
        .floor()
        .cast(pl.Int32)
    )
    row = (
        ((pl.col("plate_z_norm") - LOC_Z_MIN) / (LOC_Z_MAX - LOC_Z_MIN) * LOC_GRID_N)
        .floor()
        .cast(pl.Int32)
    )
    inside = (
        (col >= 0)
        & (col < LOC_GRID_N)
        & (row >= 0)
        & (row < LOC_GRID_N)
        & pl.col("plate_x").is_not_null()
        & pl.col("plate_z_norm").is_not_null()
    )
    return (
        pl.when(pl.col("plate_x").is_null() | pl.col("plate_z_norm").is_null())
        .then(None)
        .when(inside)
        .then(row * LOC_GRID_N + col)
        .otherwise(LOC_FAR_MISS_CLASS)
        .cast(pl.Int32)
    )


def feature_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Select just the model inputs, in the schema's canonical order.

    Checks for leakage on every call rather than once at import: this is the last
    gate before data reaches a model, and it is cheap.
    """
    assert_no_leakage(FEATURE_NAMES)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"feature_matrix missing: {missing}. Run build_features first.")
    return df.select(FEATURE_NAMES)


def encode_pitch_type(df: pl.DataFrame, col: str = TARGET_PITCH_TYPE) -> pl.Series:
    """Map pitch-type strings to stable integer labels; unknown/null -> null."""
    mapping = {pt: i for i, pt in enumerate(PITCH_TYPES)}
    return df[col].replace_strict(mapping, default=None, return_dtype=pl.Int32)
