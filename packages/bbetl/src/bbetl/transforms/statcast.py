"""raw Statcast CSV -> typed, enriched `fact_pitch` Parquet.

Derived columns are computed here, once, rather than in every query. Two of them
carry judgment worth recording:

`hb_arm_in` — horizontal break, arm-side positive. Savant ships
`api_break_x_arm`, which is exactly `pfx_x` with the handedness flip already
applied (verified against live data back to 2015), so we use it and fall back to
computing the flip ourselves when it is null. Without this normalization, LHP and
RHP movement plots mirror each other and every cross-handedness comparison is
silently wrong.

`plate_z_norm` — vertical location scaled to the batter's own strike zone. Raw
`plate_z` is not comparable between a 5'6" and a 6'7" hitter, and Savant's `zone`
field (1-9, 11-14) is far too coarse for a real heatmap. Normalizing is what makes
hot/cold maps mean anything across players.

`vaa_deg` / `haa_deg` — the pitch's approach angles where it crosses the plate.
Savant does not ship these; they have to be reconstructed from the 9-parameter
physics fit, and they are what makes the swing-path work possible at all — a
swing plane is only good or bad relative to the plane of the pitch it meets.

`x_ft` / `y_ft` — batted-ball landing spot in feet from home plate (standard
orientation: home plate at the origin, straight-away CF along +y), converted
from Savant's raw Gameday pixel coordinates `hc_x`/`hc_y`. The origin matches
the community-published constant; the scale is this project's own fit against
real `hit_distance_sc` data — see the constants below for the measurement.
Unlike `hb_arm_in`, this is
NOT mirrored by batter handedness: a spray chart shows absolute field position,
not swing-relative direction.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.sources.savant import SavantClient

log = get_logger(__name__)

LAKE_TABLE = "fact_pitch"

# --- outcome vocabulary (verified against live 2025 data) --------------------

SWING_DESCRIPTIONS = frozenset(
    {
        "foul",
        "hit_into_play",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul_tip",
        "foul_bunt",
        "missed_bunt",
        "bunt_foul_tip",
    }
)
# foul_tip is contact, not a whiff — the bat reached the ball. Including it is a
# common error that inflates whiff rate by ~1pp.
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
CALLED_STRIKE_DESCRIPTIONS = frozenset({"called_strike"})

# Pitch-clock violations and ABS-awarded calls. No pitch was thrown, so they carry
# no tracking data. They must not reach a model or a movement plot, but they do
# affect count state, so they stay in the table behind `is_tracked_pitch`.
NON_PITCH_DESCRIPTIONS = frozenset({"automatic_ball", "automatic_strike"})

# Non-competitive pitches: intentional balls and pitchouts distort usage rates.
#
# `intent_ball` was missing here until 2026-08-16 despite the comment above
# naming it — a real bug, not a judgment call. It let ~3,700 intentional balls
# into every competitive-filtered model, mart, and zone grid, at plate_x values
# of 4-11 feet because the catcher stands up and the pitcher lobs it. Those are
# genuine locations and exactly the ones no command metric should be graded on.
NON_COMPETITIVE_DESCRIPTIONS = frozenset({"pitchout", "intent_ball"})

# --- physically impossible tracking -----------------------------------------
#
# Bounds chosen to catch corruption and nothing else. Across the full 2015-2026
# lake (9.2M pitches) exactly 11 rows fall outside them, and every one is
# unambiguous garbage: release points below ground level or 11ft in the air,
# `hit_into_play` recorded at plate_x = 25ft, a plate_z of -57ft.
#
# They are deliberately far looser than the *plausible* range. An intentional
# ball really does cross 11 feet wide of the plate and a position player really
# does lob 30mph; the job here is rejecting impossibility, not unusualness.
# `quality.py` owns the tighter, warn-level plausibility checks.
MIN_RELEASE_SPEED, MAX_RELEASE_SPEED = 25.0, 108.0
MAX_PLATE_X = 15.0
MIN_PLATE_Z, MAX_PLATE_Z = -10.0, 20.0
MIN_RELEASE_Z, MAX_RELEASE_Z = 0.0, 10.0
MAX_RELEASE_X = 8.0


def _impossible_tracking_expr() -> pl.Expr:
    """True when a measurement is outside anything a real pitch can produce.

    A NULL measurement is missing, not impossible — older seasons have whole
    columns unpopulated and must not be quarantined for it.

    Columns are cast before comparison: a day file whose measurement column is
    null for every row reads back as Null dtype, and comparing that raises
    rather than returning false. `SCHEMA_OVERRIDES` normally prevents it (see
    the all-null-infers-as-String bug it was added for), but a guard that
    crashes on absent data is the wrong failure mode for a guard.
    """

    def num(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Float64)

    checks = [
        ~num("release_speed").is_between(MIN_RELEASE_SPEED, MAX_RELEASE_SPEED),
        num("plate_x").abs() > MAX_PLATE_X,
        ~num("plate_z").is_between(MIN_PLATE_Z, MAX_PLATE_Z),
        ~num("release_pos_z").is_between(MIN_RELEASE_Z, MAX_RELEASE_Z, closed="right"),
        num("release_pos_x").abs() > MAX_RELEASE_X,
    ]
    return pl.any_horizontal([c.fill_null(False) for c in checks])


# --- approach angles ---------------------------------------------------------
#
# THESE TWO CONSTANTS ARE SHARED WITH THE 3D TRAJECTORY VIZ and are validated,
# not chosen. `vx0..az` describe a constant-acceleration fit that is valid at
# Statcast's fixed y=50ft reference, NOT at the release point; `plate_x`/`plate_z`
# are measured where the ball crosses the FRONT of the plate at y=17/12, not at
# y=0. Both were confirmed against 200 real pitches to ~0.003ft mean error.
#
# They now live in three places — here, `apps/web/src/lib/trajectory.ts`, and the
# `TestTrajectory` regression test in `apps/api/tests/test_api.py`. Changing one
# without the others silently produces a different geometry in each. There is a
# test on each side pinning them against the same real pitch; keep it that way.
Y0_REF = 50.0
PLATE_Y = 17 / 12


def _approach_angle_exprs() -> list[pl.Expr]:
    """Vertical and horizontal approach angle at the plate, in degrees.

    VAA is negative: the ball is descending. It is steeper for slow breaking
    balls (~-9.5 deg for a curveball) and flattest for four-seamers (~-4.7),
    which is the sanity check to run if this ever looks wrong.

    Solved rather than approximated. Taking `atan2(vz0, |vy0|)` straight off the
    y=50 values would report the angle 48 feet in front of the plate, before
    gravity has done most of its work, and would flatten every pitch.
    """
    # A zero `ay` would be a division by zero, and a negative discriminant means
    # the fitted trajectory never reaches the plate. Both are corrupt fits, and
    # both should come back null rather than infinite.
    ay_safe = pl.when(pl.col("ay") == 0).then(None).otherwise(pl.col("ay"))
    disc = pl.col("vy0") ** 2 - 2 * pl.col("ay") * (Y0_REF - PLATE_Y)
    # The smaller root: the first time the ball reaches the plate, not the
    # second one the parabola offers on its way back.
    t = pl.when(disc >= 0).then((-pl.col("vy0") - disc.sqrt()) / ay_safe).otherwise(None)

    vx = pl.col("vx0") + pl.col("ax") * t
    vy = pl.col("vy0") + pl.col("ay") * t
    vz = pl.col("vz0") + pl.col("az") * t
    return [
        pl.arctan2(vz, vy.abs()).degrees().alias("vaa_deg"),
        pl.arctan2(vx, vy.abs()).degrees().alias("haa_deg"),
    ]


# --- batted-ball hit coordinates ---------------------------------------------
#
# Gameday-pixel-to-feet conversion. The widely-published community constants
# (home plate at pixel (125.42, 198.27), 2.495 px/ft) got the ORIGIN right but
# not the scale, measured against this project's own real data: a
# least-squares fit of `k * hypot(hc_x-x0, hc_y-y0)` against Savant's own
# `hit_distance_sc` over 158,098 real 2023-2024 balls in play (>100ft, to
# exclude the noisiest short-hop grounders) converged to x0=125.91, y0=199.54 —
# a few tenths of a foot from the published origin, i.e. confirms it — but
# k=2.339, not 2.495 (fixing the published origin and fitting k alone gives
# 2.307, so the discrepancy is in the scale, not a mislocated origin).
#
# Even at the best-fit scale, MAE against `hit_distance_sc` is ~28ft with
# r=0.886 — because `hc_x`/`hc_y` is a charted fielding location, not a
# trajectory endpoint, this is the inherent precision of the public Gameday
# coordinate, not a bug in this transform. Validated for what this feature
# needs: no systematic offset (the fitted origin lands within a foot of the
# published one) and no mirrored sign (pull-side means confirmed opposite for
# LHB vs. RHB — see `TestHitCoordinates` in `test_transforms.py`, and
# `HISTORY.md` for the full measurement). Good for spray direction and rough
# field position; do not use `x_ft`/`y_ft` where the precise landing spot
# matters more than the general area — `hit_distance_sc` is the batted ball's
# actual measured distance and should be preferred whenever precision matters.
HC_X0 = 125.91
HC_Y0 = 199.54
HC_PIXELS_PER_FOOT = 2.339


def _hit_coordinate_exprs() -> list[pl.Expr]:
    x_ft = (pl.col("hc_x") - HC_X0) * HC_PIXELS_PER_FOOT
    y_ft = (HC_Y0 - pl.col("hc_y")) * HC_PIXELS_PER_FOOT
    return [
        x_ft.alias("x_ft"),
        y_ft.alias("y_ft"),
        pl.arctan2(x_ft, y_ft).degrees().alias("spray_angle_deg"),
        (x_ft**2 + y_ft**2).sqrt().alias("hit_distance_derived_ft"),
    ]


def _base_state_expr() -> pl.Expr:
    """3-bit occupancy code, 0-7. Runner columns hold the runner's id or null."""
    return (
        pl.col("on_1b").is_not_null().cast(pl.Int8)
        + pl.col("on_2b").is_not_null().cast(pl.Int8) * 2
        + pl.col("on_3b").is_not_null().cast(pl.Int8) * 4
    ).alias("base_state")


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived columns. Pure — no I/O — so it is directly unit-testable."""
    if df.height == 0:
        return df

    # game_date arrives as a string from raw CSV but as a Date when re-read from
    # Parquet, so branch rather than cast unconditionally.
    if df.schema["game_date"] == pl.String:
        df = df.with_columns(pl.col("game_date").str.to_date("%Y-%m-%d", strict=False))

    df = df.with_columns(
        # --- identity / partitioning ---
        pl.col("game_date").dt.year().cast(pl.Int16).alias("season"),
        pl.col("game_date").dt.month().cast(pl.Int8).alias("game_month"),
        # --- movement, in inches ---
        (pl.col("pfx_z") * 12).alias("ivb_in"),
        (pl.col("pfx_x") * 12).alias("hb_in"),
        pl.coalesce(
            pl.col("api_break_x_arm") * 12,
            # Fallback: flip pfx_x so positive is always arm-side.
            pl.when(pl.col("p_throws") == "R")
            .then(-pl.col("pfx_x") * 12)
            .otherwise(pl.col("pfx_x") * 12),
        ).alias("hb_arm_in"),
        (pl.col("api_break_z_with_gravity") * 12).alias("vb_gravity_in"),
        # --- approach angles at the plate ---
        *_approach_angle_exprs(),
        # --- batted-ball landing spot, feet from home plate ---
        *_hit_coordinate_exprs(),
        # --- location, normalized to this batter's zone ---
        (
            (pl.col("plate_z") - pl.col("sz_bot"))
            / (pl.col("sz_top") - pl.col("sz_bot")).replace(0.0, None)
        ).alias("plate_z_norm"),
        # --- count / game state ---
        (pl.col("balls").cast(pl.Utf8) + "-" + pl.col("strikes").cast(pl.Utf8)).alias(
            "count_state"
        ),
        _base_state_expr(),
        (pl.col("stand") == pl.col("p_throws")).alias("is_platoon_same"),
        # --- outcome flags ---
        pl.col("description").is_in(SWING_DESCRIPTIONS).alias("is_swing"),
        pl.col("description").is_in(WHIFF_DESCRIPTIONS).alias("is_whiff"),
        pl.col("description").is_in(CALLED_STRIKE_DESCRIPTIONS).alias("is_called_strike"),
        (pl.col("description") == "hit_into_play").alias("is_in_play"),
        # `is_tracked_pitch` means "has usable tracking", which is why a
        # physically impossible record fails it for the same reason an automatic
        # ball does: no pitch was measured. Both still occupy a row because they
        # move the count, and every model, mart, chart and API route already
        # filters on this flag — so quarantining here needs no downstream change.
        (~pl.col("description").is_in(NON_PITCH_DESCRIPTIONS) & ~_impossible_tracking_expr()).alias(
            "is_tracked_pitch"
        ),
        (~pl.col("description").is_in(NON_PITCH_DESCRIPTIONS | NON_COMPETITIVE_DESCRIPTIONS)).alias(
            "is_competitive"
        ),
    )

    df = df.with_columns(
        # CSW: the pitcher-quality rate that stabilizes fastest. Defined over
        # tracked pitches only.
        (pl.col("is_called_strike") | pl.col("is_whiff")).alias("is_csw"),
        (pl.col("base_state") * 3 + pl.col("outs_when_up")).cast(pl.Int8).alias("base_out_state"),
        # In-zone by the batter's own zone, not Savant's coarse `zone` field.
        (
            (pl.col("plate_x").abs() <= 0.83)  # half plate (8.5") + ball radius
            & (pl.col("plate_z") >= pl.col("sz_bot"))
            & (pl.col("plate_z") <= pl.col("sz_top"))
        ).alias("is_in_zone"),
    )

    return df.with_columns(
        (pl.col("is_swing") & ~pl.col("is_in_zone")).alias("is_chase"),
    )


def _season_raw_files(season: int, settings: Settings) -> list[Path]:
    return sorted((settings.raw_dir / "statcast" / f"season={season}").glob("*.csv.gz"))


def build_season(season: int, *, settings: Settings | None = None) -> int:
    """Rebuild one season's Parquet partitions from landed raw files.

    Reprocessing is cheap and local because raw was landed — this never re-crawls.
    """
    s = settings or get_settings()
    files = _season_raw_files(season, s)
    if not files:
        log.warning("season %d: no raw files", season)
        return 0

    frames = [SavantClient.read_raw(p) for p in files]
    df = pl.concat(frames, how="diagonal_relaxed")

    before = df.height
    # Savant occasionally serves overlapping rows across adjacent date queries.
    df = df.unique(subset=["game_pk", "at_bat_number", "pitch_number"], keep="first")
    if df.height != before:
        log.info("season %d: dropped %d duplicate pitches", season, before - df.height)

    df = enrich(df)

    # Quarantining is silent by construction — the rows stay, they just stop
    # being tracked — so say it out loud. A season that suddenly quarantines
    # thousands means the feed changed, not that the data got worse.
    quarantined = df.filter(
        ~pl.col("description").is_in(NON_PITCH_DESCRIPTIONS) & ~pl.col("is_tracked_pitch")
    ).height
    if quarantined:
        log.info("season %d: %d pitch(es) quarantined as impossible tracking", season, quarantined)

    out_dir = s.lake_dir / LAKE_TABLE / f"season={season}"
    if out_dir.exists():
        for stale in out_dir.glob("*.parquet"):
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    # `season` is written INTO the file as well as being the directory partition.
    # DuckDB 1.5.5 raises an InternalException when a query projects only a
    # synthesized hive-partition column ("SELECT max(season) FROM fact_pitch"),
    # so we do not rely on hive synthesis at all. Parquet row-group statistics on
    # the real column give equivalent pruning, and the duplication costs nothing.
    df.write_parquet(out_dir / "part_0.parquet", compression="zstd", statistics=True)
    log.info("season %d: wrote %d pitches -> %s", season, df.height, out_dir)
    return df.height


def build_all(
    seasons: list[int] | None = None, *, settings: Settings | None = None
) -> dict[int, int]:
    s = settings or get_settings()
    targets = seasons or sorted(
        int(p.name.split("=")[1]) for p in (s.raw_dir / "statcast").glob("season=*") if p.is_dir()
    )
    result = {season: build_season(season, settings=s) for season in targets}

    # Re-register unconditionally. DuckDB persists a view's resolved schema, so a
    # rebuild that changes the column set leaves the stored view pointing at a
    # shape the files no longer have — which surfaces later as a confusing
    # internal error rather than as "your view is stale".
    from bbetl.warehouse import register_all

    register_all(settings=s)
    return result


def latest_ingested_date(settings: Settings | None = None) -> dt.date | None:
    s = settings or get_settings()
    files = sorted((s.raw_dir / "statcast").glob("season=*/*.csv.gz"))
    if not files:
        return None
    return dt.date.fromisoformat(files[-1].stem.replace(".csv", ""))
