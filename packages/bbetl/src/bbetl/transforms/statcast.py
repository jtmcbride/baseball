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
NON_COMPETITIVE_DESCRIPTIONS = frozenset({"pitchout"})


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
        (~pl.col("description").is_in(NON_PITCH_DESCRIPTIONS)).alias("is_tracked_pitch"),
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
