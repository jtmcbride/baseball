"""Chadwick Bureau register — the cross-source ID crosswalk.

Identity resolution is the single biggest practical pain in combining Statcast,
FanGraphs, Baseball Reference and Retrosheet, because each publishes its own
player key and none of them publishes the others'. Everything downstream is keyed
on `mlbam_id`; this module is the only place other keys are allowed to appear.

Two deliberate choices:

1. Unmapped players are recorded and surfaced, never silently dropped. Quietly
   discarding a player because the register lacks a row is how a leaderboard ends
   up missing someone with nobody noticing.
2. `data/player_id_overrides.csv` is applied last and always wins, so a missing or
   wrong register row can be fixed without patching code or waiting upstream.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.http import RateLimitedClient

log = get_logger(__name__)

# The register is sharded 0-f by the first hex digit of the player's key.
REGISTER_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{shard}.csv"
)
SHARDS = "0123456789abcdef"

KEY_COLUMNS = [
    "key_mlbam",
    "key_fangraphs",
    "key_bbref",
    "key_retro",
    "name_first",
    "name_last",
    "birth_year",
    "mlb_played_first",
    "mlb_played_last",
]

OVERRIDES_FILENAME = "player_id_overrides.csv"


def fetch_register(*, settings: Settings | None = None) -> pl.DataFrame:
    """Download and concatenate all register shards."""
    s = settings or get_settings()
    client = RateLimitedClient(rps=2.0, user_agent=s.user_agent, timeout_s=120.0)
    frames: list[pl.DataFrame] = []
    try:
        for shard in SHARDS:
            resp = client.get(REGISTER_URL.format(shard=shard))
            df = pl.read_csv(
                io.BytesIO(resp.content),
                infer_schema_length=0,  # register mixes types; cast explicitly below
            )
            keep = [c for c in KEY_COLUMNS if c in df.columns]
            frames.append(df.select(keep))
            log.debug("register shard %s: %d rows", shard, df.height)
    finally:
        client.close()

    reg = pl.concat(frames, how="diagonal_relaxed")
    return reg.with_columns(
        pl.col("key_mlbam").cast(pl.Int64, strict=False),
        pl.col("key_fangraphs").cast(pl.Utf8),
        pl.col("key_bbref").cast(pl.Utf8),
        pl.col("key_retro").cast(pl.Utf8),
    ).filter(pl.col("key_mlbam").is_not_null())


def load_overrides(settings: Settings | None = None) -> pl.DataFrame:
    """Manual corrections. Applied last so they always win."""
    s = settings or get_settings()
    path: Path = s.data_root / OVERRIDES_FILENAME
    schema = {
        "key_mlbam": pl.Int64,
        "key_fangraphs": pl.Utf8,
        "key_bbref": pl.Utf8,
        "key_retro": pl.Utf8,
    }
    if not path.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_csv(path, schema_overrides=schema)


def build_crosswalk(*, settings: Settings | None = None) -> pl.DataFrame:
    s = settings or get_settings()
    reg = fetch_register(settings=s)

    # A handful of players carry duplicate register rows; prefer the row with the
    # most non-null external keys rather than an arbitrary first().
    reg = (
        reg.with_columns(
            (
                pl.col("key_fangraphs").is_not_null().cast(pl.Int8)
                + pl.col("key_bbref").is_not_null().cast(pl.Int8)
                + pl.col("key_retro").is_not_null().cast(pl.Int8)
            ).alias("_completeness")
        )
        .sort("_completeness", descending=True)
        .unique(subset=["key_mlbam"], keep="first")
        .drop("_completeness")
    )

    overrides = load_overrides(s)
    if overrides.height:
        reg = reg.join(overrides, on="key_mlbam", how="left", suffix="_ovr")
        for col in ("key_fangraphs", "key_bbref", "key_retro"):
            reg = reg.with_columns(pl.coalesce(pl.col(f"{col}_ovr"), pl.col(col)).alias(col)).drop(
                f"{col}_ovr"
            )
        log.info("applied %d ID overrides", overrides.height)

    out = reg.rename({"key_mlbam": "mlbam_id"})
    lake = s.lake_dir / "dim_player_ids"
    lake.mkdir(parents=True, exist_ok=True)
    out.write_parquet(lake / "part_0.parquet", compression="zstd")
    log.info("crosswalk: %d players", out.height)
    return out


def unmapped_players(
    seen_mlbam_ids: list[int], *, settings: Settings | None = None
) -> pl.DataFrame:
    """Players present in our data but missing from the crosswalk.

    Surfaced by the quality suite so gaps are visible rather than silent.
    """
    s = settings or get_settings()
    path = s.lake_dir / "dim_player_ids" / "part_0.parquet"
    if not path.exists():
        raise FileNotFoundError("Crosswalk not built — run `bb ingest crosswalk` first.")
    known = pl.read_parquet(path).select("mlbam_id")
    seen = pl.DataFrame({"mlbam_id": seen_mlbam_ids}).unique()
    return seen.join(known, on="mlbam_id", how="anti")
