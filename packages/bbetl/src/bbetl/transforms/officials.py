"""`dim_official`: the home-plate umpire for every game.

Statcast's own `umpire` column is empty in every season (see
`bbetl.sources.statsapi`), so the only source is the raw boxscore JSON that
`ingest_officials` lands one file per game at `data/raw/officials/<game_pk>.json`.
This module turns that raw drop into a lake table the same way
`transforms.statcast` turns Savant CSVs into `fact_pitch`, so the called-strike
model can join on `game_pk` without knowing anything about the raw JSON shape.
"""

from __future__ import annotations

import json

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger

log = get_logger(__name__)

DIM_NAME = "dim_official"


def build_dim_official(*, settings: Settings | None = None) -> pl.DataFrame:
    """Read every landed officials JSON file and write `dim_official`.

    Not every game has all four umpire roles recorded (a rare boxscore gap), so
    rows are read with a schema scan over the full file list rather than the
    first N — a home-plate-only game early in the list must not truncate the
    inferred schema and null out base umpires everywhere else.
    """
    s = settings or get_settings()
    files = sorted((s.raw_dir / "officials").glob("*.json"))
    if not files:
        raise ValueError("No officials landed — run `bb ingest officials` first.")
    rows = [json.loads(f.read_text()) for f in files]
    df = pl.DataFrame(rows, infer_schema_length=None).unique(
        subset=["game_pk"], keep="last"
    )
    out = s.lake_dir / DIM_NAME
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "part_0.parquet", compression="zstd")
    log.info("wrote %s: %d rows", DIM_NAME, df.height)
    return df
