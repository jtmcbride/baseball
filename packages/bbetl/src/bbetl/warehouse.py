"""Wire lake Parquet datasets into the warehouse under logical names.

Callers ask for `fact_pitch`, never for a path. That indirection is what lets the
Postgres backend land in M3 without touching a single query.
"""

from __future__ import annotations

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbcore.storage import open_warehouse

log = get_logger(__name__)

# logical name -> path pattern relative to the lake root
LAKE_TABLES: dict[str, str] = {
    "fact_pitch": "fact_pitch/season=*/*.parquet",
    "dim_player": "dim_player/*.parquet",
    "dim_team": "dim_team/*.parquet",
    "dim_game": "dim_game/*.parquet",
    "dim_player_ids": "dim_player_ids/*.parquet",
    "mart_zone_profile": "mart_zone_profile/*.parquet",
}


def register_all(*, settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    registered: list[str] = []
    with open_warehouse(settings=s) as wh:
        for name, pattern in LAKE_TABLES.items():
            # Only register what actually exists — partial pipelines are normal
            # during a backfill and should not hard-fail the whole registration.
            if not any((s.lake_dir).glob(pattern)):
                log.debug("skipping %s — no files match %s", name, pattern)
                continue
            wh.register_lake_table(name, pattern)
            registered.append(name)
    return registered
