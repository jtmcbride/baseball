"""Shared dependencies: one warehouse for the process, plus query guards."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, Query

from bbcore.config import Settings, get_settings
from bbcore.storage import Warehouse, open_warehouse


@lru_cache(maxsize=1)
def warehouse() -> Warehouse:
    """A single read-only warehouse, reused across requests.

    DuckDB connections are not thread-safe and FastAPI serves sync handlers from a
    threadpool, so DuckDBWarehouse serializes access internally rather than
    opening a connection per request (which would reopen the DB file constantly).
    """
    return open_warehouse(read_only=True)


def settings() -> Settings:
    return get_settings()


def latest_season() -> int:
    """The most recent season that actually has data.

    Deliberately not `Settings.current_season`: the calendar rolls into a new
    season months before any pitches are thrown, and a partial backfill may stop
    anywhere. Defaulting to the calendar year silently returns empty results and
    looks like a broken app rather than an un-ingested season.
    """
    if not warehouse().table_exists("fact_pitch"):
        return get_settings().current_season
    got = warehouse().scalar("SELECT max(season) FROM fact_pitch")
    return int(got) if got is not None else get_settings().current_season


def require_table(name: str) -> None:
    """Fail with a useful message rather than a SQL error when the pipeline has
    not been run yet — the most likely reason a fresh clone 500s."""
    if not warehouse().table_exists(name):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Table '{name}' is not built yet. Run: "
                "`bb ingest statcast && bb build pitches && bb build register && bb build marts`"
            ),
        )


SeasonQuery = Query(None, ge=2015, le=2100, description="Season; omit for career.")
LimitQuery = Query(5000, ge=1, le=100_000, description="Max rows.")
