"""Spray chart (viz #8): per-batted-ball landing spots plus the smoothed
xwOBA-on-contact contour behind `mart_batter_spray`.

Pattern: `zones.py` + `pitches.py`. Batted-ball rows are per-event grain, so
they follow the pitch-level "Arrow, not JSON" rule; the contour is a single
small grid per batter-season, so it's JSON shaped exactly like `zones.py`'s
`player_zones` response — same `surface`/`reliability`/`extent`/`layout` keys —
so the client can reuse the same contour-consumption code path.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response

from bbapi.arrow import arrow_response, season_ttl
from bbapi.deps import require_table, settings, warehouse
from bbetl.transforms.spray import grid_extent

router = APIRouter(prefix="/spray", tags=["spray"])

BATTED_BALL_COLUMNS = [
    "x_ft",
    "y_ft",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "estimated_woba_using_speedangle",
    "events",
    "home_team",
]


@router.get("/extent")
def spray_extent() -> dict[str, Any]:
    """Grid geometry, so the client never hardcodes constants that could drift
    out of sync with `bbetl.transforms.spray`."""
    return grid_extent()


@router.get("/{mlbam_id}/battedballs")
def batted_balls(mlbam_id: int, season: int | None = None) -> Response:
    require_table("fact_pitch")
    cols = ", ".join(BATTED_BALL_COLUMNS)
    sql = f"""
        SELECT {cols} FROM fact_pitch
        WHERE batter = $id AND is_in_play AND is_tracked_pitch
          AND x_ft IS NOT NULL AND y_ft IS NOT NULL
          AND ($season IS NULL OR season = $season)
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
    """
    tbl = warehouse().execute(sql, {"id": mlbam_id, "season": season})
    return arrow_response(tbl, cache_seconds=season_ttl(season, settings().current_season))


@router.get("/{mlbam_id}/contour")
def spray_contour(mlbam_id: int, season: int | None = None) -> dict[str, Any]:
    require_table("mart_batter_spray")
    rows = (
        warehouse()
        .execute(
            """
        SELECT mlbam_id, season, n_batted_balls, grid_n, surface, reliability
        FROM mart_batter_spray
        WHERE mlbam_id = $id AND ($season IS NULL OR season = $season)
        ORDER BY season DESC
        LIMIT 1
        """,
            {"id": mlbam_id, "season": season},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(
            404,
            f"No spray contour for player {mlbam_id}"
            + (f" in {season}" if season else "")
            + ". The batter may not meet the minimum-batted-ball qualifier.",
        )
    row = rows[0]
    return {
        **row,
        "extent": grid_extent(),
        # Row-major, grid_n x grid_n, x-major then y — same convention as
        # `zones.py`'s `layout` field.
        "layout": "row_major_x_then_y",
    }
