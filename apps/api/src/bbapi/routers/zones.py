"""Hot/cold zone grids.

The reliability mask ships with every grid rather than being applied server-side.
The client needs both numbers: the surface to colour the cell, and the effective
sample size to decide how much to fade it. Collapsing them here — masking to null
before sending — would throw away the information that makes the chart honest.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import require_table, warehouse
from bbetl.transforms.zones import grid_extent

router = APIRouter(prefix="/zones", tags=["zones"])

VALID_METRICS = {"xwoba", "whiff", "swing", "exit_velo", "run_value", "framing", "strike_rate"}
VALID_ROLES = {"batter", "pitcher", "catcher", "umpire"}


@router.get("/extent")
def zone_extent() -> dict[str, Any]:
    """Grid geometry, so the client never hardcodes constants that could drift
    out of sync with the transform that produced the grids."""
    return grid_extent()


@router.get("/{mlbam_id}")
def player_zones(
    mlbam_id: int,
    role: str = Query("batter"),
    metric: str = Query("xwoba"),
    season: int | None = None,
) -> dict[str, Any]:
    require_table("mart_zone_profile")
    if role not in VALID_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(VALID_ROLES)}")
    if metric not in VALID_METRICS:
        raise HTTPException(400, f"metric must be one of {sorted(VALID_METRICS)}")

    rows = (
        warehouse()
        .execute(
            """
        SELECT mlbam_id, season, role, metric, n_pitches, grid_n, surface, reliability
        FROM mart_zone_profile
        WHERE mlbam_id = $id AND role = $role AND metric = $metric
          AND ($season IS NULL OR season = $season)
        ORDER BY season DESC
        LIMIT 1
        """,
            {"id": mlbam_id, "role": role, "metric": metric, "season": season},
        )
        .to_pylist()
    )

    if not rows:
        raise HTTPException(
            404,
            f"No {metric} grid for player {mlbam_id} as {role}"
            + (f" in {season}" if season else "")
            + ". The player may not meet the minimum-pitch qualifier.",
        )

    row = rows[0]
    return {
        **row,
        "extent": grid_extent(),
        # Row-major, grid_n x grid_n, x-major then z. Stated explicitly so the
        # client is not reverse-engineering the layout from a flat array.
        "layout": "row_major_x_then_z",
    }


@router.get("/{mlbam_id}/available")
def available_grids(mlbam_id: int) -> list[dict[str, Any]]:
    require_table("mart_zone_profile")
    return (
        warehouse()
        .execute(
            """
        SELECT season, role, metric, n_pitches
        FROM mart_zone_profile WHERE mlbam_id = $id
        ORDER BY season DESC, role, metric
        """,
            {"id": mlbam_id},
        )
        .to_pylist()
    )
