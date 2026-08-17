"""Swing-path plane value — the model-graded batter swing.

Reads `mart_batter_swing`, built by `bb-ml swing` / `bb-ml swing-mart`: every
row is both `SwingPathModel` heads' counterfactual, aggregated per batter x
season. If these routes 503, the models have not been trained on this
machine yet.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import latest_season, require_table, warehouse

router = APIRouter(prefix="/swing", tags=["swing"])

MART = "mart_batter_swing"

SORTABLE = {"whiff_plane_value_per_100", "contact_plane_value_per_100", "whiff_swings"}


@router.get("/{mlbam_id}")
def batter_swing(mlbam_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """One batter's swing-plane grade, newest season first."""
    require_table(MART)
    rows = (
        warehouse()
        .execute(
            f"""
        SELECT * FROM {MART}
        WHERE mlbam_id = $id AND ($season IS NULL OR season = $season)
        ORDER BY season DESC
        """,
            {"id": mlbam_id, "season": season},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, "No graded swings for this batter.")
    return rows


@router.get("")
def leaderboard(
    season: int | None = None,
    metric: str = Query("whiff_plane_value_per_100", description=f"One of {sorted(SORTABLE)}."),
    min_swings: int = Query(200, ge=1),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    require_table(MART)
    if metric not in SORTABLE:
        raise HTTPException(400, f"metric must be one of {sorted(SORTABLE)}")
    return (
        warehouse()
        .execute(
            f"""
        SELECT s.*, p.full_name
        FROM {MART} s
        LEFT JOIN dim_player p USING (mlbam_id)
        WHERE s.season = $season
          AND s.whiff_swings >= $min_swings
        ORDER BY s.{metric} DESC
        LIMIT $limit
        """,
            {
                "season": season or latest_season(),
                "min_swings": min_swings,
                "limit": limit,
            },
        )
        .to_pylist()
    )
