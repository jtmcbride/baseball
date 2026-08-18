"""Model #2's re-derived arsenals and model #11's 2D embedding/similarity.

Reads `mart_pitcher_arsenal_clusters`, `mart_arsenal_embedding`, and
`mart_arsenal_neighbors`, all built by `bb-ml arsenal` / `bb-ml arsenal-embed`.
If these routes 503, those commands have not been run on this machine yet.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import require_table, warehouse

router = APIRouter(prefix="/arsenal", tags=["arsenal"])

CLUSTER_MART = "mart_pitcher_arsenal_clusters"
EMBEDDING_MART = "mart_arsenal_embedding"
NEIGHBORS_MART = "mart_arsenal_neighbors"


# Declared before `/{mlbam_id}` -- FastAPI matches routes in registration
# order, and an int path param would otherwise swallow "embedding" as a 422.
@router.get("/embedding")
def embedding(
    season: int | None = None,
    min_pitches: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """Every embedded pitcher-season -- the points for the UMAP arsenal map."""
    require_table(EMBEDDING_MART)
    return (
        warehouse()
        .execute(
            f"""
        SELECT * FROM {EMBEDDING_MART}
        WHERE ($season IS NULL OR season = $season) AND n_pitches >= $min_pitches
        """,
            {"season": season, "min_pitches": min_pitches},
        )
        .to_pylist()
    )


@router.get("/{mlbam_id}")
def clusters(mlbam_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """One pitcher's re-derived clusters, newest season first."""
    require_table(CLUSTER_MART)
    rows = (
        warehouse()
        .execute(
            f"""
        SELECT * FROM {CLUSTER_MART}
        WHERE mlbam_id = $id AND ($season IS NULL OR season = $season)
        ORDER BY season DESC, usage_pct DESC
        """,
            {"id": mlbam_id, "season": season},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, "No re-derived arsenal for this pitcher.")
    return rows


@router.get("/{mlbam_id}/similar")
def similar(
    mlbam_id: int,
    season: int | None = None,
    limit: int = Query(10, ge=1, le=10),
) -> list[dict[str, Any]]:
    """Nearest pitcher-seasons by re-derived arsenal shape ("who does this
    pitcher resemble?" -- model #11). Defaults to the pitcher's most recent
    embedded season when `season` is omitted."""
    require_table(NEIGHBORS_MART)
    require_table(EMBEDDING_MART)
    if season is None:
        season = warehouse().scalar(
            f"SELECT max(season) FROM {EMBEDDING_MART} WHERE mlbam_id = $id", {"id": mlbam_id}
        )
        if season is None:
            raise HTTPException(404, "No embedded pitcher-season for this pitcher.")
    rows = (
        warehouse()
        .execute(
            f"""
        SELECT n.rank, n.distance, n.neighbor_id, n.neighbor_season,
               p.full_name, e.archetype_label, e.primary_label, e.primary_velo
        FROM {NEIGHBORS_MART} n
        LEFT JOIN dim_player p ON p.mlbam_id = n.neighbor_id
        LEFT JOIN {EMBEDDING_MART} e ON e.mlbam_id = n.neighbor_id AND e.season = n.neighbor_season
        WHERE n.mlbam_id = $id AND n.season = $season
        ORDER BY n.rank
        LIMIT $limit
        """,
            {"id": mlbam_id, "season": season, "limit": limit},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, "No similar pitchers for this pitcher-season.")
    return rows
