"""Catcher framing runs and umpire zone edge — the called-strike model's residual.

Reads `mart_catcher_framing` and `mart_umpire_zone`, built by `bb-ml
called-strike` / `bb-ml called-strike-mart`. If these routes 503, the model
has not been trained on this machine yet.

Umpires are not players — there is no `dim_player` row for one — so umpire
names come from `dim_official` (deduplicated by id inline) rather than the
`dim_player` join every other leaderboard uses.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import latest_season, require_table, warehouse

router = APIRouter(prefix="/framing", tags=["framing"])

CATCHER_MART = "mart_catcher_framing"
UMPIRE_MART = "mart_umpire_zone"


@router.get("/catchers/{mlbam_id}")
def catcher_framing(mlbam_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """One catcher's framing runs, newest season first."""
    require_table(CATCHER_MART)
    rows = (
        warehouse()
        .execute(
            f"""
        SELECT * FROM {CATCHER_MART}
        WHERE mlbam_id = $id AND ($season IS NULL OR season = $season)
        ORDER BY season DESC
        """,
            {"id": mlbam_id, "season": season},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, "No framing grade for this catcher.")
    return rows


@router.get("/catchers")
def catcher_leaderboard(
    season: int | None = None,
    min_pitches: int = Query(500, ge=1),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    require_table(CATCHER_MART)
    return (
        warehouse()
        .execute(
            f"""
        SELECT c.*, p.full_name
        FROM {CATCHER_MART} c
        LEFT JOIN dim_player p USING (mlbam_id)
        WHERE c.season = $season AND c.n >= $min_pitches
        ORDER BY c.framing_runs DESC
        LIMIT $limit
        """,
            {"season": season or latest_season(), "min_pitches": min_pitches, "limit": limit},
        )
        .to_pylist()
    )


@router.get("/umpires")
def umpire_leaderboard(
    season: int | None = None,
    min_pitches: int = Query(500, ge=1),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Ranked by `edge`: actual minus expected called-strike rate on borderline
    takes. Positive means strike-happy, negative means ball-happy, relative to
    the model's average-umpire expectation at that same location and count.

    Default `min_pitches` matches `mart_umpire_zone`'s own build floor
    (`MIN_UMPIRE_PITCHES` in `bbml.marts`) — a season's single busiest umpire
    still only clears ~800 *borderline* takes, so anything much stricter than
    the mart's own floor returns nothing for every umpire, every season.
    """
    require_table(UMPIRE_MART)
    require_table("dim_official")
    return (
        warehouse()
        .execute(
            f"""
        SELECT u.*, o.umpire_home_plate_name AS full_name
        FROM {UMPIRE_MART} u
        LEFT JOIN (
            SELECT DISTINCT umpire_home_plate_id, umpire_home_plate_name FROM dim_official
        ) o ON o.umpire_home_plate_id = u.mlbam_id
        WHERE u.season = $season AND u.n >= $min_pitches
        ORDER BY abs(u.edge) DESC
        LIMIT $limit
        """,
            {"season": season or latest_season(), "min_pitches": min_pitches, "limit": limit},
        )
        .to_pylist()
    )
