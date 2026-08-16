"""Stuff+ / Location+ / Pitching+ — the model-graded arsenal.

Reads `mart_pitcher_stuff`, which is built by `bb-ml stuff` rather than by
`bb build marts`: every row is three boosters scored over every pitch. If these
routes 503, the models have not been trained on this machine yet.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import latest_season, require_table, warehouse

router = APIRouter(prefix="/stuff", tags=["stuff"])

MART = "mart_pitcher_stuff"

# The rollup row a pitcher-season carries alongside its per-pitch-type rows.
ALL_PITCHES = "ALL"

# Whitelisted rather than interpolated: these land in an ORDER BY.
SORTABLE = {"stuff_plus", "location_plus", "pitching_plus", "rv_per_100", "pitches"}


@router.get("/{mlbam_id}")
def pitcher_stuff(mlbam_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """One pitcher's graded arsenal, newest season first.

    Includes the `ALL` rollup row so the caller can show a headline grade without
    re-deriving a usage weighting the mart already applied.
    """
    require_table(MART)
    rows = (
        warehouse()
        .execute(
            f"""
        SELECT * FROM {MART}
        WHERE mlbam_id = $id AND ($season IS NULL OR season = $season)
        ORDER BY season DESC,
                 CASE WHEN pitch_type = '{ALL_PITCHES}' THEN 0 ELSE 1 END,
                 pitches DESC
        """,
            {"id": mlbam_id, "season": season},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, "No graded pitches for this pitcher.")
    return rows


@router.get("")
def leaderboard(
    season: int | None = None,
    metric: str = Query("stuff_plus", description=f"One of {sorted(SORTABLE)}."),
    pitch_type: str | None = Query(None, description=f"Omit for the '{ALL_PITCHES}' rollup."),
    min_pitches: int = Query(500, ge=1),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Top pitchers by any of the three grades.

    `min_pitches` defaults high on purpose. The mart keeps rows down to 25
    pitches because they are worth showing on a player page next to their sample
    size, but a leaderboard sorted on a 25-pitch grade is a list of small
    samples, not a list of good pitchers.
    """
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
          AND s.pitch_type = $pitch_type
          AND s.pitches >= $min_pitches
        ORDER BY s.{metric} DESC
        LIMIT $limit
        """,
            {
                "season": season or latest_season(),
                "pitch_type": pitch_type or ALL_PITCHES,
                "min_pitches": min_pitches,
                "limit": limit,
            },
        )
        .to_pylist()
    )
