"""Player search, profile, and arsenal — small JSON payloads."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import latest_season, require_table, warehouse

router = APIRouter(prefix="/players", tags=["players"])


@router.get("/search")
def search_players(
    q: str = Query(..., min_length=2, description="Name fragment."),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Name search over players who actually appear in our pitch data."""
    require_table("dim_player")
    sql = """
        SELECT p.mlbam_id, p.full_name, p.primary_position, p.bats, p.throws,
               p.mlb_debut_date
        FROM dim_player p
        WHERE lower(p.full_name) LIKE lower($pattern)
        ORDER BY
            -- Prefix matches first: typing "Judge" should not rank
            -- "Judgeson" style substring hits above the obvious answer.
            CASE WHEN lower(p.full_name) LIKE lower($prefix) THEN 0 ELSE 1 END,
            p.full_name
        LIMIT $limit
    """
    tbl = warehouse().execute(sql, {"pattern": f"%{q}%", "prefix": f"{q}%", "limit": limit})
    return tbl.to_pylist()


@router.get("/{mlbam_id}/profile")
def player_profile(mlbam_id: int, season: int | None = None) -> dict[str, Any]:
    require_table("dim_player")
    bio = (
        warehouse()
        .execute("SELECT * FROM dim_player WHERE mlbam_id = $id", {"id": mlbam_id})
        .to_pylist()
    )
    if not bio:
        raise HTTPException(404, f"No player with mlbam_id {mlbam_id}")

    seasons = (
        warehouse()
        .execute(
            """
        SELECT season,
               count(*) FILTER (WHERE pitcher = $id) AS pitches_thrown,
               count(*) FILTER (WHERE batter  = $id) AS pitches_seen
        FROM fact_pitch
        WHERE pitcher = $id OR batter = $id
        GROUP BY season ORDER BY season DESC
        """,
            {"id": mlbam_id},
        )
        .to_pylist()
    )

    return {"player": bio[0], "seasons": seasons, "requested_season": season}


@router.get("/{mlbam_id}/arsenal")
def player_arsenal(mlbam_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """Pitch arsenal with league percentile ranks.

    Percentiles come from the mart, where they are computed within
    (season, pitch_type) — ranking a slider against sliders rather than against
    fastballs.
    """
    require_table("mart_pitcher_arsenal")
    sql = """
        SELECT * FROM mart_pitcher_arsenal
        WHERE mlbam_id = $id
          AND ($season IS NULL OR season = $season)
        ORDER BY season DESC, pitches DESC
    """
    return warehouse().execute(sql, {"id": mlbam_id, "season": season}).to_pylist()


@router.get("/{mlbam_id}/ids")
def player_external_ids(mlbam_id: int) -> dict[str, Any]:
    """Cross-source keys (FanGraphs, Baseball Reference, Retrosheet)."""
    require_table("dim_player_ids")
    rows = (
        warehouse()
        .execute("SELECT * FROM dim_player_ids WHERE mlbam_id = $id", {"id": mlbam_id})
        .to_pylist()
    )
    if not rows:
        raise HTTPException(404, f"No crosswalk entry for {mlbam_id}")
    return rows[0]


@router.get("/{mlbam_id}/games")
def player_games(
    mlbam_id: int,
    season: int | None = None,
    limit: int = Query(25, ge=1, le=200),
) -> list[dict[str, Any]]:
    """A pitcher's most recent games — backs the at-bat replay picker."""
    require_table("fact_pitch")
    sql = """
        SELECT game_pk, game_date, season, count(*) AS pitches
        FROM fact_pitch
        WHERE pitcher = $id AND ($season IS NULL OR season = $season)
        GROUP BY game_pk, game_date, season
        ORDER BY game_date DESC
        LIMIT $limit
    """
    return warehouse().execute(sql, {"id": mlbam_id, "season": season, "limit": limit}).to_pylist()


@router.get("")
def list_leaders(
    season: int | None = None,
    min_pitches: int = Query(500, ge=1),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Pitchers by volume — the default landing list."""
    require_table("mart_pitcher_arsenal")
    sql = """
        SELECT a.mlbam_id, p.full_name, a.season, a.p_throws,
               sum(a.pitches) AS pitches,
               count(*) AS pitch_types,
               round(sum(a.pitches * a.velo_avg) / sum(a.pitches), 1) AS velo_avg,
               round(sum(a.pitches * a.csw_pct) / sum(a.pitches), 1) AS csw_pct
        FROM mart_pitcher_arsenal a
        JOIN dim_player p USING (mlbam_id)
        WHERE ($season IS NULL OR a.season = $season)
        GROUP BY 1,2,3,4
        HAVING sum(a.pitches) >= $min_pitches
        ORDER BY pitches DESC
        LIMIT $limit
    """
    return (
        warehouse()
        .execute(
            sql,
            {"season": season or latest_season(), "min_pitches": min_pitches, "limit": limit},
        )
        .to_pylist()
    )
