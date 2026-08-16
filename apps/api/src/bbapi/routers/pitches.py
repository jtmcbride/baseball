"""Pitch-level data — the large payloads, served as Arrow IPC."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from bbapi.arrow import arrow_response, season_ttl
from bbapi.deps import latest_season, require_table, settings, warehouse

router = APIRouter(prefix="/pitches", tags=["pitches"])

# Columns the charts actually need. Selecting all 119 would triple the payload
# for no benefit — the movement plot and pitch scatter use a fraction of them.
PITCH_COLUMNS = [
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "ivb_in",
    "hb_arm_in",
    "arm_angle",
    "spin_axis",
    "plate_x",
    "plate_z",
    "plate_z_norm",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "description",
    "events",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "delta_run_exp",
    "is_swing",
    "is_whiff",
    "is_called_strike",
    "is_in_play",
    "is_in_zone",
    "is_chase",
    "is_csw",
]


@router.get("")
def get_pitches(
    pitcher_id: int | None = None,
    batter_id: int | None = None,
    season: int | None = None,
    game_pk: int | None = None,
    pitch_type: str | None = None,
    vs_hand: str | None = Query(None, pattern="^[LR]$"),
    limit: int = Query(50_000, ge=1, le=500_000),
) -> Response:
    """Filtered pitch rows as Arrow IPC.

    At least one of pitcher_id, batter_id, or game_pk is required — an unfiltered
    scan of 7.7M pitches is never what a chart wants and would be trivial to
    trigger accidentally.
    """
    require_table("fact_pitch")
    if pitcher_id is None and batter_id is None and game_pk is None:
        raise HTTPException(400, "Provide at least one of: pitcher_id, batter_id, game_pk.")

    cols = ", ".join(PITCH_COLUMNS)
    sql = f"""
        SELECT {cols} FROM fact_pitch
        WHERE is_tracked_pitch
          AND ($pitcher IS NULL OR pitcher = $pitcher)
          AND ($batter  IS NULL OR batter  = $batter)
          AND ($season  IS NULL OR season  = $season)
          AND ($game_pk IS NULL OR game_pk = $game_pk)
          AND ($pitch_type IS NULL OR pitch_type = $pitch_type)
          AND ($vs_hand IS NULL OR stand = $vs_hand)
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        LIMIT $limit
    """
    tbl = warehouse().execute(
        sql,
        {
            "pitcher": pitcher_id,
            "batter": batter_id,
            "season": season,
            "game_pk": game_pk,
            "pitch_type": pitch_type,
            "vs_hand": vs_hand,
            "limit": limit,
        },
    )
    return arrow_response(tbl, cache_seconds=season_ttl(season, settings().current_season))


@router.get("/movement")
def movement_summary(
    pitcher_id: int,
    season: int | None = None,
) -> Response:
    """Per-pitch movement points plus league reference ellipses.

    The league averages come back in the same payload so the chart can draw its
    reference marks without a second round trip.
    """
    require_table("fact_pitch")
    sql = """
        SELECT pitch_type, pitch_name, release_speed, ivb_in, hb_arm_in,
               release_spin_rate, is_whiff, is_swing, description
        FROM fact_pitch
        WHERE pitcher = $id AND is_tracked_pitch AND is_competitive
          AND pitch_type IS NOT NULL
          AND ($season IS NULL OR season = $season)
    """
    tbl = warehouse().execute(sql, {"id": pitcher_id, "season": season})
    return arrow_response(tbl, cache_seconds=season_ttl(season, settings().current_season))


@router.get("/league-shapes")
def league_pitch_shapes(season: int | None = None, hand: str = Query("R", pattern="^[LR]$")):
    """League-average shape per pitch type — reference marks for movement plots."""
    require_table("fact_pitch")
    return (
        warehouse()
        .execute(
            """
        SELECT pitch_type,
               count(*) AS n,
               round(avg(release_speed), 1) AS velo,
               round(avg(ivb_in), 1)        AS ivb_in,
               round(avg(hb_arm_in), 1)     AS hb_arm_in,
               round(stddev_samp(ivb_in), 2)    AS ivb_sd,
               round(stddev_samp(hb_arm_in), 2) AS hb_sd
        FROM fact_pitch
        WHERE is_tracked_pitch AND is_competitive AND pitch_type IS NOT NULL
          AND p_throws = $hand
          AND ($season IS NULL OR season = $season)
        GROUP BY pitch_type
        HAVING count(*) >= 100
        ORDER BY n DESC
        """,
            {"season": season or latest_season(), "hand": hand},
        )
        .to_pylist()
    )
