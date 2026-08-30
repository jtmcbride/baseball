"""Pitch-level data — the large payloads, served as Arrow IPC."""

from __future__ import annotations

from typing import Any

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


TRAJECTORY_COLUMNS = [
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
    "release_speed",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
]


@router.get("/trajectory")
def pitch_trajectory(game_pk: int, at_bat_number: int, pitch_number: int) -> dict[str, Any]:
    """The raw 9-parameter physics fit for one pitch, keyed by its natural id.

    `vx0/vy0/vz0/ax/ay/az` are Statcast's fitted constant-acceleration
    trajectory, valid at the fixed reference y=50ft — NOT at the actual release
    point. The client reconstructs the exact flight path (release to plate) by
    solving the same quadratic backward from y=50 to `release_pos_y`, then
    forward to y=17/12 (front of plate); see `PitchTrajectory3D.tsx`. Validated
    against real `plate_x`/`plate_z` to ~0.003ft mean error before shipping —
    do not change the y=50 or y=17/12 reference points without re-validating.
    """
    require_table("fact_pitch")
    cols = ", ".join(TRAJECTORY_COLUMNS)
    row = (
        warehouse()
        .execute(
            f"""
        SELECT {cols} FROM fact_pitch
        WHERE game_pk = $game_pk AND at_bat_number = $at_bat_number
          AND pitch_number = $pitch_number AND vx0 IS NOT NULL
        LIMIT 1
        """,
            {"game_pk": game_pk, "at_bat_number": at_bat_number, "pitch_number": pitch_number},
        )
        .to_pylist()
    )
    if not row:
        raise HTTPException(404, "No tracked trajectory for this pitch.")
    return row[0]


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


# The physics columns plus the identity/shape columns a comparison view needs to
# label a line and let the reader jump back to the real pitch it came from.
EXEMPLAR_COLUMNS = [
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    *TRAJECTORY_COLUMNS,
    "ivb_in",
    "hb_arm_in",
    "release_spin_rate",
]


@router.get("/arsenal-trajectories")
def arsenal_trajectories(
    pitcher_id: int,
    season: int | None = None,
    vs_hand: str | None = Query(None, pattern="^[LR]$"),
    min_pitches: int = Query(15, ge=1, le=5_000),
) -> list[dict[str, Any]]:
    """One representative flight per pitch type — the arsenal, overlaid.

    A pitch-comparison view has to answer "what does this pitcher's slider look
    like next to his fastball", and there is no such thing as an average
    *trajectory*: averaging the nine physics parameters across a season produces
    a path no pitch ever took, and the average of two release points is a
    release point in between them that the pitcher never used. So this returns
    a real thrown pitch — the one nearest its own type's centroid in
    standardized (velo, IVB, arm-side HB) space, which is the closest thing to
    "his typical slider" that is also a pitch that actually happened.

    Types below `min_pitches` are dropped: the exemplar of a 3-pitch sample is
    as likely to be a mislabel as a pitch, and it would draw at the same weight
    as the fastball.
    """
    require_table("fact_pitch")
    cols = ", ".join(f"src.{c}" for c in EXEMPLAR_COLUMNS)
    sql = f"""
        WITH src AS (
            SELECT {", ".join(EXEMPLAR_COLUMNS)}
            FROM fact_pitch
            WHERE pitcher = $id AND is_tracked_pitch AND is_competitive
              AND pitch_type IS NOT NULL AND vx0 IS NOT NULL
              AND release_speed IS NOT NULL AND ivb_in IS NOT NULL
              AND hb_arm_in IS NOT NULL
              AND ($season IS NULL OR season = $season)
              AND ($vs_hand IS NULL OR stand = $vs_hand)
        ),
        shape AS (
            SELECT pitch_type,
                   count(*)                       AS n,
                   avg(release_speed)             AS velo_avg,
                   avg(ivb_in)                    AS ivb_avg,
                   avg(hb_arm_in)                 AS hb_avg,
                   -- A single-valued sample has zero (or null) spread; the
                   -- coalesce keeps its z-score finite rather than dividing by
                   -- zero and ranking every candidate NaN.
                   coalesce(nullif(stddev_samp(release_speed), 0), 1) AS velo_sd,
                   coalesce(nullif(stddev_samp(ivb_in), 0), 1)        AS ivb_sd,
                   coalesce(nullif(stddev_samp(hb_arm_in), 0), 1)     AS hb_sd
            FROM src GROUP BY pitch_type
        ),
        ranked AS (
            SELECT {cols}, shape.n, shape.velo_avg, shape.ivb_avg, shape.hb_avg,
                   row_number() OVER (
                       PARTITION BY src.pitch_type ORDER BY
                           pow((src.release_speed - shape.velo_avg) / shape.velo_sd, 2)
                         + pow((src.ivb_in       - shape.ivb_avg)  / shape.ivb_sd,  2)
                         + pow((src.hb_arm_in    - shape.hb_avg)   / shape.hb_sd,   2)
                   ) AS rn
            FROM src JOIN shape USING (pitch_type)
            WHERE shape.n >= $min_pitches
        )
        SELECT * EXCLUDE (rn, velo_avg, ivb_avg, hb_avg),
               round(velo_avg, 1) AS velo_avg,
               round(ivb_avg, 1)  AS ivb_avg,
               round(hb_avg, 1)   AS hb_avg,
               -- Share of every competitive tracked pitch in the window, not of
               -- the kept types only: a type dropped by `min_pitches` must not
               -- inflate the usage of the ones that survived.
               round(100.0 * n / (SELECT count(*) FROM src), 1) AS usage_pct
        FROM ranked WHERE rn = 1
        ORDER BY n DESC
    """
    rows = (
        warehouse()
        .execute(
            sql,
            {"id": pitcher_id, "season": season, "vs_hand": vs_hand, "min_pitches": min_pitches},
        )
        .to_pylist()
    )
    if not rows:
        raise HTTPException(
            404,
            f"No tracked pitch types for pitcher {pitcher_id}"
            + (f" in {season}" if season else "")
            + f" with at least {min_pitches} pitches.",
        )
    return rows
