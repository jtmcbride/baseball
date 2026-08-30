"""Whole at-bats — the sequence, not the pitch.

Everything else in this API is keyed by a player or a single pitch. An at-bat is
the unit a reader actually reasons in ("he set up the slider with two fastballs
away"), and the sequence is the information: the same slider is a different
pitch after 0-0 heat than it is after a changeup. So these routes return the
pitches of one plate appearance *in order*, each carrying its own physics fit,
so the client can fly all of them through the same strike zone.

Untracked pitches (pitch-clock violations, ABS-awarded calls, older seasons
missing the physics fit) are returned too, with null physics. They are part of
the sequence and they move the count; dropping them here would silently renumber
the at-bat and make a 3-2 count appear out of nowhere.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from bbapi.deps import require_table, warehouse

router = APIRouter(prefix="/atbats", tags=["atbats"])

# Physics plus outcome plus count state — everything the 3D view and the
# sequence list need for one pitch, in one payload.
AT_BAT_PITCH_COLUMNS = [
    "pitch_number",
    "balls",
    "strikes",
    "outs_when_up",
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
    "release_speed",
    "release_extension",
    "release_spin_rate",
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
    "plate_z_norm",
    "sz_top",
    "sz_bot",
    "ivb_in",
    "hb_arm_in",
    "description",
    "events",
    "is_swing",
    "is_whiff",
    "is_called_strike",
    "is_in_play",
    "delta_run_exp",
]


def _names(ids: list[int]) -> dict[int, str]:
    """mlbam_id -> full name, or an empty map when the dim table is not built.

    Names are cosmetic here; an at-bat with an id instead of a name is still a
    usable at-bat, so a missing `dim_player` degrades the label rather than
    failing the request.
    """
    ids = [i for i in {*ids} if i is not None]
    if not ids or not warehouse().table_exists("dim_player"):
        return {}
    rows = (
        warehouse()
        .execute(
            "SELECT mlbam_id, full_name FROM dim_player WHERE mlbam_id IN "
            f"({', '.join(str(int(i)) for i in ids)})"
        )
        .to_pylist()
    )
    return {r["mlbam_id"]: r["full_name"] for r in rows}


@router.get("")
def list_atbats(
    pitcher_id: int | None = None,
    batter_id: int | None = None,
    game_pk: int | None = None,
    season: int | None = None,
    limit: int = Query(60, ge=1, le=500),
) -> list[dict[str, Any]]:
    """At-bat summaries for a picker: who, when, how many pitches, what happened.

    At least one of pitcher_id, batter_id, or game_pk is required, for the same
    reason `/pitches` requires one — an unfiltered scan of every plate
    appearance ever is never what a picker wants.
    """
    require_table("fact_pitch")
    if pitcher_id is None and batter_id is None and game_pk is None:
        raise HTTPException(400, "Provide at least one of: pitcher_id, batter_id, game_pk.")

    rows = (
        warehouse()
        .execute(
            """
        SELECT game_pk, at_bat_number,
               any_value(game_date) AS game_date,
               any_value(pitcher)   AS pitcher,
               any_value(batter)    AS batter,
               any_value(inning)    AS inning,
               any_value(stand)     AS stand,
               any_value(p_throws)  AS p_throws,
               count(*)             AS pitches,
               count(*) FILTER (WHERE vx0 IS NOT NULL) AS tracked_pitches,
               -- The outcome is the LAST pitch's, not any pitch's: `events` is
               -- null on every pitch of the at-bat except the one that ended
               -- it, and max() over a mostly-null column would still find it
               -- but would pick alphabetically among the foul-then-single case
               -- if the data ever carried two.
               arg_max(events, pitch_number)      AS result,
               arg_max(description, pitch_number) AS result_description,
               round(sum(delta_run_exp), 3)       AS run_value
        FROM fact_pitch
        WHERE ($pitcher IS NULL OR pitcher = $pitcher)
          AND ($batter  IS NULL OR batter  = $batter)
          AND ($game_pk IS NULL OR game_pk = $game_pk)
          AND ($season  IS NULL OR season  = $season)
        GROUP BY game_pk, at_bat_number
        ORDER BY game_date DESC, game_pk DESC, at_bat_number
        LIMIT $limit
        """,
            {
                "pitcher": pitcher_id,
                "batter": batter_id,
                "game_pk": game_pk,
                "season": season,
                "limit": limit,
            },
        )
        .to_pylist()
    )
    names = _names([r["pitcher"] for r in rows] + [r["batter"] for r in rows])
    for r in rows:
        r["pitcher_name"] = names.get(r["pitcher"])
        r["batter_name"] = names.get(r["batter"])
    return rows


@router.get("/{game_pk}/{at_bat_number}")
def get_atbat(game_pk: int, at_bat_number: int) -> dict[str, Any]:
    """One plate appearance: the header plus every pitch in order.

    `sz_top`/`sz_bot` come back on each pitch as Statcast measured them for that
    delivery — they are a per-pitch estimate of this batter's zone, not a
    constant, and they wobble by an inch or two within one at-bat. The 3D view
    draws one zone for the at-bat, so the header carries the mean; the per-pitch
    values stay on the pitches for anything that needs them.
    """
    require_table("fact_pitch")
    cols = ", ".join(AT_BAT_PITCH_COLUMNS)
    pitches = (
        warehouse()
        .execute(
            f"""
        SELECT {cols} FROM fact_pitch
        WHERE game_pk = $game_pk AND at_bat_number = $at_bat_number
        ORDER BY pitch_number
        """,
            {"game_pk": game_pk, "at_bat_number": at_bat_number},
        )
        .to_pylist()
    )
    if not pitches:
        raise HTTPException(404, f"No at-bat {at_bat_number} in game {game_pk}.")

    head = (
        warehouse()
        .execute(
            """
        SELECT any_value(game_date) AS game_date,
               any_value(season)    AS season,
               any_value(pitcher)   AS pitcher,
               any_value(batter)    AS batter,
               any_value(inning)    AS inning,
               any_value(home_team) AS home_team,
               avg(sz_top)          AS sz_top,
               avg(sz_bot)          AS sz_bot,
               arg_max(events, pitch_number)      AS result,
               arg_max(description, pitch_number) AS result_description,
               round(sum(delta_run_exp), 3)       AS run_value
        FROM fact_pitch
        WHERE game_pk = $game_pk AND at_bat_number = $at_bat_number
        """,
            {"game_pk": game_pk, "at_bat_number": at_bat_number},
        )
        .to_pylist()[0]
    )
    names = _names([head["pitcher"], head["batter"]])
    return {
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        **head,
        "pitcher_name": names.get(head["pitcher"]),
        "batter_name": names.get(head["batter"]),
        "p_throws": pitches[0]["p_throws"],
        "stand": pitches[0]["stand"],
        "pitches": pitches,
    }
