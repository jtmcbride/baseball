"""MLB Stats API client — official, free, and not a scrape.

Supplies the dimensions Statcast lacks: game/venue metadata, player biography and
handedness, team identity, and umpires.

On umpires specifically: Statcast ships an `umpire` column, but it is empty on
every row in every season (verified 2015 and 2025). The home-plate umpire is a
real input to a called-strike model and to next-pitch features, so it has to come
from this API's boxscore `officials` block instead. That costs one request per
game, so it is a separate opt-in job rather than part of the M1 path.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Iterator
from typing import Any

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.http import RateLimitedClient
from bbetl.manifest import Manifest, RunRecord

log = get_logger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1  # MLB

# `people` accepts a batched id list, which turns ~2,500 player lookups into ~25
# requests. Kept below 100 to stay clear of URL length limits.
PEOPLE_BATCH = 90


class StatsAPIClient:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = RateLimitedClient(
            rps=self.settings.statsapi_rps,
            user_agent=self.settings.user_agent,
            timeout_s=60.0,
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.client.get(f"{BASE}{path}", params=params)
        return resp.json()

    # --- schedule / games ----------------------------------------------------

    def schedule(self, start: dt.date, end: dt.date, game_types: str = "R,F,D,L,W,S") -> list[dict]:
        """All games in a date range. One request covers a whole range."""
        payload = self._get(
            "/schedule",
            {
                "sportId": SPORT_ID,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "gameTypes": game_types,
            },
        )
        return [g for d in payload.get("dates", []) for g in d.get("games", [])]

    def boxscore_officials(self, game_pk: int) -> dict[str, Any]:
        bs = self._get(f"/game/{game_pk}/boxscore")
        out: dict[str, Any] = {"game_pk": game_pk}
        for off in bs.get("officials", []):
            role = off.get("officialType", "").lower().replace(" ", "_")
            out[f"umpire_{role}_id"] = off["official"]["id"]
            out[f"umpire_{role}_name"] = off["official"]["fullName"]
        return out

    # --- people --------------------------------------------------------------

    def people(self, person_ids: Iterable[int]) -> list[dict]:
        ids = [int(p) for p in person_ids]
        rows: list[dict] = []
        for chunk in _chunks(ids, PEOPLE_BATCH):
            payload = self._get("/people", {"personIds": ",".join(map(str, chunk))})
            rows.extend(payload.get("people", []))
        return rows

    def teams(self, season: int) -> list[dict]:
        return self._get("/teams", {"sportId": SPORT_ID, "season": season}).get("teams", [])

    def close(self) -> None:
        self.client.close()


def _chunks(seq: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --- shaping -----------------------------------------------------------------


def games_to_frame(games: list[dict]) -> pl.DataFrame:
    rows = [
        {
            "game_pk": g["gamePk"],
            "game_date": g.get("officialDate"),
            "game_type": g.get("gameType"),
            "season": int(g["season"]) if g.get("season") else None,
            "day_night": g.get("dayNight"),
            "double_header": g.get("doubleHeader"),
            "game_number": g.get("gameNumber"),
            "venue_id": g.get("venue", {}).get("id"),
            "venue_name": g.get("venue", {}).get("name"),
            "status": g.get("status", {}).get("detailedState"),
            "home_team_id": g.get("teams", {}).get("home", {}).get("team", {}).get("id"),
            "away_team_id": g.get("teams", {}).get("away", {}).get("team", {}).get("id"),
            "home_score": g.get("teams", {}).get("home", {}).get("score"),
            "away_score": g.get("teams", {}).get("away", {}).get("score"),
        }
        for g in games
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        pl.col("game_date").cast(pl.Date, strict=False),
        pl.col("game_pk").cast(pl.Int64),
        pl.col("season").cast(pl.Int16),
    )


def people_to_frame(people: list[dict]) -> pl.DataFrame:
    rows = [
        {
            "mlbam_id": p["id"],
            "full_name": p.get("fullName"),
            "first_name": p.get("firstName"),
            "last_name": p.get("lastName"),
            "bats": (p.get("batSide") or {}).get("code"),
            "throws": (p.get("pitchHand") or {}).get("code"),
            "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
            "position_type": (p.get("primaryPosition") or {}).get("type"),
            "birth_date": p.get("birthDate"),
            "birth_country": p.get("birthCountry"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "mlb_debut_date": p.get("mlbDebutDate"),
            "active": p.get("active"),
        }
        for p in people
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        pl.col("mlbam_id").cast(pl.Int64),
        pl.col("birth_date").cast(pl.Date, strict=False),
        pl.col("mlb_debut_date").cast(pl.Date, strict=False),
    )


def teams_to_frame(teams: list[dict]) -> pl.DataFrame:
    rows = [
        {
            "team_id": t["id"],
            "abbreviation": t.get("abbreviation"),
            "name": t.get("name"),
            "team_code": t.get("teamCode"),
            "franchise_name": t.get("franchiseName"),
            "club_name": t.get("clubName"),
            "league": (t.get("league") or {}).get("name"),
            "division": (t.get("division") or {}).get("name"),
            "venue_id": (t.get("venue") or {}).get("id"),
            "venue_name": (t.get("venue") or {}).get("name"),
            "season": int(t["season"]) if t.get("season") else None,
        }
        for t in teams
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        pl.col("team_id").cast(pl.Int64), pl.col("season").cast(pl.Int16)
    )


# --- jobs --------------------------------------------------------------------


def ingest_games(
    start: dt.date,
    end: dt.date,
    *,
    settings: Settings | None = None,
    client: StatsAPIClient | None = None,
) -> int:
    """Build `dim_game` for a date range. Cheap: one request per season slice."""
    s = settings or get_settings()
    cl = client or StatsAPIClient(settings=s)
    owned = client is None
    try:
        frames = []
        for season in range(start.year, end.year + 1):
            lo = max(start, dt.date(season, 1, 1))
            hi = min(end, dt.date(season, 12, 31))
            games = cl.schedule(lo, hi)
            log.info("schedule %s..%s -> %d games", lo, hi, len(games))
            if games:
                frames.append(games_to_frame(games))
        if not frames:
            return 0
        df = pl.concat(frames, how="diagonal_relaxed").unique(subset=["game_pk"])
        _write_dim(df, "dim_game", s)
        return df.height
    finally:
        if owned:
            cl.close()


def ingest_people(
    person_ids: Iterable[int],
    *,
    settings: Settings | None = None,
    client: StatsAPIClient | None = None,
) -> int:
    """Build `dim_player` for the given MLBAM ids (normally taken from fact_pitch)."""
    s = settings or get_settings()
    cl = client or StatsAPIClient(settings=s)
    owned = client is None
    try:
        ids = sorted({int(p) for p in person_ids})
        log.info("fetching %d people in batches of %d", len(ids), PEOPLE_BATCH)
        df = people_to_frame(cl.people(ids))
        if df.height:
            _write_dim(df, "dim_player", s)
        return df.height
    finally:
        if owned:
            cl.close()


def ingest_teams(
    seasons: list[int], *, settings: Settings | None = None, client: StatsAPIClient | None = None
) -> int:
    s = settings or get_settings()
    cl = client or StatsAPIClient(settings=s)
    owned = client is None
    try:
        frames = [teams_to_frame(cl.teams(yr)) for yr in seasons]
        frames = [f for f in frames if f.height]
        if not frames:
            return 0
        df = pl.concat(frames, how="diagonal_relaxed").unique(subset=["team_id", "season"])
        _write_dim(df, "dim_team", s)
        return df.height
    finally:
        if owned:
            cl.close()


def ingest_officials(
    game_pks: Iterable[int],
    *,
    settings: Settings | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
) -> int:
    """One request per game — run this only when the umpire models need it.

    Statcast's own `umpire` column is empty, so this is the only source.
    """
    s = settings or get_settings()
    mf = manifest or Manifest(settings=s)
    cl = StatsAPIClient(settings=s)
    out_dir = s.raw_dir / "officials"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        for pk in game_pks:
            key = str(pk)
            if not force and mf.is_done("officials", key):
                continue
            try:
                rec = cl.boxscore_officials(int(pk))
                (out_dir / f"{pk}.json").write_text(json.dumps(rec))
                mf.record(RunRecord("officials", key, "ok", row_count=1))
                n += 1
            except Exception as exc:
                mf.record(RunRecord("officials", key, "failed", error=str(exc)[:500]))
                log.error("officials %s failed: %s", pk, exc)
    finally:
        cl.close()
    return n


def _write_dim(df: pl.DataFrame, name: str, settings: Settings) -> None:
    out = settings.lake_dir / name
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "part_0.parquet", compression="zstd")
    log.info("wrote %s: %d rows", name, df.height)
