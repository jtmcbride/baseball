"""Baseball Savant Statcast client — the spine of the dataset.

Why this talks to the CSV endpoint directly rather than going through pybaseball:
we need raw landing, resumable partitions, explicit rate limiting, and — above all
— the saturation guard below, none of which the library exposes.

THE 25,000-ROW CAP
------------------
`statcast_search/csv` caps a response at 25,000 rows (24,999 data rows + header).
It does not error, does not set a header, and does not emit a marker. It returns
HTTP 200 and simply stops mid-day. A 14-day request verified against the live
endpoint returns exactly 24,999 rows ending partway through day 8.

That is the worst possible failure mode here: a backfill that looks successful and
silently drops a third of its pitches, poisoning every statistic and model
downstream with no visible symptom. Hence `SATURATION_THRESHOLD` and
`_fetch_by_team` — and hence day-sized partitions, which run ~4,400 pitches and
leave ~5.6x headroom.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.http import RateLimitedClient
from bbetl.manifest import Manifest, RunRecord, hash_url, sha256_bytes

log = get_logger(__name__)

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# The endpoint returns at most 25,000 lines including the header.
SAVANT_ROW_CAP = 24_999
# Trip the guard slightly early; an exact-cap match is not the only way to saturate.
SATURATION_THRESHOLD = 24_900

SOURCE = "statcast"

# All 30 clubs, used only as the subdivision axis when a partition saturates.
TEAM_CODES = [
    "AZ",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CWS",
    "CIN",
    "CLE",
    "COL",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "OAK",
    "PHI",
    "PIT",
    "SD",
    "SF",
    "SEA",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSH",
]

# Statcast's natural key. `sv_id` is unreliable and must not be used for this.
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]

# Columns whose inferred dtype is wrong or unstable across seasons. Everything
# else is inferred. `player_name` etc. stay strings; the numeric ones below are
# the ones polars would otherwise guess inconsistently between partitions, which
# breaks the Parquet schema union at read time.
SCHEMA_OVERRIDES: dict[str, type[pl.DataType]] = {
    "game_pk": pl.Int64,
    "batter": pl.Int64,
    "pitcher": pl.Int64,
    "at_bat_number": pl.Int32,
    "pitch_number": pl.Int32,
    "balls": pl.Int8,
    "strikes": pl.Int8,
    "outs_when_up": pl.Int8,
    "inning": pl.Int8,
    "zone": pl.Int8,
    "on_1b": pl.Int64,
    "on_2b": pl.Int64,
    "on_3b": pl.Int64,
    "hit_location": pl.Int8,
    "launch_speed_angle": pl.Int8,
    "umpire": pl.Int64,
    "spin_dir": pl.Float64,
    "spin_rate_deprecated": pl.Float64,
    "break_angle_deprecated": pl.Float64,
    "break_length_deprecated": pl.Float64,
    "tfs_deprecated": pl.Float64,
    "tfs_zulu_deprecated": pl.Utf8,
    # Bat/swing tracking: null before 2024 (bat_speed, swing_length) and before
    # 2025 (attack_*, swing_path_tilt). Pinning them to Float64 keeps the schema
    # stable across the whole 2015+ lake instead of flipping to Null on old
    # partitions, which is what makes `union_by_name` reads work.
    "bat_speed": pl.Float64,
    "swing_length": pl.Float64,
    "attack_angle": pl.Float64,
    "attack_direction": pl.Float64,
    "swing_path_tilt": pl.Float64,
    "intercept_ball_minus_batter_pos_x_inches": pl.Float64,
    "intercept_ball_minus_batter_pos_y_inches": pl.Float64,
    "arm_angle": pl.Float64,
    "miss_distance": pl.Float64,
    "hyper_speed": pl.Float64,
}


@dataclass(frozen=True)
class DayResult:
    game_date: dt.date
    frame: pl.DataFrame
    saturated_subdivided: bool
    raw_path: Path | None


class SavantClient:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = RateLimitedClient(
            rps=self.settings.savant_rps,
            user_agent=self.settings.user_agent,
            timeout_s=self.settings.savant_timeout_s,
            max_retries=self.settings.savant_max_retries,
        )

    # --- low level -----------------------------------------------------------

    @staticmethod
    def _params(start: dt.date, end: dt.date, *, team: str | None = None) -> dict[str, str]:
        params = {
            "all": "true",
            "type": "details",
            "game_date_gt": start.isoformat(),
            "game_date_lt": end.isoformat(),
            "hfSea": f"{start.year}|",
            "min_pitches": "0",
            "min_results": "0",
            "group_by": "name",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
        }
        if team:
            params["hfTeam"] = f"{team}|"
        return params

    def _get_csv(self, start: dt.date, end: dt.date, *, team: str | None = None) -> bytes:
        resp = self.client.get(SAVANT_CSV_URL, params=self._params(start, end, team=team))
        return resp.content

    @staticmethod
    def _parse(payload: bytes) -> pl.DataFrame:
        if not payload.strip():
            return pl.DataFrame()
        df = pl.read_csv(
            io.BytesIO(payload),
            schema_overrides=SCHEMA_OVERRIDES,
            null_values=["null", "NA", ""],
            infer_schema_length=20_000,
            truncate_ragged_lines=True,
        )
        # The endpoint emits a UTF-8 BOM on the first header cell.
        if df.width and df.columns[0].startswith("﻿"):
            df = df.rename({df.columns[0]: df.columns[0].lstrip("﻿")})
        return df

    # --- day fetch with saturation guard -------------------------------------

    def fetch_day(self, game_date: dt.date) -> tuple[pl.DataFrame, bool]:
        """Fetch one game date. Returns (frame, was_subdivided).

        Subdivides by team if the response comes back at the row cap, so a freak
        high-volume day cannot silently truncate.
        """
        payload = self._get_csv(game_date, game_date)
        df = self._parse(payload)

        if df.height < SATURATION_THRESHOLD:
            return df, False

        log.warning(
            "%s returned %d rows (>= saturation threshold %d) — subdividing by team",
            game_date,
            df.height,
            SATURATION_THRESHOLD,
        )
        return self._fetch_by_team(game_date), True

    def _fetch_by_team(self, game_date: dt.date) -> pl.DataFrame:
        """Fallback path: pull each club's slice and dedupe on the pitch key.

        `hfTeam` returns one side of each matchup, so every pitch is covered
        exactly once across all 30 clubs — but dedupe anyway rather than trust it.
        """
        frames: list[pl.DataFrame] = []
        for team in TEAM_CODES:
            part = self._parse(self._get_csv(game_date, game_date, team=team))
            if part.height >= SATURATION_THRESHOLD:
                raise RuntimeError(
                    f"Partition {game_date} team={team} still saturated at {part.height} rows. "
                    "Subdivision axis exhausted — a finer split is required before trusting "
                    "this date."
                )
            if part.height:
                frames.append(part)
        if not frames:
            return pl.DataFrame()
        combined = pl.concat(frames, how="diagonal_relaxed")
        return combined.unique(subset=PITCH_KEY, keep="first")

    # --- raw landing ---------------------------------------------------------

    def raw_path_for(self, game_date: dt.date) -> Path:
        return (
            self.settings.raw_dir
            / SOURCE
            / f"season={game_date.year}"
            / f"{game_date.isoformat()}.csv.gz"
        )

    def land_raw(self, game_date: dt.date, df: pl.DataFrame) -> tuple[Path, str]:
        """Write the parsed frame back out as gzipped CSV.

        Landing raw is what makes the ~3h backfill a one-time cost: every later
        schema change or transform bug becomes a local reprocess instead of
        another crawl.
        """
        path = self.raw_path_for(game_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        df.write_csv(buf)
        payload = buf.getvalue()
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(payload)
        return path, sha256_bytes(payload)

    @staticmethod
    def read_raw(path: Path) -> pl.DataFrame:
        with gzip.open(path, "rb") as fh:
            return SavantClient._parse(fh.read())

    def close(self) -> None:
        self.client.close()


# --- orchestration -----------------------------------------------------------


def game_dates(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Every calendar date in range that could plausibly host a game.

    Feb-Nov only: skipping the offseason removes ~40% of otherwise-wasted requests
    across an 11-season backfill. Spring training and postseason both fall inside
    this window.
    """
    cur = start
    while cur <= end:
        if 2 <= cur.month <= 11:
            yield cur
        cur += dt.timedelta(days=1)


def ingest_range(
    start: dt.date,
    end: dt.date,
    *,
    settings: Settings | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
    client: SavantClient | None = None,
) -> dict[str, int]:
    """Idempotent day-by-day backfill. Safe to interrupt and re-run."""
    s = settings or get_settings()
    mf = manifest or Manifest(settings=s)
    own_client = client is None
    cl = client or SavantClient(settings=s)

    stats = {"fetched": 0, "skipped": 0, "empty": 0, "failed": 0, "rows": 0, "subdivided": 0}
    dates = list(game_dates(start, end))
    log.info("statcast backfill %s..%s — %d candidate dates", start, end, len(dates))

    try:
        for i, day in enumerate(dates, 1):
            key = day.isoformat()
            if not force and mf.is_done(SOURCE, key):
                stats["skipped"] += 1
                continue

            try:
                df, subdivided = cl.fetch_day(day)
            except Exception as exc:
                log.error("%s failed: %s", key, exc)
                mf.record(RunRecord(SOURCE, key, "failed", error=str(exc)[:500]))
                stats["failed"] += 1
                continue

            if df.height == 0:
                mf.record(RunRecord(SOURCE, key, "empty", row_count=0))
                stats["empty"] += 1
                continue

            path, digest = cl.land_raw(day, df)
            mf.record(
                RunRecord(
                    source=SOURCE,
                    partition_key=key,
                    status="ok",
                    row_count=df.height,
                    content_sha256=digest,
                    url_hash=hash_url(SAVANT_CSV_URL, cl._params(day, day)),
                    raw_path=str(path.relative_to(s.data_root)),
                )
            )
            stats["fetched"] += 1
            stats["rows"] += df.height
            stats["subdivided"] += int(subdivided)

            if i % 25 == 0 or subdivided:
                log.info(
                    "[%d/%d] %s  rows=%-6d  total=%d", i, len(dates), key, df.height, stats["rows"]
                )
    finally:
        if own_client:
            cl.close()

    log.info("statcast backfill done: %s", stats)
    return stats


def refresh_recent(
    *,
    settings: Settings | None = None,
    manifest: Manifest | None = None,
    days: int | None = None,
) -> dict[str, int]:
    """Re-pull a trailing window.

    Savant revises published data — pitch classifications get corrected and xwOBA
    recomputed for days already ingested. Append-only ingest therefore goes stale
    in place, which is invisible until a model trains on it.
    """
    s = settings or get_settings()
    mf = manifest or Manifest(settings=s)
    window = days if days is not None else s.refresh_window_days
    end = dt.date.today()
    start = end - dt.timedelta(days=window)
    keys = [d.isoformat() for d in game_dates(start, end)]
    dropped = mf.invalidate(SOURCE, keys)
    log.info("invalidated %d manifest rows across trailing %d days", dropped, window)
    return ingest_range(start, end, settings=s, manifest=mf)
