"""Ingest manifest — what makes every job idempotent and resumable.

One row per (source, partition_key). A completed partition is skipped on re-run
unless `--force`. This is what turns a 3-hour backfill into a one-time cost: an
interrupted run resumes, and a transform bug is a local reprocess rather than
another crawl.

Stored in its own SQLite file rather than the DuckDB warehouse on purpose — the
warehouse gets rebuilt from the lake routinely, and losing the crawl bookkeeping
during a rebuild would force a re-crawl.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bbcore.config import Settings, get_settings

Status = Literal["ok", "empty", "failed", "partial"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_runs (
    source          TEXT NOT NULL,
    partition_key   TEXT NOT NULL,
    status          TEXT NOT NULL,
    row_count       INTEGER,
    content_sha256  TEXT,
    url_hash        TEXT,
    raw_path        TEXT,
    error           TEXT,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (source, partition_key)
);
CREATE INDEX IF NOT EXISTS idx_ingest_status ON ingest_runs(source, status);
"""


@dataclass(frozen=True)
class RunRecord:
    source: str
    partition_key: str
    status: Status
    row_count: int | None = None
    content_sha256: str | None = None
    url_hash: str | None = None
    raw_path: str | None = None
    error: str | None = None


class Manifest:
    def __init__(self, path: Path | None = None, *, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self.path = path or (s.data_root / "ingest_manifest.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30.0)
        try:
            con.execute("PRAGMA journal_mode = WAL")
            yield con
            con.commit()
        finally:
            con.close()

    def is_done(self, source: str, partition_key: str) -> bool:
        """`empty` counts as done — a day with no games is a legitimate result,
        and re-fetching it forever would waste most of a backfill's budget."""
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM ingest_runs WHERE source = ? AND partition_key = ?",
                (source, partition_key),
            ).fetchone()
        return row is not None and row[0] in ("ok", "empty")

    def record(self, rec: RunRecord) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO ingest_runs
                    (source, partition_key, status, row_count, content_sha256,
                     url_hash, raw_path, error, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, partition_key) DO UPDATE SET
                    status = excluded.status,
                    row_count = excluded.row_count,
                    content_sha256 = excluded.content_sha256,
                    url_hash = excluded.url_hash,
                    raw_path = excluded.raw_path,
                    error = excluded.error,
                    fetched_at = excluded.fetched_at
                """,
                (
                    rec.source,
                    rec.partition_key,
                    rec.status,
                    rec.row_count,
                    rec.content_sha256,
                    rec.url_hash,
                    rec.raw_path,
                    rec.error,
                    dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                ),
            )

    def invalidate(self, source: str, partition_keys: list[str]) -> int:
        """Drop bookkeeping so the next run re-fetches. Used by --force and by the
        nightly trailing-window refresh (Savant revises data after publication)."""
        if not partition_keys:
            return 0
        marks = ",".join("?" * len(partition_keys))
        with self._connect() as con:
            cur = con.execute(
                f"DELETE FROM ingest_runs WHERE source = ? AND partition_key IN ({marks})",
                (source, *partition_keys),
            )
            return cur.rowcount

    def summary(self, source: str | None = None) -> list[tuple[str, str, int, int]]:
        q = (
            "SELECT source, status, count(*), coalesce(sum(row_count), 0) "
            "FROM ingest_runs {where} GROUP BY source, status ORDER BY source, status"
        )
        with self._connect() as con:
            if source:
                return con.execute(q.format(where="WHERE source = ?"), (source,)).fetchall()
            return con.execute(q.format(where="")).fetchall()

    def failures(self, source: str) -> list[tuple[str, str]]:
        with self._connect() as con:
            return con.execute(
                "SELECT partition_key, error FROM ingest_runs "
                "WHERE source = ? AND status = 'failed' ORDER BY partition_key",
                (source,),
            ).fetchall()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_url(url: str, params: dict[str, object] | None = None) -> str:
    canonical = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
