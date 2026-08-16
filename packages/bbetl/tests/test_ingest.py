"""Tests for the ingestion machinery: rate limiting, manifest, saturation guard."""

from __future__ import annotations

import datetime as dt
import time

import polars as pl
import pytest

from bbetl.http import TokenBucket
from bbetl.manifest import Manifest, RunRecord, hash_url
from bbetl.sources import savant
from bbetl.sources.savant import SATURATION_THRESHOLD, SavantClient, game_dates


class TestTokenBucket:
    def test_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError):
            TokenBucket(0)

    def test_throttles_to_the_configured_rate(self):
        bucket = TokenBucket(rps=20.0, burst=1)
        bucket.acquire()  # drains the initial token
        start = time.monotonic()
        for _ in range(3):
            bucket.acquire()
        elapsed = time.monotonic() - start
        # 3 tokens at 20/s ~= 0.15s. Generous lower bound to avoid flakiness.
        assert elapsed >= 0.10


class TestManifest:
    def test_records_and_reports_done(self, tmp_path):
        mf = Manifest(tmp_path / "m.sqlite")
        assert not mf.is_done("statcast", "2025-06-14")
        mf.record(RunRecord("statcast", "2025-06-14", "ok", row_count=4435))
        assert mf.is_done("statcast", "2025-06-14")

    def test_empty_counts_as_done(self, tmp_path):
        """An off-day legitimately has no pitches. Re-fetching it forever would
        waste most of a backfill's request budget."""
        mf = Manifest(tmp_path / "m.sqlite")
        mf.record(RunRecord("statcast", "2025-12-25", "empty", row_count=0))
        assert mf.is_done("statcast", "2025-12-25")

    def test_failed_is_not_done_so_it_retries(self, tmp_path):
        mf = Manifest(tmp_path / "m.sqlite")
        mf.record(RunRecord("statcast", "2025-06-14", "failed", error="timeout"))
        assert not mf.is_done("statcast", "2025-06-14")

    def test_record_is_upsert_not_duplicate(self, tmp_path):
        mf = Manifest(tmp_path / "m.sqlite")
        mf.record(RunRecord("statcast", "2025-06-14", "failed", error="boom"))
        mf.record(RunRecord("statcast", "2025-06-14", "ok", row_count=10))
        rows = mf.summary("statcast")
        assert sum(r[2] for r in rows) == 1
        assert mf.is_done("statcast", "2025-06-14")

    def test_invalidate_forces_refetch(self, tmp_path):
        """Backs the nightly trailing-window refresh: Savant revises published
        data, so ingested days must be able to go stale on purpose."""
        mf = Manifest(tmp_path / "m.sqlite")
        mf.record(RunRecord("statcast", "2025-06-14", "ok", row_count=1))
        assert mf.invalidate("statcast", ["2025-06-14"]) == 1
        assert not mf.is_done("statcast", "2025-06-14")

    def test_hash_url_is_order_independent(self):
        assert hash_url("u", {"a": 1, "b": 2}) == hash_url("u", {"b": 2, "a": 1})


class TestGameDates:
    def test_skips_the_offseason(self):
        days = list(game_dates(dt.date(2025, 1, 1), dt.date(2025, 12, 31)))
        assert all(2 <= d.month <= 11 for d in days)
        assert dt.date(2025, 1, 15) not in days
        assert dt.date(2025, 6, 14) in days

    def test_includes_spring_training_and_postseason(self):
        days = list(game_dates(dt.date(2025, 2, 1), dt.date(2025, 11, 30)))
        assert dt.date(2025, 2, 25) in days  # spring training
        assert dt.date(2025, 10, 28) in days  # world series


class TestSaturationGuard:
    """The cap does not error — it returns HTTP 200 with the data silently cut
    short. Verified against the live endpoint: a 14-day request comes back with
    exactly 24,999 rows ending partway through day 8."""

    def _fake_frame(self, n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "game_pk": [1] * n,
                "at_bat_number": list(range(n)),
                "pitch_number": [1] * n,
            }
        )

    def test_normal_day_does_not_subdivide(self, monkeypatch):
        client = SavantClient.__new__(SavantClient)
        monkeypatch.setattr(SavantClient, "_get_csv", lambda self, *a, **k: b"", raising=True)
        monkeypatch.setattr(
            SavantClient, "_parse", staticmethod(lambda payload: self_frame), raising=True
        )
        self_frame = self._fake_frame(4435)
        monkeypatch.setattr(
            SavantClient, "_parse", staticmethod(lambda payload: self_frame), raising=True
        )
        df, subdivided = SavantClient.fetch_day(client, dt.date(2025, 6, 14))
        assert not subdivided
        assert df.height == 4435

    def test_saturated_day_triggers_subdivision(self, monkeypatch):
        client = SavantClient.__new__(SavantClient)
        big = self._fake_frame(SATURATION_THRESHOLD + 10)
        small = self._fake_frame(50)

        monkeypatch.setattr(SavantClient, "_get_csv", lambda self, *a, **k: b"", raising=True)
        monkeypatch.setattr(SavantClient, "_parse", staticmethod(lambda payload: big), raising=True)
        called = {"n": 0}

        def fake_by_team(self, game_date):
            called["n"] += 1
            return small

        monkeypatch.setattr(SavantClient, "_fetch_by_team", fake_by_team, raising=True)
        df, subdivided = SavantClient.fetch_day(client, dt.date(2025, 6, 14))
        assert subdivided
        assert called["n"] == 1
        assert df.height == 50

    def test_subdivision_raises_if_still_saturated(self, monkeypatch):
        """Exhausting the subdivision axis must fail loudly rather than return
        data we know to be incomplete."""
        client = SavantClient.__new__(SavantClient)
        big = self._fake_frame(SATURATION_THRESHOLD + 10)
        monkeypatch.setattr(SavantClient, "_get_csv", lambda self, *a, **k: b"", raising=True)
        monkeypatch.setattr(SavantClient, "_parse", staticmethod(lambda payload: big), raising=True)
        with pytest.raises(RuntimeError, match="still saturated"):
            SavantClient._fetch_by_team(client, dt.date(2025, 6, 14))

    def test_subdivision_dedupes_on_the_pitch_key(self, monkeypatch):
        """hfTeam returns one side of each matchup, but overlap must not
        double-count pitches."""
        client = SavantClient.__new__(SavantClient)
        dup = pl.DataFrame({"game_pk": [1, 1], "at_bat_number": [1, 1], "pitch_number": [1, 1]})
        monkeypatch.setattr(SavantClient, "_get_csv", lambda self, *a, **k: b"", raising=True)
        monkeypatch.setattr(SavantClient, "_parse", staticmethod(lambda payload: dup), raising=True)
        out = SavantClient._fetch_by_team(client, dt.date(2025, 6, 14))
        assert out.height == 1


class TestParsing:
    def test_strips_utf8_bom_from_first_column(self):
        """Savant emits a BOM on the header's first cell; unstripped it yields a
        column literally named '\\ufeffpitch_type' and every downstream reference
        to `pitch_type` fails."""
        csv = '﻿"pitch_type","release_speed"\n"FF","95.1"\n'.encode()
        df = SavantClient._parse(csv)
        assert df.columns[0] == "pitch_type"

    def test_empty_payload_returns_empty_frame(self):
        assert SavantClient._parse(b"").height == 0
        assert SavantClient._parse(b"   ").height == 0

    def test_params_pin_the_season_and_dates(self):
        p = SavantClient._params(dt.date(2025, 6, 14), dt.date(2025, 6, 14))
        assert p["game_date_gt"] == "2025-06-14"
        assert p["game_date_lt"] == "2025-06-14"
        assert p["hfSea"] == "2025|"

    def test_team_param_is_added_only_when_subdividing(self):
        assert "hfTeam" not in SavantClient._params(dt.date(2025, 6, 14), dt.date(2025, 6, 14))
        p = SavantClient._params(dt.date(2025, 6, 14), dt.date(2025, 6, 14), team="NYY")
        assert p["hfTeam"] == "NYY|"


def test_team_codes_cover_all_thirty_clubs():
    assert len(savant.TEAM_CODES) == 30
    assert len(set(savant.TEAM_CODES)) == 30
