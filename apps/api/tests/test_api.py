"""API contract tests. Run against whatever the local pipeline has built."""

from __future__ import annotations

import io
import math

import pyarrow.ipc as ipc
import pytest
from fastapi.testclient import TestClient

from bbapi.arrow import ARROW_MEDIA_TYPE
from bbapi.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def health(client: TestClient) -> dict:
    return client.get("/health").json()


def _needs(health: dict, table: str) -> None:
    if not health["tables"].get(table):
        pytest.skip(f"{table} not built — run the pipeline first")


@pytest.fixture(scope="module")
def a_pitcher(client: TestClient, health: dict) -> int:
    _needs(health, "mart_pitcher_arsenal")
    rows = client.get("/players", params={"limit": 1, "min_pitches": 1}).json()
    if not rows:
        pytest.skip("no pitchers in mart")
    return rows[0]["mlbam_id"]


class TestMeta:
    def test_health_reports_pipeline_state(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "fact_pitch" in body["tables"]

    def test_seasons(self, client, health):
        _needs(health, "fact_pitch")
        rows = client.get("/seasons").json()
        assert rows
        assert rows[0]["pitches"] > 0


class TestPlayers:
    def test_search_requires_two_characters(self, client):
        assert client.get("/players/search", params={"q": "a"}).status_code == 422

    def test_search_finds_a_known_player(self, client, health):
        _needs(health, "dim_player")
        rows = client.get("/players/search", params={"q": "Kershaw"}).json()
        assert any("Kershaw" in r["full_name"] for r in rows)

    def test_unknown_player_is_404(self, client, health):
        _needs(health, "dim_player")
        assert client.get("/players/1/profile").status_code == 404

    def test_profile_shape(self, client, health, a_pitcher):
        body = client.get(f"/players/{a_pitcher}/profile").json()
        assert body["player"]["mlbam_id"] == a_pitcher
        assert isinstance(body["seasons"], list)

    def test_arsenal_usage_sums_to_100(self, client, health, a_pitcher):
        """Guards the window function in the mart: a broken PARTITION BY would
        still return plausible-looking rows."""
        rows = client.get(f"/players/{a_pitcher}/arsenal").json()
        assert rows
        by_season: dict[int, float] = {}
        for r in rows:
            by_season[r["season"]] = by_season.get(r["season"], 0) + r["usage_pct"]
        for total in by_season.values():
            assert total == pytest.approx(100.0, abs=0.01)

    def test_external_ids(self, client, health, a_pitcher):
        _needs(health, "dim_player_ids")
        body = client.get(f"/players/{a_pitcher}/ids").json()
        assert body["mlbam_id"] == a_pitcher
        assert "key_fangraphs" in body


class TestArrowPayloads:
    def test_pitches_requires_a_filter(self, client, health):
        """An unfiltered scan of 7.7M rows is never what a chart wants."""
        _needs(health, "fact_pitch")
        assert client.get("/pitches").status_code == 400

    def test_pitches_round_trip_as_arrow(self, client, health, a_pitcher):
        _needs(health, "fact_pitch")
        r = client.get("/pitches", params={"pitcher_id": a_pitcher})
        assert r.status_code == 200
        assert r.headers["content-type"] == ARROW_MEDIA_TYPE

        reader = ipc.open_stream(io.BytesIO(r.content))
        table = reader.read_all()
        assert table.num_rows == int(r.headers["X-Row-Count"])
        assert table.num_rows > 0
        for col in ("pitch_type", "release_speed", "ivb_in", "hb_arm_in", "plate_z_norm"):
            assert col in table.column_names

    def test_arrow_is_substantially_smaller_than_json(self, client, health, a_pitcher):
        """The reason this format was chosen. If the gap ever closes, the
        complexity is no longer justified."""
        import json

        r = client.get("/pitches", params={"pitcher_id": a_pitcher})
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        if table.num_rows < 200:
            pytest.skip("sample too small for a meaningful comparison")
        json_size = len(json.dumps(table.to_pylist(), default=str).encode())
        assert len(r.content) < json_size / 2

    def test_season_filter_narrows_results(self, client, health, a_pitcher):
        _needs(health, "fact_pitch")
        allr = client.get("/pitches", params={"pitcher_id": a_pitcher})
        one = client.get("/pitches", params={"pitcher_id": a_pitcher, "season": 1999})
        assert int(one.headers["X-Row-Count"]) == 0
        assert int(allr.headers["X-Row-Count"]) > 0

    def test_completed_seasons_are_cached_hard(self, client, health, a_pitcher):
        r = client.get("/pitches", params={"pitcher_id": a_pitcher, "season": 2015})
        assert "max-age=2592000" in r.headers.get("cache-control", "")


def _solve_t(y0: float, vy0: float, ay: float, y_target: float) -> float:
    """Time at which y(t) = y_target, given y(t) = y0 + vy0*t + 0.5*ay*t^2.

    Two roots exist; the physically meaningful one for a pitch (release just a
    few hundredths of a second from the y=50 reference, flight well under a
    second) is always the one with the smaller magnitude.
    """
    a, b, c = 0.5 * ay, vy0, y0 - y_target
    disc = b * b - 4 * a * c
    r1 = (-b + math.sqrt(disc)) / (2 * a)
    r2 = (-b - math.sqrt(disc)) / (2 * a)
    return r1 if abs(r1) < abs(r2) else r2


class TestTrajectory:
    """The physics reconstruction the 3D pitch trajectory viz depends on."""

    def test_returns_physics_params_for_a_known_pitch(self, client, health, a_pitcher):
        _needs(health, "fact_pitch")
        r = client.get("/pitches", params={"pitcher_id": a_pitcher, "limit": 1})
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        if table.num_rows == 0:
            pytest.skip("sampled pitcher has no pitches")
        row = table.to_pylist()[0]
        got = client.get(
            "/pitches/trajectory",
            params={
                "game_pk": row["game_pk"],
                "at_bat_number": row["at_bat_number"],
                "pitch_number": row["pitch_number"],
            },
        )
        if got.status_code == 404:
            pytest.skip("sampled pitch has no tracked physics params")
        assert got.status_code == 200
        body = got.json()
        for key in (
            "vx0",
            "vy0",
            "vz0",
            "ax",
            "ay",
            "az",
            "release_pos_x",
            "release_pos_y",
            "release_pos_z",
        ):
            assert body[key] is not None

    def test_reconstructed_flight_matches_actual_plate_crossing(self, client, health, a_pitcher):
        """vx0/vy0/vz0/ax/ay/az are valid at y=50ft, not at release. Integrating
        backward to release then forward to y=17/12 must land on the real
        plate_x/plate_z Savant separately measured — that agreement is the
        whole justification for treating this as exact, not approximate."""
        _needs(health, "fact_pitch")
        r = client.get("/pitches", params={"pitcher_id": a_pitcher, "limit": 50})
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        checked = 0
        for row in table.to_pylist():
            got = client.get(
                "/pitches/trajectory",
                params={
                    "game_pk": row["game_pk"],
                    "at_bat_number": row["at_bat_number"],
                    "pitch_number": row["pitch_number"],
                },
            )
            if got.status_code != 200:
                continue
            b = got.json()
            t_r = _solve_t(50.0, b["vy0"], b["ay"], b["release_pos_y"])
            vx_r = b["vx0"] + b["ax"] * t_r
            vy_r = b["vy0"] + b["ay"] * t_r
            vz_r = b["vz0"] + b["az"] * t_r
            tau = _solve_t(b["release_pos_y"], vy_r, b["ay"], 17 / 12)
            px = b["release_pos_x"] + vx_r * tau + 0.5 * b["ax"] * tau * tau
            pz = b["release_pos_z"] + vz_r * tau + 0.5 * b["az"] * tau * tau
            assert abs(px - b["plate_x"]) < 0.05
            assert abs(pz - b["plate_z"]) < 0.05
            checked += 1
            if checked >= 10:
                break
        if checked == 0:
            pytest.skip("no tracked pitches in sample")


class TestZones:
    def test_extent_is_self_describing(self, client):
        ext = client.get("/zones/extent").json()
        assert ext["grid_n"] > 0
        assert ext["x_min"] < ext["x_max"]
        assert "min_reliable_n" in ext

    def test_rejects_unknown_metric(self, client, health):
        _needs(health, "mart_zone_profile")
        r = client.get("/zones/1", params={"metric": "bogus"})
        assert r.status_code == 400

    def test_rejects_unknown_role(self, client, health):
        _needs(health, "mart_zone_profile")
        r = client.get("/zones/1", params={"role": "umpire"})
        assert r.status_code == 400

    def test_grid_ships_surface_and_reliability_together(self, client, health):
        """Both arrays are required: the surface colours the cell, the
        reliability decides how much to fade it."""
        _needs(health, "mart_zone_profile")
        # Find any player that has a grid.
        rows = client.get("/players", params={"limit": 25, "min_pitches": 1}).json()
        for row in rows:
            got = client.get(
                f"/zones/{row['mlbam_id']}", params={"role": "pitcher", "metric": "whiff"}
            )
            if got.status_code == 200:
                body = got.json()
                n = body["grid_n"]
                assert len(body["surface"]) == n * n
                assert len(body["reliability"]) == n * n
                assert body["layout"] == "row_major_x_then_z"
                return
        pytest.skip("no zone grids built for the sampled players")


def _model_available() -> bool:
    from bbcore.config import get_settings
    from bbml.registry import latest_dir

    return latest_dir("next_pitch", settings=get_settings()) is not None


class TestPredict:
    def test_next_pitch_returns_a_distribution(self, client, health, a_pitcher):
        if not _model_available():
            pytest.skip("no next-pitch model registered — run `bb-ml next-pitch` first")
        r = client.post(
            "/predict/next-pitch",
            json={
                "pitcher_id": a_pitcher,
                "batter_id": a_pitcher,
                "season": 2025,
                "balls": 0,
                "strikes": 2,
                "stand": "R",
                "p_throws": "R",
                "home_team": "NYY",
            },
        )
        assert r.status_code == 200
        body = r.json()
        probs = [row["probability"] for row in body["pitch_type"]]
        assert probs == sorted(probs, reverse=True)
        assert sum(probs) <= 1.0 + 1e-6

    def test_replay_pairs_actual_with_predicted(self, client, health):
        if not _model_available():
            pytest.skip("no next-pitch model registered — run `bb-ml next-pitch` first")
        rows = client.get(
            "/players", params={"limit": 1, "min_pitches": 200, "season": 2025}
        ).json()
        if not rows:
            pytest.skip("no qualifying pitcher for a replay")
        pitcher_id = rows[0]["mlbam_id"]
        games = client.get(f"/players/{pitcher_id}/games", params={"limit": 1}).json()
        if not games:
            pytest.skip("no games found for the sampled pitcher")
        r = client.get(f"/games/{games[0]['game_pk']}/replay", params={"pitcher_id": pitcher_id})
        assert r.status_code == 200
        pitches = r.json()
        assert pitches
        first = pitches[0]
        assert first["actual_pitch_type"]
        assert first["predicted_pitch_type"]
        assert "actual_location_class" in first
