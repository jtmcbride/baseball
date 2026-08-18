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
        r = client.get("/zones/1", params={"role": "bogus"})
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


class TestSpray:
    """Spray chart (viz #8): per-batted-ball rows plus the smoothed contour."""

    def test_extent_is_self_describing(self, client):
        ext = client.get("/spray/extent").json()
        assert ext["grid_n"] > 0
        assert ext["x_min"] < ext["x_max"]
        assert ext["y_min"] < ext["y_max"]
        assert "min_reliable_n" in ext

    def test_batted_balls_round_trip_as_arrow(self, client, health, graded_batter):
        _needs(health, "fact_pitch")
        r = client.get(f"/spray/{graded_batter}/battedballs")
        assert r.status_code == 200
        assert r.headers["content-type"] == ARROW_MEDIA_TYPE
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        assert table.num_rows == int(r.headers["X-Row-Count"])
        for col in ("x_ft", "y_ft", "estimated_woba_using_speedangle", "home_team"):
            assert col in table.column_names

    def test_unknown_batter_contour_is_404(self, client, health):
        _needs(health, "mart_batter_spray")
        assert client.get("/spray/1/contour").status_code == 404

    def test_contour_ships_surface_and_reliability_together(self, client, health, graded_batter):
        _needs(health, "mart_batter_spray")
        got = client.get(f"/spray/{graded_batter}/contour")
        if got.status_code != 200:
            pytest.skip("this graded batter has no qualifying spray-mart season")
        body = got.json()
        n = body["grid_n"]
        assert len(body["surface"]) == n * n
        assert len(body["reliability"]) == n * n
        assert body["layout"] == "row_major_x_then_y"


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


@pytest.fixture(scope="module")
def graded_pitcher(client: TestClient, health: dict) -> int:
    _needs(health, "mart_pitcher_stuff")
    rows = client.get("/stuff", params={"limit": 1, "min_pitches": 500}).json()
    if not rows:
        pytest.skip("no graded pitcher-seasons above the leaderboard threshold")
    return rows[0]["mlbam_id"]


class TestStuff:
    """Stuff+ / Location+ / Pitching+ routes, backed by mart_pitcher_stuff."""

    def test_arsenal_carries_all_three_grades_plus_a_rollup(self, client, graded_pitcher):
        rows = client.get(f"/stuff/{graded_pitcher}").json()
        assert rows
        for row in rows:
            for grade in ("stuff_plus", "location_plus", "pitching_plus"):
                assert isinstance(row[grade], (int, float))
        # The rollup must lead, so a caller can read a headline off row zero.
        assert rows[0]["pitch_type"] == "ALL"
        assert rows[0]["usage_pct"] == pytest.approx(100.0)

    def test_the_rollup_covers_at_least_its_parts(self, client, graded_pitcher):
        rows = client.get(f"/stuff/{graded_pitcher}").json()
        season = rows[0]["season"]
        same = [r for r in rows if r["season"] == season]
        rollup = next(r for r in same if r["pitch_type"] == "ALL")
        parts = [r for r in same if r["pitch_type"] != "ALL"]
        # Pitch types below the mart's minimum are dropped, so the parts can sum
        # to less than the whole — but never to more.
        assert sum(p["pitches"] for p in parts) <= rollup["pitches"]

    def test_unknown_pitcher_is_404_not_an_empty_list(self, client, health):
        _needs(health, "mart_pitcher_stuff")
        assert client.get("/stuff/1").status_code == 404

    def test_leaderboard_is_sorted_by_the_requested_metric(self, client, health):
        _needs(health, "mart_pitcher_stuff")
        rows = client.get(
            "/stuff", params={"metric": "location_plus", "min_pitches": 500, "limit": 20}
        ).json()
        if len(rows) < 2:
            pytest.skip("not enough qualified pitchers to check ordering")
        values = [r["location_plus"] for r in rows]
        assert values == sorted(values, reverse=True)

    def test_leaderboard_respects_the_sample_size_floor(self, client, health):
        _needs(health, "mart_pitcher_stuff")
        rows = client.get("/stuff", params={"min_pitches": 800, "limit": 50}).json()
        assert all(r["pitches"] >= 800 for r in rows)

    def test_an_arbitrary_metric_cannot_reach_the_order_by(self, client, health):
        """The metric is interpolated into SQL, so the whitelist is load-bearing."""
        _needs(health, "mart_pitcher_stuff")
        r = client.get("/stuff", params={"metric": "pitches; DROP TABLE dim_player"})
        assert r.status_code == 400


@pytest.fixture(scope="module")
def graded_batter(client: TestClient, health: dict) -> int:
    _needs(health, "mart_batter_swing")
    rows = client.get("/swing", params={"limit": 1, "min_swings": 200}).json()
    if not rows:
        pytest.skip("no graded batter-seasons above the leaderboard threshold")
    return rows[0]["mlbam_id"]


class TestSwing:
    """Swing-path plane-value routes, backed by mart_batter_swing."""

    def test_batter_carries_both_heads(self, client, graded_batter):
        rows = client.get(f"/swing/{graded_batter}").json()
        assert rows
        for row in rows:
            assert isinstance(row["whiff_plane_value_per_100"], (int, float))
            assert isinstance(row["attack_angle"], (int, float))

    def test_unknown_batter_is_404_not_an_empty_list(self, client, health):
        _needs(health, "mart_batter_swing")
        assert client.get("/swing/1").status_code == 404

    def test_leaderboard_is_sorted_by_the_requested_metric(self, client, health):
        _needs(health, "mart_batter_swing")
        rows = client.get(
            "/swing", params={"metric": "whiff_plane_value_per_100", "min_swings": 200, "limit": 20}
        ).json()
        if len(rows) < 2:
            pytest.skip("not enough qualified batters to check ordering")
        values = [r["whiff_plane_value_per_100"] for r in rows]
        assert values == sorted(values, reverse=True)

    def test_leaderboard_respects_the_sample_size_floor(self, client, health):
        _needs(health, "mart_batter_swing")
        rows = client.get("/swing", params={"min_swings": 400, "limit": 50}).json()
        assert all(r["whiff_swings"] >= 400 for r in rows)

    def test_an_arbitrary_metric_cannot_reach_the_order_by(self, client, health):
        _needs(health, "mart_batter_swing")
        r = client.get("/swing", params={"metric": "pitches; DROP TABLE dim_player"})
        assert r.status_code == 400

    def test_per_swing_pitches_round_trip_as_arrow(self, client, health, graded_batter):
        """The per-swing route (viz #19) — Arrow IPC, same predicate as the mart."""
        _needs(health, "fact_pitch")
        r = client.get(f"/swing/{graded_batter}/pitches")
        assert r.status_code == 200
        assert r.headers["content-type"] == ARROW_MEDIA_TYPE
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        assert table.num_rows == int(r.headers["X-Row-Count"])
        for col in ("attack_angle", "vaa_deg", "swing_length", "is_whiff", "pitch_type"):
            assert col in table.column_names

    def test_per_swing_pitches_respects_season_filter(self, client, health, graded_batter):
        _needs(health, "fact_pitch")
        rows = client.get(f"/swing/{graded_batter}").json()
        season = rows[0]["season"]
        r = client.get(f"/swing/{graded_batter}/pitches", params={"season": season})
        table = ipc.open_stream(io.BytesIO(r.content)).read_all()
        assert table.num_rows > 0


@pytest.fixture(scope="module")
def graded_catcher(client: TestClient, health: dict) -> int:
    _needs(health, "mart_catcher_framing")
    rows = client.get("/framing/catchers", params={"limit": 1, "min_pitches": 500}).json()
    if not rows:
        pytest.skip("no graded catcher-seasons above the leaderboard threshold")
    return rows[0]["mlbam_id"]


class TestFraming:
    """Catcher framing runs and umpire zone edge, backed by mart_catcher_framing
    and mart_umpire_zone."""

    def test_catcher_detail(self, client, graded_catcher):
        rows = client.get(f"/framing/catchers/{graded_catcher}").json()
        assert rows
        assert isinstance(rows[0]["framing_runs"], (int, float))

    def test_unknown_catcher_is_404_not_an_empty_list(self, client, health):
        _needs(health, "mart_catcher_framing")
        assert client.get("/framing/catchers/1").status_code == 404

    def test_catcher_leaderboard_is_sorted_by_framing_runs(self, client, health):
        _needs(health, "mart_catcher_framing")
        rows = client.get("/framing/catchers", params={"min_pitches": 500, "limit": 20}).json()
        if len(rows) < 2:
            pytest.skip("not enough qualified catchers to check ordering")
        values = [r["framing_runs"] for r in rows]
        assert values == sorted(values, reverse=True)

    def test_umpire_leaderboard_carries_a_name(self, client, health):
        _needs(health, "mart_umpire_zone")
        rows = client.get("/framing/umpires", params={"min_pitches": 500, "limit": 20}).json()
        if not rows:
            pytest.skip("no qualified umpire-seasons")
        assert rows[0]["full_name"]

    def test_umpire_leaderboard_is_sorted_by_edge_magnitude(self, client, health):
        _needs(health, "mart_umpire_zone")
        rows = client.get("/framing/umpires", params={"min_pitches": 500, "limit": 20}).json()
        if len(rows) < 2:
            pytest.skip("not enough qualified umpires to check ordering")
        magnitudes = [abs(r["edge"]) for r in rows]
        assert magnitudes == sorted(magnitudes, reverse=True)


@pytest.fixture(scope="module")
def embedded_pitcher(client: TestClient, health: dict) -> int:
    _needs(health, "mart_arsenal_embedding")
    rows = client.get("/arsenal/embedding", params={"min_pitches": 1}).json()
    if not rows:
        pytest.skip("no embedded pitcher-seasons")
    return rows[0]["mlbam_id"]


class TestArsenal:
    """Model #2's re-derived arsenals (clusters) and model #11's 2D embedding
    plus nearest-neighbor similarity (viz #12)."""

    def test_pitcher_clusters_usage_sums_to_100(self, client, health):
        _needs(health, "mart_pitcher_arsenal_clusters")
        rows = client.get("/players", params={"limit": 1, "min_pitches": 1}).json()
        if not rows:
            pytest.skip("no pitchers")
        # Not every pitcher qualifies for the cluster mart (min_pitches floor,
        # 2020+ only) -- walk the embedding mart instead, which only lists
        # pitchers that do.
        embedded = client.get("/arsenal/embedding", params={"min_pitches": 1}).json()
        if not embedded:
            pytest.skip("no embedded pitcher-seasons")
        pid, season = embedded[0]["mlbam_id"], embedded[0]["season"]
        clusters = client.get(f"/arsenal/{pid}", params={"season": season}).json()
        assert clusters
        total = sum(r["usage_pct"] for r in clusters)
        assert math.isclose(total, 100.0, abs_tol=0.5)

    def test_unknown_pitcher_is_404_not_an_empty_list(self, client, health):
        _needs(health, "mart_pitcher_arsenal_clusters")
        assert client.get("/arsenal/1").status_code == 404

    def test_embedding_route_is_matched_before_the_int_path_param(self, client, health):
        """Regression: `/arsenal/embedding` must resolve to the literal route,
        not attempt to parse "embedding" as `mlbam_id` and 422."""
        _needs(health, "mart_arsenal_embedding")
        r = client.get("/arsenal/embedding")
        assert r.status_code == 200

    def test_embedding_rows_have_finite_coordinates(self, client, health):
        _needs(health, "mart_arsenal_embedding")
        rows = client.get("/arsenal/embedding", params={"min_pitches": 1}).json()
        if not rows:
            pytest.skip("no embedded pitcher-seasons")
        for row in rows[:200]:
            assert math.isfinite(row["x"])
            assert math.isfinite(row["y"])

    def test_embedding_has_one_row_per_pitcher_season(self, client, health):
        _needs(health, "mart_arsenal_embedding")
        rows = client.get("/arsenal/embedding", params={"min_pitches": 1}).json()
        keys = [(r["mlbam_id"], r["season"]) for r in rows]
        assert len(keys) == len(set(keys))

    def test_similar_pitchers_respects_limit(self, client, health, embedded_pitcher):
        _needs(health, "mart_arsenal_neighbors")
        rows = client.get(f"/arsenal/{embedded_pitcher}/similar", params={"limit": 3}).json()
        assert 0 < len(rows) <= 3

    def test_similar_pitchers_never_returns_the_query_pitcher_season_itself(
        self, client, health, embedded_pitcher
    ):
        # A DIFFERENT season of the same pitcher is a legitimate neighbor (a
        # submariner's closest match is often his own other seasons) -- only
        # the exact queried (pitcher, season) pair must never appear.
        _needs(health, "mart_arsenal_neighbors")
        rows = client.get(f"/arsenal/{embedded_pitcher}/similar", params={"limit": 10}).json()
        seasons = client.get(f"/arsenal/{embedded_pitcher}").json()
        queried_season = max(r["season"] for r in seasons)
        assert not any(
            r["neighbor_id"] == embedded_pitcher and r["neighbor_season"] == queried_season for r in rows
        )

    def test_unknown_pitcher_similar_is_404(self, client, health):
        _needs(health, "mart_arsenal_neighbors")
        assert client.get("/arsenal/1/similar").status_code == 404
