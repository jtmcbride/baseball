"""build_batter_spray_mart on a fixture lake -- viz #8's smoothed field-position
surface, batter x season. Writes/reads a real tiny Parquet lake under tmp_path
rather than mocking `load_batted_ball_frame`, so the season-partition glob and
the qualifier logic are exercised together."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import Settings
from bbml.marts import MART_BATTER_SPRAY, build_batter_spray_mart


def _write_fixture_lake(settings: Settings, *, n_qualified: int, n_thin: int) -> None:
    rng = np.random.default_rng(0)

    def rows(batter: int, season: int, n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "batter": [batter] * n,
                "season": [season] * n,
                "x_ft": rng.normal(50.0, 40.0, n),
                "y_ft": rng.normal(250.0, 60.0, n),
                "launch_speed": rng.normal(90.0, 8.0, n),
                "launch_angle": rng.normal(15.0, 10.0, n),
                "bb_type": ["fly_ball"] * n,
                "estimated_woba_using_speedangle": rng.uniform(0.1, 0.9, n),
                "events": ["field_out"] * n,
                "home_team": ["NYY"] * n,
                "is_tracked_pitch": [True] * n,
                "is_competitive": [True] * n,
                "game_type": ["R"] * n,
                "is_in_play": [True] * n,
            }
        )

    df = pl.concat(
        [
            rows(1, 2024, n_qualified),
            rows(2, 2024, n_thin),
        ]
    )
    out_dir = settings.lake_dir / "fact_pitch" / "season=2024"
    out_dir.mkdir(parents=True)
    df.write_parquet(out_dir / "part_0.parquet")


@pytest.fixture
def lake_settings(tmp_path) -> Settings:
    s = Settings(data_root=tmp_path)
    s.ensure_dirs()
    return s


class TestBuildBatterSprayMart:
    def test_qualified_batter_gets_one_grid_row(self, lake_settings):
        _write_fixture_lake(lake_settings, n_qualified=150, n_thin=10)
        out = build_batter_spray_mart(settings=lake_settings, min_batted_balls=100)
        assert out["mlbam_id"].to_list() == [1]
        assert out["season"].to_list() == [2024]
        assert out["n_batted_balls"][0] == 150

    def test_thin_batter_is_excluded(self, lake_settings):
        _write_fixture_lake(lake_settings, n_qualified=150, n_thin=10)
        out = build_batter_spray_mart(settings=lake_settings, min_batted_balls=100)
        assert 2 not in out["mlbam_id"].to_list()

    def test_writes_a_readable_parquet(self, lake_settings):
        _write_fixture_lake(lake_settings, n_qualified=150, n_thin=10)
        build_batter_spray_mart(settings=lake_settings, min_batted_balls=100)
        written = pl.read_parquet(lake_settings.lake_dir / MART_BATTER_SPRAY / "*.parquet")
        assert written.height == 1
        assert len(written["surface"][0]) == 60 * 60

    def test_no_batted_balls_returns_an_empty_frame(self, lake_settings):
        out_dir = lake_settings.lake_dir / "fact_pitch" / "season=2024"
        out_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "batter": [1],
                "season": [2024],
                "x_ft": [None],
                "y_ft": [None],
                "launch_speed": [90.0],
                "launch_angle": [10.0],
                "bb_type": ["fly_ball"],
                "estimated_woba_using_speedangle": [0.5],
                "events": ["field_out"],
                "home_team": ["NYY"],
                "is_tracked_pitch": [True],
                "is_competitive": [True],
                "game_type": ["R"],
                "is_in_play": [True],
            },
            schema_overrides={"x_ft": pl.Float64, "y_ft": pl.Float64},
        ).write_parquet(out_dir / "part_0.parquet")
        out = build_batter_spray_mart(settings=lake_settings, min_batted_balls=100)
        assert out.height == 0
