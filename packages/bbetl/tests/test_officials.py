"""`dim_official` transform tests."""

from __future__ import annotations

import json

import polars as pl
import pytest

from bbcore.config import Settings
from bbetl.transforms.officials import build_dim_official


def _settings(tmp_path) -> Settings:
    return Settings(data_root=tmp_path)


class TestBuildDimOfficial:
    def test_missing_raw_files_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="No officials landed"):
            build_dim_official(settings=_settings(tmp_path))

    def test_reads_every_landed_game(self, tmp_path):
        s = _settings(tmp_path)
        (s.raw_dir / "officials").mkdir(parents=True)
        for pk, ump in ((1, "CB Bucknor"), (2, "Angel Hernandez")):
            (s.raw_dir / "officials" / f"{pk}.json").write_text(
                json.dumps(
                    {
                        "game_pk": pk,
                        "umpire_home_plate_id": 100 + pk,
                        "umpire_home_plate_name": ump,
                    }
                )
            )
        df = build_dim_official(settings=s)
        assert df.height == 2
        assert set(df["game_pk"].to_list()) == {1, 2}
        out = pl.read_parquet(s.lake_dir / "dim_official" / "part_0.parquet")
        assert out.height == 2

    def test_a_game_missing_base_umpires_does_not_lose_columns(self, tmp_path):
        """A boxscore gap on one game must not silently truncate the schema for
        every other game — see the schema-scan note in the transform."""
        s = _settings(tmp_path)
        (s.raw_dir / "officials").mkdir(parents=True)
        (s.raw_dir / "officials" / "1.json").write_text(
            json.dumps({"game_pk": 1, "umpire_home_plate_id": 101})
        )
        (s.raw_dir / "officials" / "2.json").write_text(
            json.dumps(
                {
                    "game_pk": 2,
                    "umpire_home_plate_id": 102,
                    "umpire_first_base_id": 202,
                }
            )
        )
        df = build_dim_official(settings=s)
        assert "umpire_first_base_id" in df.columns
        row1 = df.filter(pl.col("game_pk") == 1)
        assert row1["umpire_first_base_id"][0] is None
