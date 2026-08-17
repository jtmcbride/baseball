"""Tests for the mart-building helpers that don't need a registered model.

`marts.py` is otherwise thin orchestration over already-tested feature/model
code (see `test_called_strike.py`, `test_swing_path.py`) and isn't unit-tested
directly for that reason -- `_build_entity_grids` is the exception because it
carries real, new logic: per-season qualifying thresholds and a null-id-column
filter. The filter exists because `framing_runs`/`umpire_zone_rate` in
`models/called_strike.py` shipped without it first and silently produced a
phantom "unknown umpire" row per pre-2023 season (see that module's
docstring) -- these tests pin the fix at the source instead of relying on
someone noticing a stray row in a Parquet file a second time.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bbetl.transforms.zones import MetricSpec
from bbml.marts import _build_entity_grids

ID_COL = "entity_id"


def make_takes(n: int, *, entity_id: int | None, season: int, x: float = 0.0, z: float = 0.5):
    rng = np.random.default_rng(entity_id or 0)
    return pl.DataFrame(
        {
            ID_COL: [entity_id] * n,
            "season": [season] * n,
            "plate_x": rng.normal(x, 0.3, n),
            "plate_z_norm": rng.normal(z, 0.3, n),
            "_w": rng.normal(0.0, 0.05, n),
        }
    )


SPEC = MetricSpec("edge", "_w")


class TestBuildEntityGrids:
    def test_a_null_id_column_is_dropped_not_grouped(self):
        """A giant null-id group (pre-coverage seasons, real umpire ids only
        from 2023+) must not turn into a phantom entity row."""
        known = make_takes(600, entity_id=1, season=2024)
        unknown = make_takes(50_000, entity_id=None, season=2019)
        frame = pl.concat([known, unknown])

        out = _build_entity_grids(frame, id_col=ID_COL, role="umpire", spec=SPEC, min_pitches=500)

        assert out.height == 1
        assert out["mlbam_id"].to_list() == [1]
        assert None not in out["mlbam_id"].to_list()

    def test_below_the_pitch_qualifier_is_excluded(self):
        thin = make_takes(100, entity_id=2, season=2024)
        out = _build_entity_grids(thin, id_col=ID_COL, role="catcher", spec=SPEC, min_pitches=500)
        assert out.height == 0

    def test_qualifying_entity_seasons_each_get_one_row(self):
        a = make_takes(600, entity_id=1, season=2023)
        b = make_takes(700, entity_id=2, season=2024)
        frame = pl.concat([a, b])

        out = _build_entity_grids(frame, id_col=ID_COL, role="catcher", spec=SPEC, min_pitches=500)

        assert sorted(zip(out["mlbam_id"].to_list(), out["season"].to_list(), strict=True)) == [
            (1, 2023),
            (2, 2024),
        ]
        assert set(out["role"].to_list()) == {"catcher"}
        assert set(out["metric"].to_list()) == {"edge"}
