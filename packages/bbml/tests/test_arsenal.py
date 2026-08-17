"""Arsenal re-classification tests.

`TestClusterRecovery` is the reason this model exists, so it is pinned against
synthetic data with a known ground truth rather than trusted on shape alone: a
GMM that fails to separate two well-separated physical shapes, or that splits
one shape into two, would make every `arsenal_size_diff` in the output
meaningless while still looking like a plausible-shaped table.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import get_settings
from bbml.features.arsenal import CLUSTER_FEATURE_NAMES, build_arsenal_frame
from bbml.models.arsenal import (
    MIN_PITCHES,
    _circular_mean_deg,
    _label_clusters,
    build_arsenal_clusters,
    cluster_pitcher_season,
    yoy_stability,
)


def _pitches(n: int, *, velo: float, ivb: float, hb: float, spin_deg: float, label: str, seed: int):
    rng = np.random.default_rng(seed)
    spin_rad = np.deg2rad(spin_deg)
    # Dates spread randomly across a season, not chronologically segregated —
    # a `_split_group` candidate that happens to line up with the calendar
    # gets rejected as within-season drift (see the module docstring's
    # second "measured, not assumed" section), so two genuinely concurrent
    # pitch shapes in a test fixture must not be date-separated either.
    dates = np.array(["2025-04-01"], dtype="datetime64[D]") + rng.integers(0, 180, n)
    return pl.DataFrame(
        {
            "release_speed": rng.normal(velo, 0.8, n),
            "ivb_in": rng.normal(ivb, 1.0, n),
            "hb_arm_in": rng.normal(hb, 1.0, n),
            "spin_axis_sin": np.sin(spin_rad) + rng.normal(0, 0.03, n),
            "spin_axis_cos": np.cos(spin_rad) + rng.normal(0, 0.03, n),
            "release_extension": rng.normal(6.4, 0.1, n),
            "release_pos_x_arm": rng.normal(-1.9, 0.1, n),
            "release_pos_z": rng.normal(5.9, 0.1, n),
            "pitch_type": [label] * n,
            "game_date": dates,
        }
    )


class TestCircularMean:
    def test_averages_angles_that_straddle_the_zero_wraparound(self):
        sin = np.sin(np.deg2rad([359.0, 1.0]))
        cos = np.cos(np.deg2rad([359.0, 1.0]))
        # A naive arithmetic mean of 359 and 1 gives 180 -- exactly backwards.
        assert _circular_mean_deg(sin, cos) < 5.0 or _circular_mean_deg(sin, cos) > 355.0

    def test_agrees_with_arithmetic_mean_away_from_the_wraparound(self):
        sin = np.sin(np.deg2rad([80.0, 100.0]))
        cos = np.cos(np.deg2rad([80.0, 100.0]))
        assert abs(_circular_mean_deg(sin, cos) - 90.0) < 0.5


class TestLabelClusters:
    def test_unique_labels_stay_bare(self):
        assert _label_clusters(["FF", "SL", "CH"], [95.0, 84.0, 86.0]) == ["FF", "SL", "CH"]

    def test_shared_labels_get_suffixed_hardest_first(self):
        out = _label_clusters(["SL", "SL"], [88.0, 80.0])
        assert out == ["SL", "SL-2"]

    def test_three_way_collision_orders_by_velocity(self):
        out = _label_clusters(["SL", "SL", "SL"], [80.0, 90.0, 85.0])
        # index 1 (90mph) hardest -> bare label; then 85, then 80.
        assert out[1] == "SL"
        assert out[2] == "SL-2"
        assert out[0] == "SL-3"


class TestClusterRecovery:
    def test_finds_two_shapes_savant_blurred_into_one_label(self):
        """Two well-separated physical shapes, both labeled 'SL' -- the blur
        this model exists to catch. GMM should still find k=2."""
        hard = _pitches(200, velo=88, ivb=2, hb=3, spin_deg=180, label="SL", seed=1)
        sweep = _pitches(200, velo=78, ivb=-4, hb=14, spin_deg=230, label="SL", seed=2)
        df = pl.concat([hard, sweep])

        out = cluster_pitcher_season(df)

        assert out["cluster_k"][0] == 2
        assert out["arsenal_size_diff"][0] == 1  # GMM found one MORE shape than Savant labeled
        labels = sorted(out["label"].to_list())
        assert labels == ["SL", "SL-2"]

    def test_recovers_one_shape_savant_over_split_into_two_labels(self):
        """One physical shape, split across two Savant labels -- GMM should
        merge it back to k=1."""
        one_shape_a = _pitches(150, velo=94, ivb=15, hb=8, spin_deg=210, label="FF", seed=3)
        one_shape_b = _pitches(150, velo=94.2, ivb=15.3, hb=8.1, spin_deg=211, label="SI", seed=4)
        df = pl.concat([one_shape_a, one_shape_b])

        out = cluster_pitcher_season(df)

        assert out["cluster_k"][0] == 1
        assert out["arsenal_size_diff"][0] == -1  # Savant claimed 2 types for 1 real shape
        assert out["n_savant_labels"][0] == 2
        assert out["purity"][0] < 1.0  # the merged cluster is NOT purely one label

    def test_recovers_a_clean_three_pitch_mix_with_full_purity(self):
        ff = _pitches(200, velo=95, ivb=16, hb=8, spin_deg=210, label="FF", seed=5)
        sl = _pitches(150, velo=85, ivb=1, hb=2, spin_deg=100, label="SL", seed=6)
        ch = _pitches(120, velo=86, ivb=6, hb=14, spin_deg=250, label="CH", seed=7)
        df = pl.concat([ff, sl, ch])

        out = cluster_pitcher_season(df)

        assert out["cluster_k"][0] == 3
        assert out["arsenal_size_diff"][0] == 0
        assert set(out["label"].to_list()) == {"FF", "SL", "CH"}
        assert (out["purity"] > 0.95).all()

    def test_cluster_row_shape_and_usage_sums_to_100(self):
        ff = _pitches(200, velo=95, ivb=16, hb=8, spin_deg=210, label="FF", seed=8)
        sl = _pitches(150, velo=85, ivb=1, hb=2, spin_deg=100, label="SL", seed=9)
        out = cluster_pitcher_season(pl.concat([ff, sl]))

        assert set(CLUSTER_FEATURE_NAMES) == {
            "release_speed", "ivb_in", "hb_arm_in", "spin_axis_sin", "spin_axis_cos",
        }
        assert out["n"].sum() == 350
        assert abs(out["usage_pct"].sum() - 100.0) < 1e-6


class TestBuildArsenalClusters:
    def test_qualifying_and_thin_pitcher_seasons_are_separated(self):
        qualifies = _pitches(MIN_PITCHES + 50, velo=95, ivb=16, hb=8, spin_deg=210, label="FF", seed=10)
        qualifies = qualifies.with_columns(pl.lit(1).alias("pitcher"), pl.lit(2024).alias("season"))
        thin = _pitches(50, velo=95, ivb=16, hb=8, spin_deg=210, label="FF", seed=11)
        thin = thin.with_columns(pl.lit(2).alias("pitcher"), pl.lit(2024).alias("season"))

        out = build_arsenal_clusters(pl.concat([qualifies, thin]))

        assert set(out["pitcher"].to_list()) == {1}

    def test_each_pitcher_season_is_clustered_independently(self):
        a = _pitches(MIN_PITCHES, velo=95, ivb=16, hb=8, spin_deg=210, label="FF", seed=12)
        a = a.with_columns(pl.lit(1).alias("pitcher"), pl.lit(2023).alias("season"))
        b = _pitches(MIN_PITCHES, velo=78, ivb=-4, hb=14, spin_deg=230, label="CU", seed=13)
        b = b.with_columns(pl.lit(1).alias("pitcher"), pl.lit(2024).alias("season"))

        out = build_arsenal_clusters(pl.concat([a, b]))

        assert sorted(out["season"].unique().to_list()) == [2023, 2024]
        assert out.filter(pl.col("season") == 2023)["label"].to_list() == ["FF"]
        assert out.filter(pl.col("season") == 2024)["label"].to_list() == ["CU"]

    def test_empty_input_returns_empty_frame_not_an_error(self):
        empty = pl.DataFrame(
            schema={
                **{c: pl.Float64 for c in CLUSTER_FEATURE_NAMES},
                "pitcher": pl.Int64, "season": pl.Int64, "pitch_type": pl.Utf8,
            }
        )
        out = build_arsenal_clusters(empty)
        assert out.height == 0


def _lake_available() -> bool:
    return bool(list((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet")))


@pytest.mark.skipif(not _lake_available(), reason="no local lake")
class TestRealData:
    def test_2015_has_no_spin_axis_and_yields_nothing_not_a_crash(self):
        """`spin_axis` wasn't published at all in 2015 (0% coverage, `bb check
        --coverage`) -- every 2015 pitch fails the CLUSTER_FEATURE_NAMES
        null-filter, so the season contributes zero rows. That's the correct
        behavior (this model's real usable range is 2016+, unlike every other
        mart in this codebase which covers the full 2015-2026 backfill) --
        this test exists so a future all-null season fails loud with an
        obviously-empty frame instead of silently, the way the *next* bug
        (below) did until a full-history build actually crashed on it."""
        frame = build_arsenal_frame(seasons=[2015])
        assert frame.height == 0

    def test_null_pitch_types_do_not_crash_the_loader(self):
        """A full 2015-2026 build crashed with `np.unique` unable to sort
        `None` against strings: `is_tracked_pitch & is_competitive` does not
        guarantee `pitch_type` was ever classified. `mart_pitcher_arsenal.sql`
        already filters this same case; `build_arsenal_frame` was missing it.
        2016 is the earliest season with any usable spin_axis coverage (see
        the 2015 test above) and reproduces the crash before the fix."""
        frame = build_arsenal_frame(seasons=[2016])
        assert frame.height > 0
        assert frame["pitch_type"].null_count() == 0

    def test_builds_on_a_real_recent_season(self):
        frame = build_arsenal_frame(seasons=[2025])
        out = build_arsenal_clusters(frame)
        assert out.height > 0
        # Every qualifying pitcher-season's usage should still sum to ~100.
        totals = out.group_by(["pitcher", "season"]).agg(pl.col("usage_pct").sum().alias("total"))
        assert (totals["total"].to_numpy() > 99.0).all()
        assert (totals["total"].to_numpy() < 101.0).all()

    def test_most_pitcher_seasons_are_not_wildly_off_from_savant(self):
        """Sanity check, not a precision claim: most pitcher-seasons should be
        within +-1 of Savant's own count, since both are looking at the same
        pitches. The bar is measured, not assumed -- see the module
        docstring's history of this exact number moving as real bugs got
        fixed (0.02 before the merge/split redesign existed at all, then
        0.76, then 0.67 once release-point features were removed and some
        previously-suppressed real splits came back). 0.55 has margin below
        the last measured value; if this regresses toward the original
        near-zero failure mode, that is the bug this test exists to catch."""
        frame = build_arsenal_frame(seasons=[2025])
        out = build_arsenal_clusters(frame)
        per_season = out.unique(subset=["pitcher", "season"])
        close = (per_season["arsenal_size_diff"].abs() <= 1).sum()
        assert close / per_season.height > 0.55

    def test_arsenal_size_is_year_over_year_stable_not_noise(self):
        """The reliability check this model is held to at the top of the
        module docstring: measured 0.56-0.61 for `cluster_k_yoy` across
        2023-2025, in the 0.5-0.7 range published framing metrics report as
        real and sticky (see `models/called_strike.py`'s equivalent check).
        A number near zero here would mean the merge/split output is mostly
        single-season noise, not a property of the pitcher."""
        frame = build_arsenal_frame(seasons=[2024, 2025])
        out = build_arsenal_clusters(frame)
        result = yoy_stability(out)
        assert result["n_pairs"] > 100
        assert result["cluster_k_yoy"] > 0.35
