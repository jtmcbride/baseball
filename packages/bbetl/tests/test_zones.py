"""Tests for the hot/cold zone smoothing."""

from __future__ import annotations

import numpy as np
import polars as pl

from bbetl.transforms.zones import (
    GRID_N,
    METRICS,
    MIN_RELIABLE_N,
    MetricSpec,
    build_grid,
    grid_extent,
)


def make_pitches(n: int, *, x: float, z: float, value: float, spread: float = 0.05):
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "plate_x": rng.normal(x, spread, n),
            "plate_z_norm": rng.normal(z, spread, n),
            "value": np.full(n, value),
            "is_in_play": [True] * n,
            "is_swing": [True] * n,
        }
    )


SPEC = MetricSpec("test", "value")


class TestSmoothing:
    def test_returns_none_without_data(self):
        empty = pl.DataFrame(
            {"plate_x": [], "plate_z_norm": [], "value": []},
            schema={"plate_x": pl.Float64, "plate_z_norm": pl.Float64, "value": pl.Float64},
        )
        assert build_grid(empty, SPEC) is None

    def test_grid_has_the_declared_shape(self):
        surface, reliability, n = build_grid(make_pitches(200, x=0, z=0.5, value=1.0), SPEC)
        assert surface.shape == (GRID_N, GRID_N)
        assert reliability.shape == (GRID_N, GRID_N)
        assert n == 200

    def test_constant_outcome_yields_that_constant_where_data_exists(self):
        """A kernel-weighted mean of a constant must be that constant — this
        catches a numerator/denominator mismatch in the ratio."""
        surface, reliability, _ = build_grid(make_pitches(500, x=0.0, z=0.5, value=0.400), SPEC)
        dense = reliability > MIN_RELIABLE_N
        assert dense.any()
        assert np.allclose(surface[dense], 0.400, atol=1e-3)

    def test_hot_region_appears_where_the_hot_pitches_are(self):
        cold = make_pitches(400, x=-0.6, z=0.5, value=0.200)
        hot = make_pitches(400, x=0.6, z=0.5, value=0.900)
        surface, _, _ = build_grid(pl.concat([cold, hot]), SPEC)

        # Column index for x=-0.6 vs x=+0.6 on a grid spanning [-2, 2].
        left = int((-0.6 - (-2.0)) / 4.0 * GRID_N)
        right = int((0.6 - (-2.0)) / 4.0 * GRID_N)
        mid = GRID_N // 2
        assert surface[right, mid] > surface[left, mid]
        assert surface[right, mid] > 0.6
        assert surface[left, mid] < 0.4

    def test_smoothing_borrows_strength_from_neighbours(self):
        """The reason for kernel regression: a lone pitch should not paint its
        cell at full value the way a raw binned average would."""
        many = make_pitches(500, x=0.0, z=0.5, value=0.0)
        one = pl.DataFrame(
            {
                "plate_x": [0.08],
                "plate_z_norm": [0.5],
                "value": [1.0],
                "is_in_play": [True],
                "is_swing": [True],
            }
        )
        surface, _, _ = build_grid(pl.concat([many, one]), SPEC)
        col = int((0.08 - (-2.0)) / 4.0 * GRID_N)
        row = GRID_N // 2
        assert surface[col, row] < 0.25

    def test_reliability_tracks_sample_density(self):
        _surface, reliability, _ = build_grid(make_pitches(600, x=0.0, z=0.5, value=1.0), SPEC)
        centre = reliability[GRID_N // 2, GRID_N // 2]
        corner = reliability[0, 0]
        assert centre > MIN_RELIABLE_N
        assert corner < centre
        # The empty corner must be flagged unreliable rather than merely rendered.
        assert corner < MIN_RELIABLE_N

    def test_edge_mode_does_not_fabricate_a_cold_rim(self):
        """Zero-padded convolution drags edge cells toward zero and invents a
        cold border on every chart. 'nearest' padding avoids that."""
        surface, reliability, _ = build_grid(
            make_pitches(800, x=0.0, z=0.5, value=0.5, spread=0.8), SPEC
        )
        dense = reliability > MIN_RELIABLE_N
        assert np.allclose(surface[dense], 0.5, atol=0.05)

    def test_out_of_range_pitches_are_clipped_not_dropped(self):
        far = pl.DataFrame(
            {
                "plate_x": [-9.0, 9.0],
                "plate_z_norm": [-9.0, 9.0],
                "value": [1.0, 1.0],
                "is_in_play": [True, True],
                "is_swing": [True, True],
            }
        )
        result = build_grid(far, SPEC)
        assert result is not None
        assert result[2] == 2


class TestMetricSubsets:
    def test_each_metric_is_defined_over_the_right_subset(self):
        """xwOBA over batted balls, whiff over swings. Averaging either over all
        pitches would silently change what the number means."""
        by_name = {m.name: m for m in METRICS}
        assert by_name["xwoba"].subset is not None
        assert by_name["whiff"].subset is not None
        assert by_name["swing"].subset is None

    def test_rate_metrics_are_scaled_to_percent(self):
        by_name = {m.name: m for m in METRICS}
        assert by_name["whiff"].scale == 100.0
        assert by_name["xwoba"].scale == 1.0


def test_grid_extent_is_self_describing():
    """Shipped to the client so it never hardcodes geometry that could drift."""
    ext = grid_extent()
    assert ext["grid_n"] == GRID_N
    assert ext["x_min"] < ext["x_max"]
    assert ext["z_min"] < ext["z_max"]
    assert ext["min_reliable_n"] == MIN_RELIABLE_N
