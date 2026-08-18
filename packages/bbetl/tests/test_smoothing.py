"""kernel_regress_2d — the shared numeric core behind zones.py and spray.py."""

from __future__ import annotations

import numpy as np

from bbetl.transforms.smoothing import kernel_regress_2d


def test_matches_zones_pys_own_output_after_extraction():
    """Regression guard: the extracted helper must reproduce zones.py's
    pre-refactor smoothed output exactly, on zones.py's own grid."""
    from bbetl.transforms.zones import _SIGMA_X, _SIGMA_Z, X_EDGES, Z_EDGES, _smooth_ratio

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(-0.5, 1.5, n)
    w = rng.uniform(0.0, 1.0, n)

    surface, eff_n = _smooth_ratio(x, z, w)
    surface2, eff_n2 = kernel_regress_2d(x, z, w, X_EDGES, Z_EDGES, _SIGMA_X, _SIGMA_Z)

    np.testing.assert_allclose(surface, surface2, equal_nan=True)
    np.testing.assert_allclose(eff_n, eff_n2)


def test_a_single_cluster_of_points_smooths_to_a_peak_at_its_own_location():
    edges = np.linspace(-10.0, 10.0, 21)
    x = np.full(50, 3.0)
    y = np.full(50, -2.0)
    w = np.full(50, 1.0)
    _surface, eff_n = kernel_regress_2d(x, y, w, edges, edges, sigma_x=1.0, sigma_y=1.0)
    peak = np.unravel_index(np.nanargmax(eff_n), eff_n.shape)
    peak_x = (edges[peak[0]] + edges[peak[0] + 1]) / 2
    peak_y = (edges[peak[1]] + edges[peak[1] + 1]) / 2
    assert peak_x == 3.5
    assert peak_y == -1.5


def test_empty_cells_do_not_pull_the_surface_toward_zero_at_the_edge():
    """'nearest' padding, not zero-padding -- see zones.py's own comment on
    why zero-padding fabricates a cold rim."""
    edges = np.linspace(-10.0, 10.0, 21)
    x = np.full(100, -9.0)
    y = np.full(100, -9.0)
    w = np.full(100, 5.0)
    surface, _ = kernel_regress_2d(x, y, w, edges, edges, sigma_x=1.5, sigma_y=1.5)
    corner_cell = surface[0, 0]
    assert not np.isnan(corner_cell)
    assert corner_cell > 0
