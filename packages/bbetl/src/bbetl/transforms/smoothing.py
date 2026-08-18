"""Shared 2D kernel regression core, extracted from `zones.py`.

`zones.py`'s strike-zone grids and `spray.py`'s field-position grids are the same
statistical operation — bin an outcome-weighted count and a raw count onto a
grid, Gaussian-smooth both with the same kernel, divide — over two different
coordinate systems and bandwidths. Duplicating the numeric core would let the
two silently drift; this module is that core, parameterized by grid/bandwidth so
each caller supplies its own measured constants.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def kernel_regress_2d(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-regress `w` onto the grid defined by `x_edges`/`y_edges`.

    Returns `(surface, effective_n)`. `effective_n` is a per-cell effective
    sample size (a kernel-area-scaled smoothed density, not the raw smoothed
    count), so it can be thresholded and shown to the user directly. Both
    numerator and denominator get the identical kernel, so their ratio is a
    proper weighted local mean rather than a smoothed-then-divided
    approximation.
    """
    num, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    den, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    sigma = (sigma_x, sigma_y)
    # 'nearest' rather than the default zero-padding: zero-padding pulls the
    # edge cells toward zero and fabricates a cold rim around every chart.
    num_s = gaussian_filter(num, sigma=sigma, mode="nearest")
    den_s = gaussian_filter(den, sigma=sigma, mode="nearest")

    kernel_area = 2.0 * np.pi * sigma_x * sigma_y
    with np.errstate(invalid="ignore", divide="ignore"):
        surface = np.where(den_s > 1e-9, num_s / den_s, np.nan)
    return surface, den_s * kernel_area
