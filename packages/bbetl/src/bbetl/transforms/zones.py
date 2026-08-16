"""mart_zone_profile: smoothed hot/cold surfaces with an explicit reliability mask.

WHY NOT RAW BINNED AVERAGES
---------------------------
The obvious implementation -- bin pitches into a 5x5 grid and average the outcome
per cell -- is what most public hot/cold charts do, and it is mostly rendering
noise. A batter sees a few thousand pitches a season spread over the zone and its
surroundings; the corner cells end up with a handful of batted balls each, so a
single home run swings a cell from ice-blue to dark-red. The chart looks
authoritative and means almost nothing.

Instead we estimate a smooth surface: bin finely, then convolve the outcome-weighted
counts and the raw pitch counts with the same Gaussian kernel and divide. That is a
Nadaraya-Watson kernel regression, and it borrows strength from neighbouring cells
in exactly the way the sparse-corner problem requires.

THE RELIABILITY MASK
--------------------
Smoothing hides thin data rather than fixing it: a cell can be interpolated almost
entirely from its neighbours and still render at full saturation. So every grid
ships alongside `reliability`, the smoothed effective sample size per cell, and the
UI fades cells below a threshold. Showing where the data runs out is both more
honest than the usual presentation and a genuine differentiator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger

log = get_logger(__name__)

Role = Literal["batter", "pitcher"]

# --- grid definition ---------------------------------------------------------
# x in feet from plate centre; z normalized to each batter's own strike zone, so
# 0.0 is the bottom of the zone and 1.0 the top. Padding beyond [0,1] keeps chase
# territory visible, which is where the interesting hitter differences live.
GRID_N = 50
X_MIN, X_MAX = -2.0, 2.0
Z_MIN, Z_MAX = -0.5, 1.5

# Kernel width in feet / zone-heights. Roughly a baseball's width in x, and a
# comparable fraction of the zone in z: wide enough to stabilize sparse corners,
# tight enough to preserve real high/low and in/out structure.
BANDWIDTH_X_FT = 0.22
BANDWIDTH_Z_NORM = 0.16

# Cells backed by fewer than this many effective pitches are not trustworthy.
# ~25 is where a rate estimate starts to mean anything; below it the surface is
# mostly interpolated from neighbours. Surfaced to the client, never silently
# zeroed.
MIN_RELIABLE_N = 25.0

X_EDGES = np.linspace(X_MIN, X_MAX, GRID_N + 1)
Z_EDGES = np.linspace(Z_MIN, Z_MAX, GRID_N + 1)
_SIGMA_X = BANDWIDTH_X_FT / ((X_MAX - X_MIN) / GRID_N)
_SIGMA_Z = BANDWIDTH_Z_NORM / ((Z_MAX - Z_MIN) / GRID_N)

# gaussian_filter returns smoothed density PER CELL, which is not what a reader
# means by "how much data backs this cell". Multiplying by the kernel's effective
# area converts it to an effective sample size: roughly the number of pitches
# within one bandwidth of the cell. That is the number worth thresholding on and
# the number worth showing in a tooltip.
_KERNEL_AREA = 2.0 * np.pi * _SIGMA_X * _SIGMA_Z


@dataclass(frozen=True)
class MetricSpec:
    """A hot/cold metric.

    `value_col` is averaged over the pitches matching `subset`. Restricting the
    subset is what makes each metric mean what its name says: xwOBA is defined
    over batted balls, whiff rate over swings, not over all pitches.
    """

    name: str
    value_col: str
    subset: pl.Expr | None = None
    scale: float = 1.0


METRICS: list[MetricSpec] = [
    MetricSpec("xwoba", "estimated_woba_using_speedangle", pl.col("is_in_play")),
    MetricSpec("whiff", "is_whiff", pl.col("is_swing"), scale=100.0),
    MetricSpec("swing", "is_swing", None, scale=100.0),
    MetricSpec("exit_velo", "launch_speed", pl.col("is_in_play")),
    # Run value, flipped so positive is good for the batter.
    MetricSpec("run_value", "delta_run_exp", None),
]


def _smooth_ratio(x: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-regress `w` onto the grid. Returns (surface, effective_n).

    `effective_n` is an effective sample size in pitches, not a per-cell density
    (see _KERNEL_AREA), so it can be thresholded and shown directly.

    Both numerator and denominator get the identical kernel, so their ratio is a
    proper weighted local mean rather than a smoothed-then-divided approximation.
    """
    num, _, _ = np.histogram2d(x, z, bins=[X_EDGES, Z_EDGES], weights=w)
    den, _, _ = np.histogram2d(x, z, bins=[X_EDGES, Z_EDGES])

    sigma = (_SIGMA_X, _SIGMA_Z)
    # 'nearest' rather than the default zero-padding: zero-padding pulls the edge
    # cells toward zero and fabricates a cold rim around every chart.
    num_s = gaussian_filter(num, sigma=sigma, mode="nearest")
    den_s = gaussian_filter(den, sigma=sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        surface = np.where(den_s > 1e-9, num_s / den_s, np.nan)
    return surface, den_s * _KERNEL_AREA


def build_grid(df: pl.DataFrame, spec: MetricSpec) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Build one (surface, reliability, n) triple, or None if there is no data."""
    sub = df.filter(spec.subset) if spec.subset is not None else df
    sub = sub.filter(
        pl.col("plate_x").is_not_null()
        & pl.col("plate_z_norm").is_not_null()
        & pl.col(spec.value_col).is_not_null()
    )
    if sub.height == 0:
        return None

    x = sub["plate_x"].to_numpy().astype(np.float64)
    z = sub["plate_z_norm"].to_numpy().astype(np.float64)
    w = sub[spec.value_col].to_numpy().astype(np.float64) * spec.scale

    # Clip rather than drop: a pitch well outside the grid still carries
    # information about the edge of the surface.
    x = np.clip(x, X_MIN, X_MAX - 1e-9)
    z = np.clip(z, Z_MIN, Z_MAX - 1e-9)

    surface, eff_n = _smooth_ratio(x, z, w)
    return surface.astype(np.float32), eff_n.astype(np.float32), sub.height


def build_zone_profiles(
    *,
    role: Role,
    min_pitches: int = 250,
    settings: Settings | None = None,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Build smoothed grids for every qualified player-season."""
    s = settings or get_settings()
    id_col = "batter" if role == "batter" else "pitcher"

    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch") & pl.col("is_competitive") & (pl.col("game_type") == "R")
    )
    if seasons:
        lf = lf.filter(pl.col("season").is_in(seasons))

    needed = [
        id_col,
        "season",
        "plate_x",
        "plate_z_norm",
        "is_swing",
        "is_whiff",
        "is_in_play",
        "estimated_woba_using_speedangle",
        "launch_speed",
        "delta_run_exp",
    ]
    df = lf.select(needed).collect()
    if df.height == 0:
        return pl.DataFrame()

    counts = df.group_by([id_col, "season"]).len().filter(pl.col("len") >= min_pitches)
    qualified = set(zip(counts[id_col].to_list(), counts["season"].to_list(), strict=True))
    log.info("%s zone profiles: %d qualified player-seasons", role, len(qualified))

    rows: list[dict] = []
    for (pid, season), group in df.group_by([id_col, "season"], maintain_order=True):
        if (pid, season) not in qualified:
            continue
        for spec in METRICS:
            built = build_grid(group, spec)
            if built is None:
                continue
            surface, eff_n, n = built
            rows.append(
                {
                    "mlbam_id": int(pid),
                    "season": int(season),
                    "role": role,
                    "metric": spec.name,
                    "n_pitches": n,
                    "grid_n": GRID_N,
                    # NaN does not survive JSON; the API converts these to null.
                    "surface": np.nan_to_num(surface, nan=np.nan).ravel().tolist(),
                    "reliability": eff_n.ravel().tolist(),
                }
            )

    if not rows:
        return pl.DataFrame()

    out = pl.DataFrame(rows)
    out_dir = s.lake_dir / "mart_zone_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_dir / f"{role}.parquet", compression="zstd")
    log.info("wrote %d zone grids for role=%s", out.height, role)
    return out


def grid_extent() -> dict[str, float | int]:
    """Grid geometry, shipped to the client so it can place cells without
    hardcoding constants that would silently drift from this module."""
    return {
        "grid_n": GRID_N,
        "x_min": X_MIN,
        "x_max": X_MAX,
        "z_min": Z_MIN,
        "z_max": Z_MAX,
        "min_reliable_n": MIN_RELIABLE_N,
    }
