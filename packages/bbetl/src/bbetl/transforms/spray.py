"""`mart_batter_spray`: smoothed xwOBA-on-contact surface over absolute field
position, batter x season (viz #8's spray chart contour).

Same Nadaraya-Watson kernel regression as `zones.py`'s hot/cold grids — see
that module's docstring for why a smoothed surface with an honest reliability
mask beats raw binned averages — sharing its numeric core via
`bbetl.transforms.smoothing.kernel_regress_2d`. The grid itself differs from
`zones.py`'s: this one spans the whole field in feet from home plate
(`x_ft`/`y_ft`, absolute position, not mirrored by handedness — see
`transforms/statcast.py`'s hit-coordinate comment), not the strike zone's
normalized units, so it gets its own bandwidth/reliability constants rather
than reusing the zone-profile ones unchanged.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.transforms.smoothing import kernel_regress_2d

log = get_logger(__name__)

# Grid extent in feet from home plate. Comfortably past the deepest MLB fence
# (~440ft in a few parks' power alleys) in both x and y so a foul-territory or
# warning-track ball never clips the edge.
GRID_N = 60
X_MIN, X_MAX = -350.0, 350.0
Y_MIN, Y_MAX = 0.0, 450.0

# Bandwidth in feet. A batted ball's landing spot is a much noisier signal per
# swing than a pitch's plate location (one imprecise physical event vs. a
# tightly tracked crossing point), and a batter sees far fewer balls in play
# than pitches in a season, so this is wider than the strike-zone grid's
# ~0.22ft-equivalent bandwidth -- 18ft is roughly the width of a spray zone
# grouping (pull/oppo gap), tight enough to keep pull vs. oppo structure
# visible, wide enough to stabilize sparse deep-field cells.
BANDWIDTH_FT = 18.0

# Cells backed by fewer than this many effective batted balls are not
# trustworthy. Lower than the zone grid's MIN_RELIABLE_N=25 because a full
# batter-season qualifies at ~100 batted balls total (MIN_BATTED_BALLS in
# marts.py) spread across a much larger 2D area than the strike zone -- 25
# would fade almost the entire surface.
MIN_RELIABLE_N = 12.0

X_EDGES = np.linspace(X_MIN, X_MAX, GRID_N + 1)
Y_EDGES = np.linspace(Y_MIN, Y_MAX, GRID_N + 1)
_SIGMA = BANDWIDTH_FT / ((X_MAX - X_MIN) / GRID_N)


def build_grid(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, int] | None:
    """One (surface, reliability, n) triple for a batter-season's batted balls,
    or None if there is no usable data. `surface` is weighted by
    `estimated_woba_using_speedangle` — no custom expected-outcome model exists
    for batted-ball quality (see `HISTORY.md`), and this is the same column
    `mart_zone_profile`'s xwoba metric and Stuff+/`RunValue` already consume.
    """
    sub = df.filter(
        pl.col("x_ft").is_not_null()
        & pl.col("y_ft").is_not_null()
        & pl.col("estimated_woba_using_speedangle").is_not_null()
    )
    if sub.height == 0:
        return None

    x = np.clip(sub["x_ft"].to_numpy().astype(np.float64), X_MIN, X_MAX - 1e-9)
    y = np.clip(sub["y_ft"].to_numpy().astype(np.float64), Y_MIN, Y_MAX - 1e-9)
    w = sub["estimated_woba_using_speedangle"].to_numpy().astype(np.float64)

    surface, eff_n = kernel_regress_2d(x, y, w, X_EDGES, Y_EDGES, _SIGMA, _SIGMA)
    return surface.astype(np.float32), eff_n.astype(np.float32), sub.height


def load_batted_ball_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Tracked, competitive, regular-season balls in play with a landing spot."""
    s = settings or get_settings()
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch")
        & pl.col("is_competitive")
        & (pl.col("game_type") == "R")
        & pl.col("is_in_play")
        & pl.col("x_ft").is_not_null()
        & pl.col("y_ft").is_not_null()
    )
    if seasons:
        lf = lf.filter(pl.col("season").is_in(seasons))
    return lf.select(
        "batter", "season", "x_ft", "y_ft", "launch_speed", "launch_angle",
        "bb_type", "estimated_woba_using_speedangle", "events", "home_team",
    ).collect()


def grid_extent() -> dict[str, float | int]:
    """Grid geometry, shipped to the client so it never hardcodes constants
    that could drift out of sync with this module — same reasoning as
    `zones.py`'s own `grid_extent`."""
    return {
        "grid_n": GRID_N,
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "min_reliable_n": MIN_RELIABLE_N,
    }
