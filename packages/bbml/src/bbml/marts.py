"""`mart_pitcher_stuff` — the model-scored mart behind the Stuff+ UI.

Unlike `mart_pitcher_arsenal` this cannot be a SQL file: producing a row means
running three boosters over every pitch, so the build lives here in Python and
lands a Parquet file in the lake exactly like the zone-profile mart does.

Grain is pitcher x season x pitch_type, plus a rollup row per pitcher-season
under `pitch_type = 'ALL'`. The rollup is the plain mean over the pitcher's
pitches, which is a usage-weighted average of his pitch types by construction —
a pitcher who throws an elite slider 8% of the time does not get to average it
against his fastball as an equal.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbetl.transforms.zones import GRID_N, MetricSpec, build_grid
from bbml.features.called_strike import (
    CATCHER_COLUMN,
    TARGET_CALLED_STRIKE,
    UMPIRE_COLUMN,
    build_called_strike_frame,
)
from bbml.features.run_value import RunValue
from bbml.features.stuff import ROLES, TARGET_RUN_VALUE, build_pitch_quality_frame
from bbml.features.swing import build_swing_frame
from bbml.models.called_strike import CalledStrikeModel, framing_runs, umpire_zone_rate
from bbml.models.pitch_quality import PitchQualityModel
from bbml.models.swing_path import SwingPathModel, plane_value_by_batter
from bbml.registry import latest_dir

log = get_logger(__name__)

MART_TABLE = "mart_pitcher_stuff"
ALL_PITCHES = "ALL"

# Below this a grade is one bad afternoon rather than a property of the pitch.
# `mart_pitcher_arsenal` uses 50 for percentile eligibility; this is lower
# because rows are still useful to display with their sample size beside them,
# and the API applies its own threshold for leaderboards.
MIN_PITCHES = 25


def load_models() -> dict[str, PitchQualityModel]:
    """The registered head for each role. Raises if any is missing."""
    models = {}
    for role in ROLES:
        directory = latest_dir(f"{role}_plus")
        if directory is None:
            raise FileNotFoundError(
                f"No registered {role}_plus model. Run `bb-ml stuff` before building {MART_TABLE}."
            )
        models[role] = PitchQualityModel.load(directory)
    return models


def build_pitch_quality_mart(
    *,
    seasons: list[int] | None = None,
    min_pitches: int = MIN_PITCHES,
    settings: Settings | None = None,
) -> pl.DataFrame:
    s = settings or get_settings()
    models = load_models()

    frame = build_pitch_quality_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no pitches to score")
        return pl.DataFrame()

    # The observed run value rides along so the UI can show a grade beside the
    # result it is claiming to see through. Fitted on the same frame it labels:
    # this is a descriptive mart, not an evaluation, so there is no split to
    # respect here.
    frame = RunValue.fit(frame).attach(frame)
    scored = frame.with_columns(
        [pl.Series(f"{role}_plus", models[role].plus(frame)) for role in ROLES]
    )

    metrics = [
        *[pl.col(f"{role}_plus").mean().round(1) for role in ROLES],
        # RV/100 in the same units and sign as mart_pitcher_arsenal.
        (100.0 * pl.col(TARGET_RUN_VALUE).mean()).round(3).alias("rv_per_100"),
        pl.len().alias("pitches"),
    ]
    by_type = scored.group_by(["pitcher", "season", "pitch_type"]).agg(metrics)
    overall = (
        scored.group_by(["pitcher", "season"])
        .agg(metrics)
        .with_columns(pl.lit(ALL_PITCHES).alias("pitch_type"))
    )

    out = (
        pl.concat([by_type, overall.select(by_type.columns)])
        .filter(pl.col("pitch_type").is_not_null() & (pl.col("pitches") >= min_pitches))
        .rename({"pitcher": "mlbam_id"})
        .with_columns(
            (
                100.0
                * pl.col("pitches")
                / pl.col("pitches")
                .filter(pl.col("pitch_type") == ALL_PITCHES)
                .first()
                .over(["mlbam_id", "season"])
            )
            .round(1)
            .alias("usage_pct")
        )
        .sort(["mlbam_id", "season", "pitch_type"])
    )

    out_dir = s.lake_dir / MART_TABLE
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_dir / "part_0.parquet", compression="zstd", statistics=True)
    log.info("wrote %d rows -> %s", out.height, out_dir)
    _register(s)
    return out


MART_CATCHER_FRAMING = "mart_catcher_framing"
MART_UMPIRE_ZONE = "mart_umpire_zone"

# Season grain, so lower than the swing-path batter qualifier: even a backup
# catcher clears this most seasons, and a season with fewer takes than this is
# one nobody should be drawing a framing conclusion from anyway.
MIN_CATCHER_PITCHES = 500
MIN_UMPIRE_PITCHES = 500


def load_called_strike_model() -> tuple[CalledStrikeModel, RunValue]:
    """The registered called-strike model, with the `RunValue` table it shipped
    with — see `PitchQualityModel`'s equivalent note on why the target
    definition rides along with the artifact rather than being refit."""
    directory = latest_dir("called_strike")
    if directory is None:
        raise FileNotFoundError(
            "No registered called_strike model. Run `bb-ml called-strike` first."
        )
    model = CalledStrikeModel.load(directory)
    rv = RunValue.load(directory / "run_value.json")
    return model, rv


def build_catcher_framing_mart(
    *,
    seasons: list[int] | None = None,
    min_pitches: int = MIN_CATCHER_PITCHES,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`mart_catcher_framing`: framing runs per catcher x season (viz #20)."""
    s = settings or get_settings()
    model, rv = load_called_strike_model()
    frame = build_called_strike_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no taken pitches to score")
        return pl.DataFrame()

    rows = [
        framing_runs(part, model, rv, group_col=CATCHER_COLUMN, min_pitches=min_pitches).with_columns(
            pl.lit(season).alias("season")
        )
        for (season,), part in frame.group_by("season", maintain_order=True)
    ]
    out = (
        pl.concat(rows)
        .rename({CATCHER_COLUMN: "mlbam_id"})
        .sort(["season", "framing_runs"], descending=[False, True])
    )
    _write(out, MART_CATCHER_FRAMING, s)
    return out


MART_BATTER_SWING = "mart_batter_swing"

# Matches `bb-ml swing`'s own default qualifier — a mart built with a looser
# threshold than the CLI leaderboard it's meant to agree with would be a
# second, silently different definition of "enough swings".
MIN_BATTER_SWINGS = 200


def build_batter_swing_mart(
    *,
    seasons: list[int] | None = None,
    min_swings: int = MIN_BATTER_SWINGS,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`mart_batter_swing`: both `plane_value` heads per batter x season.

    `plane_value_by_batter` already does the aggregation (features/swing.py's
    counterfactual, per head); this just runs it for both heads and joins
    them into one row per batter-season so the UI doesn't need two round trips
    for two numbers about the same swing.
    """
    s = settings or get_settings()
    whiff_dir, contact_dir = latest_dir("swing_whiff"), latest_dir("swing_contact")
    if whiff_dir is None or contact_dir is None:
        raise FileNotFoundError(
            "No registered swing_whiff/swing_contact model. Run `bb-ml swing` first."
        )
    whiff_model = SwingPathModel.load(whiff_dir)
    contact_model = SwingPathModel.load(contact_dir)

    frame = build_swing_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no swings to score")
        return pl.DataFrame()

    whiff_board = plane_value_by_batter(whiff_model, frame, min_swings=min_swings).rename(
        {"plane_value_per_100": "whiff_plane_value_per_100", "swings": "whiff_swings"}
    )
    contact_board = plane_value_by_batter(contact_model, frame, min_swings=min_swings).select(
        "batter",
        "season",
        pl.col("plane_value_per_100").alias("contact_plane_value_per_100"),
        pl.col("swings").alias("contact_swings"),
    )
    out = (
        whiff_board.join(contact_board, on=["batter", "season"], how="left")
        .rename({"batter": "mlbam_id"})
        .sort(["season", "whiff_plane_value_per_100"], descending=[False, True])
    )
    _write(out, MART_BATTER_SWING, s)
    return out


def build_umpire_zone_mart(
    *,
    seasons: list[int] | None = None,
    min_pitches: int = MIN_UMPIRE_PITCHES,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`mart_umpire_zone`: actual-vs-expected borderline strike rate per umpire
    x season (viz #13), plus the same framing-runs formula grouped by umpire
    instead of catcher."""
    s = settings or get_settings()
    model, rv = load_called_strike_model()
    frame = build_called_strike_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no taken pitches to score")
        return pl.DataFrame()

    rows = [
        umpire_zone_rate(part, model, min_pitches=min_pitches)
        .join(
            framing_runs(
                part, model, rv, group_col=UMPIRE_COLUMN, min_pitches=min_pitches
            ).rename({"n": "framing_n"}),
            on=UMPIRE_COLUMN,
            how="left",
        )
        .with_columns(pl.lit(season).alias("season"))
        for (season,), part in frame.group_by("season", maintain_order=True)
    ]
    out = (
        pl.concat(rows)
        .rename({UMPIRE_COLUMN: "mlbam_id"})
        .sort(["season", "edge"], descending=[False, True])
    )
    _write(out, MART_UMPIRE_ZONE, s)
    return out


MART_ZONE_PROFILE = "mart_zone_profile"

# Same qualifier as the scalar marts above -- a grid needs the same amount of
# data to be worth drawing as a season total needs to be worth publishing.
MIN_GRID_PITCHES = MIN_CATCHER_PITCHES


def _build_entity_grids(
    frame: pl.DataFrame, *, id_col: str, role: str, spec: MetricSpec, min_pitches: int
) -> pl.DataFrame:
    """One smoothed grid per (entity, season), same row shape as
    `mart_zone_profile` (`bbetl.transforms.zones.build_zone_profiles`) so
    `StrikeZoneHeatmap` and `/zones/{id}` need no special-casing for
    role="catcher"/"umpire" -- they already take any role/metric pair.

    `id_col` can be structurally null (`UMPIRE_COLUMN` only covers 2023+, see
    `dim_official`) -- filtered before grouping. This is the exact null-group
    bug `framing_runs`/`umpire_zone_rate` had (see `models/called_strike.py`);
    fixed here from the start rather than caught after the fact.
    """
    frame = frame.filter(pl.col(id_col).is_not_null())
    counts = frame.group_by([id_col, "season"]).len().filter(pl.col("len") >= min_pitches)
    qualified = set(zip(counts[id_col].to_list(), counts["season"].to_list(), strict=True))

    rows: list[dict] = []
    for (pid, season), group in frame.group_by([id_col, "season"], maintain_order=True):
        if (pid, season) not in qualified:
            continue
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
                "surface": np.nan_to_num(surface, nan=np.nan).ravel().tolist(),
                "reliability": eff_n.ravel().tolist(),
            }
        )
    return pl.DataFrame(rows)


def build_catcher_framing_grid(
    *,
    seasons: list[int] | None = None,
    min_pitches: int = MIN_GRID_PITCHES,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`mart_zone_profile` role="catcher": where a catcher's framing edge runs
    positive/negative across the zone (viz #20) -- not just the season total
    `mart_catcher_framing` already has. Weight per pitch is the same residual
    `framing_runs` sums, `actual_strike - P(strike)`, left as strikes rather
    than run-scaled so the surface reads as "does he steal/lose calls here",
    independent of which counts those calls happened to come in.
    """
    s = settings or get_settings()
    model, _rv = load_called_strike_model()
    frame = build_called_strike_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no taken pitches to score")
        return pl.DataFrame()

    p = model.predict_proba(frame)
    actual = frame[TARGET_CALLED_STRIKE].cast(pl.Float64).to_numpy()
    frame = frame.with_columns(pl.Series("_edge", actual - p))

    out = _build_entity_grids(
        frame,
        id_col=CATCHER_COLUMN,
        role="catcher",
        spec=MetricSpec("framing", "_edge"),
        min_pitches=min_pitches,
    )
    if out.height == 0:
        return out
    _write_zone_grid(out, "catcher", s)
    return out


def build_umpire_zone_grid(
    *,
    seasons: list[int] | None = None,
    min_pitches: int = MIN_GRID_PITCHES,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """`mart_zone_profile` role="umpire": actual called-strike rate by
    location (viz #13) -- the client draws its 50% contour against the
    rulebook rectangle the chart already has, to show the umpire's effective
    zone shape rather than its color alone. No model score needed here, only
    the umpire's own ball/strike calls -- `is_called_strike` IS the rate being
    mapped, unlike the catcher grid which needs the residual against a
    prediction.
    """
    s = settings or get_settings()
    frame = build_called_strike_frame(seasons=seasons, settings=s)
    if frame.height == 0:
        log.warning("no taken pitches to score")
        return pl.DataFrame()

    frame = frame.with_columns(
        (pl.col(TARGET_CALLED_STRIKE).cast(pl.Float64) * 100.0).alias("_strike_pct")
    )
    out = _build_entity_grids(
        frame,
        id_col=UMPIRE_COLUMN,
        role="umpire",
        spec=MetricSpec("strike_rate", "_strike_pct"),
        min_pitches=min_pitches,
    )
    if out.height == 0:
        return out
    _write_zone_grid(out, "umpire", s)
    return out


def _write_zone_grid(df: pl.DataFrame, filename: str, settings: Settings) -> None:
    out_dir = settings.lake_dir / MART_ZONE_PROFILE
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / f"{filename}.parquet", compression="zstd", statistics=True)
    log.info("wrote %d %s zone grids -> %s", df.height, filename, out_dir)
    _register_table(MART_ZONE_PROFILE, settings)


def _write(df: pl.DataFrame, name: str, settings: Settings) -> None:
    out_dir = settings.lake_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "part_0.parquet", compression="zstd", statistics=True)
    log.info("wrote %d rows -> %s", df.height, out_dir)
    _register_table(name, settings)


def _register_table(name: str, settings: Settings) -> None:
    """Best effort. The lake file is the deliverable; the view is a convenience.

    `open_warehouse` takes an exclusive lock, so this fails whenever the API
    server is up — which is the normal state while iterating on the UI. Say so
    and move on rather than losing a mart build to it.
    """
    from bbcore.storage import open_warehouse

    try:
        with open_warehouse(settings=settings) as wh:
            wh.register_lake_table(name, f"{name}/*.parquet")
        log.info("registered %s in the warehouse", name)
    except Exception as exc:
        log.warning(
            "could not register %s (%s). The Parquet is written; run `bb build register` "
            "with the API stopped.",
            name,
            exc,
        )


def _register(settings: Settings) -> None:
    _register_table(MART_TABLE, settings)
