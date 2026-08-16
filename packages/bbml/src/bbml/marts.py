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

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbml.features.run_value import RunValue
from bbml.features.stuff import ROLES, TARGET_RUN_VALUE, build_pitch_quality_frame
from bbml.models.pitch_quality import PitchQualityModel
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


def _register(settings: Settings) -> None:
    """Best effort. The lake file is the deliverable; the view is a convenience.

    `open_warehouse` takes an exclusive lock, so this fails whenever the API
    server is up — which is the normal state while iterating on the UI. Say so
    and move on rather than losing a mart build to it.
    """
    from bbcore.storage import open_warehouse

    try:
        with open_warehouse(settings=settings) as wh:
            wh.register_lake_table(MART_TABLE, f"{MART_TABLE}/*.parquet")
        log.info("registered %s in the warehouse", MART_TABLE)
    except Exception as exc:
        log.warning(
            "could not register %s (%s). The Parquet is written; run `bb build register` "
            "with the API stopped.",
            MART_TABLE,
            exc,
        )
