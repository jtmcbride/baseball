"""Adapters: normalize each source into the frame `build_features` expects.

This is the ONLY layer allowed to differ between batch and live, because the
inputs genuinely differ in shape — Parquet columns versus a game-feed state
object. Everything after this point is shared.

The rule this module enforces: an adapter may rename, reshape, and reorder. It
may not compute a feature. The moment an adapter derives something the other
adapter doesn't, the two paths have forked and the parity test is the only thing
standing between that and a silently degraded live model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from bbcore.config import Settings, get_settings
from bbml.features.build import REQUIRED_COLUMNS, build_features

# Lake columns that already carry the right name and meaning.
_PASSTHROUGH = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "season",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "base_state",
    "bat_score",
    "fld_score",
    "stand",
    "p_throws",
    "home_team",
    "inning_topbot",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    "pitch_type",
    "plate_x",
    "plate_z_norm",
    "release_speed",
    "is_swing",
    "is_whiff",
    "is_in_zone",
]


def load_batch_frame(
    *,
    seasons: list[int] | None = None,
    pitcher_ids: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Read the pitch history the batch path trains on.

    Only tracked, competitive, regular-season pitches: automatic ball/strike calls
    carry no pitch at all, and pitchouts/intentional balls would distort the usage
    priors the model leans on.
    """
    s = settings or get_settings()
    pattern = str(s.lake_dir / "fact_pitch" / "season=*" / "*.parquet")
    lf = pl.scan_parquet(pattern, hive_partitioning=False).filter(
        pl.col("is_tracked_pitch") & pl.col("is_competitive") & (pl.col("game_type") == "R")
    )
    if seasons:
        lf = lf.filter(pl.col("season").is_in(seasons))
    if pitcher_ids:
        lf = lf.filter(pl.col("pitcher").is_in(pitcher_ids))
    return lf.select(_PASSTHROUGH).collect()


def build_batch_features(
    *,
    seasons: list[int] | None = None,
    pitcher_ids: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    return build_features(
        load_batch_frame(seasons=seasons, pitcher_ids=pitcher_ids, settings=settings)
    )


@dataclass
class PendingPitch:
    """The pre-pitch state the next pitch will be thrown into.

    Deliberately has no field for the pitch itself. Live inference cannot know
    velocity, location, or outcome before release, and encoding that in the type
    means a leaking feature fails as a null rather than as a silent accuracy lift.
    """

    game_pk: int
    at_bat_number: int
    pitch_number: int
    game_date: Any
    season: int
    pitcher: int
    batter: int
    balls: int
    strikes: int
    outs_when_up: int
    inning: int
    base_state: int
    bat_score: int
    fld_score: int
    stand: str
    p_throws: str
    home_team: str
    inning_topbot: str
    n_thruorder_pitcher: int | None = None
    pitcher_days_since_prev_game: int | None = None


@dataclass
class IncrementalState:
    """The live path's accumulator.

    Holds the pitches already thrown (which the live caller must track anyway to
    know the sequence) and appends a synthetic row for the pitch about to be
    thrown. `build_features` then runs over the whole thing exactly as it does in
    batch, and the last row is the live feature vector.

    Seed with `history` from the lake to give the season-to-date priors something
    to work from; without it the first pitch of a live outing has null priors,
    which is correct but weaker.
    """

    history: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={c: None for c in REQUIRED_COLUMNS})
    )

    @classmethod
    def seeded(
        cls,
        pitcher_id: int,
        season: int,
        *,
        settings: Settings | None = None,
    ) -> IncrementalState:
        """Load this pitcher's season-to-date pitches so priors are warm."""
        hist = load_batch_frame(seasons=[season], pitcher_ids=[pitcher_id], settings=settings)
        return cls(history=hist)

    def record(self, completed: dict[str, Any]) -> None:
        """Append a pitch that has now been thrown and resolved."""
        row = {c: completed.get(c) for c in REQUIRED_COLUMNS}
        frame = pl.DataFrame([row])
        if self.history.height:
            frame = frame.cast(self.history.schema)  # type: ignore[arg-type]
        self.history = pl.concat([self.history, frame], how="diagonal_relaxed")

    def features_for(self, pending: PendingPitch) -> pl.DataFrame:
        """Feature vector for the next pitch: one row, same code path as batch."""
        row = {c: None for c in REQUIRED_COLUMNS}
        for k, v in pending.__dict__.items():
            if k in row:
                row[k] = v
        frame = pl.DataFrame([row])
        combined = pl.concat([self.history, frame], how="diagonal_relaxed")
        # Targets are meaningless here — the pitch has not been thrown.
        return build_features(combined, with_targets=False).tail(1)
