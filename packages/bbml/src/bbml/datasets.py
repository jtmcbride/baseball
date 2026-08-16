"""Dataset assembly and splitting, with the leakage rules written as assertions.

Every rule here exists because violating it produces a model that looks excellent
in evaluation and is worthless in use. None of them fail loudly on their own —
that is the point of encoding them as checks rather than as documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.build import ORDER_COLS
from bbml.features.schema import (
    FEATURE_NAMES,
    TARGET_LOCATION,
    TARGET_PITCH_TYPE,
    assert_no_leakage,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame

    def describe(self) -> str:
        def span(d: pl.DataFrame) -> str:
            if d.height == 0:
                return "empty"
            return f"{d.height:,} rows, {d['game_date'].min()}..{d['game_date'].max()}"

        return f"train: {span(self.train)}\nval:   {span(self.val)}\ntest:  {span(self.test)}"


def season_split(
    df: pl.DataFrame,
    *,
    train_through: int,
    val_season: int,
    test_from: int,
) -> Split:
    """The default split: whole seasons, ordered in time.

    Random splitting is catastrophic here and tempting because it "works": pitches
    from the same plate appearance land on both sides, so the model sees the
    sequence it is being asked to predict and scores brilliantly on nothing.
    """
    return Split(
        train=df.filter(pl.col("season") <= train_through),
        val=df.filter(pl.col("season") == val_season),
        test=df.filter(pl.col("season") >= test_from),
    )


def date_split(df: pl.DataFrame, *, train_frac: float = 0.7, val_frac: float = 0.15) -> Split:
    """Chronological split by date, for when only part of one season exists.

    Splits on DATE BOUNDARIES, never on row position: cutting mid-game would put
    the first half of a plate appearance in train and the second half in test,
    which is the same leak as random splitting wearing a chronological disguise.
    """
    if df.height == 0:
        return Split(df, df, df)

    dates = df["game_date"].unique().sort()
    n = len(dates)
    train_end = dates[max(0, int(n * train_frac) - 1)]
    val_end = dates[max(0, int(n * (train_frac + val_frac)) - 1)]

    return Split(
        train=df.filter(pl.col("game_date") <= train_end),
        val=df.filter((pl.col("game_date") > train_end) & (pl.col("game_date") <= val_end)),
        test=df.filter(pl.col("game_date") > val_end),
    )


def pitcher_holdout_split(df: pl.DataFrame, *, holdout_frac: float = 0.2, seed: int = 0) -> Split:
    """Hold out whole pitchers, to measure generalization to unseen arms.

    A time split still lets the model memorize individual pitchers it has seen for
    years. This answers the different question: how does it do on a rookie called
    up tomorrow? Use it as a second evaluation, not a replacement.
    """
    pitchers = df["pitcher"].unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    held = set(
        rng.choice(pitchers, size=max(1, int(len(pitchers) * holdout_frac)), replace=False).tolist()
    )
    in_holdout = pl.col("pitcher").is_in(list(held))
    return Split(
        train=df.filter(~in_holdout),
        val=df.filter(in_holdout).sample(fraction=0.5, seed=seed),
        test=df.filter(in_holdout),
    )


# --- leakage assertions ------------------------------------------------------


def assert_split_is_temporal(split: Split) -> None:
    """Train must end before val starts, and val before test."""
    for earlier, later, names in (
        (split.train, split.val, ("train", "val")),
        (split.val, split.test, ("val", "test")),
    ):
        if earlier.height == 0 or later.height == 0:
            continue
        emax, lmin = earlier["game_date"].max(), later["game_date"].min()
        if emax >= lmin:  # type: ignore[operator]
            raise AssertionError(
                f"{names[0]} ends {emax} but {names[1]} starts {lmin} — the split "
                "overlaps in time, so the model can see its own future."
            )


def assert_no_plate_appearance_straddles(split: Split) -> None:
    """No plate appearance may appear in more than one split.

    Sequence features make a straddled PA a direct leak: the test row's
    `prev_pitch_type_1` is a pitch the model was trained on.
    """

    def pa_keys(d: pl.DataFrame) -> set[tuple]:
        if d.height == 0:
            return set()
        return set(zip(d["game_pk"].to_list(), d["at_bat_number"].to_list(), strict=True))

    tr, va, te = pa_keys(split.train), pa_keys(split.val), pa_keys(split.test)
    for a, b, names in (
        (tr, va, ("train", "val")),
        (tr, te, ("train", "test")),
        (va, te, ("val", "test")),
    ):
        shared = a & b
        if shared:
            raise AssertionError(
                f"{len(shared)} plate appearances appear in both {names[0]} and "
                f"{names[1]} (e.g. {sorted(shared)[:3]})."
            )


def assert_features_are_clean(df: pl.DataFrame) -> None:
    """No feature may describe the pitch being predicted."""
    assert_no_leakage(FEATURE_NAMES)
    present = [c for c in FEATURE_NAMES if c in df.columns]
    if len(present) != len(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(present))
        raise AssertionError(f"Feature frame is missing {missing}")


def validate(split: Split, *, check_features: bool = True) -> None:
    """Run every leakage check. Called before any training run.

    `check_features=False` is for the pitch-quality models (`features/stuff.py`),
    whose inputs are deliberately the thrown pitch itself. The temporal and
    plate-appearance checks still apply to them and are the ones that matter
    there; only the next-pitch feature contract does not.
    """
    assert_split_is_temporal(split)
    assert_no_plate_appearance_straddles(split)
    if check_features:
        for part in (split.train, split.val, split.test):
            if part.height:
                assert_features_are_clean(part)
    log.info("split validated\n%s", split.describe())


# --- assembly ----------------------------------------------------------------


def prepare(
    df: pl.DataFrame, *, target: str = TARGET_PITCH_TYPE, drop_null_targets: bool = True
) -> pl.DataFrame:
    """Sort chronologically and drop rows whose target is unknown."""
    out = df.sort(ORDER_COLS)
    if drop_null_targets and target in out.columns:
        before = out.height
        out = out.filter(pl.col(target).is_not_null())
        if out.height != before:
            log.info("dropped %d rows with a null %s", before - out.height, target)
    return out


def auto_split(df: pl.DataFrame, *, check_features: bool = True) -> Split:
    """Pick the widest temporally-valid split the available data supports.

    Whole-season splits need at least three seasons. With less — which is the
    normal state during a partial backfill — fall back to a chronological
    date split rather than silently producing an empty validation set.
    """
    seasons = sorted(df["season"].unique().to_list())
    if len(seasons) >= 3:
        split = season_split(
            df,
            train_through=seasons[-3],
            val_season=seasons[-2],
            test_from=seasons[-1],
        )
        log.info("season split over %s", seasons)
    else:
        split = date_split(df)
        log.info("only %d season(s) available — using a chronological date split", len(seasons))
    validate(split, check_features=check_features)
    return split


def xy(df: pl.DataFrame, target: str = TARGET_PITCH_TYPE) -> tuple[pl.DataFrame, pl.Series]:
    assert_features_are_clean(df)
    return df.select(FEATURE_NAMES), df[target]


__all__ = [
    "TARGET_LOCATION",
    "TARGET_PITCH_TYPE",
    "Split",
    "auto_split",
    "date_split",
    "pitcher_holdout_split",
    "prepare",
    "season_split",
    "validate",
    "xy",
]
