"""Called-strike probability, and the framing runs it makes possible.

WHAT THIS MODEL IS FOR
-----------------------
`CalledStrikeModel.predict_proba` answers one question per take: given where
the pitch crossed the plate, how it moved, and the count, how likely was a
called strike *if graded by an average umpire, average catcher, and average
pitcher*? `catcher`/`umpire` are deliberately not features — see
`features/called_strike.py`. The gap between that expectation and what actually
happened is the residual every downstream use of this model reads:

  * `framing_runs`, grouped by catcher — the catcher framing map (viz #20).
  * `framing_runs`, grouped by umpire — one input to the umpire zone map
    (viz #13; the other is `umpire_zone_rate`'s spatial view of the same gap).

CALIBRATION IS THE GATE, NOT ACCURACY
---------------------------------------
`framing_runs` sums `(actual - p)` over hundreds of takes per catcher. A model
that ranks pitches perfectly by strike likelihood but is uniformly 2 points
overconfident manufactures a uniform, fake *negative* framing runs number for
every catcher in the league — the errors do not cancel because they share a
sign. `binary_ece` is checked against real held-out data in
`tests/test_called_strike.py` for exactly this reason; a high AUC with poor
calibration is a model that must not be trusted for this purpose even though
it "works" by the usual classification metrics.

WHY THE FRAMING FORMULA REUSES `RunValue` RATHER THAN A NEW TABLE
--------------------------------------------------------------------
`RunValue.marginal_strike_value(balls, strikes)` is the run-expectancy swing
between a take being called a ball vs a strike, entirely derived from the
count run-expectancy table `run_value.py` already fits for Stuff+/Location+/
Pitching+. A called-strike-specific run-value table would be the same twelve
numbers computed a second way from the same plate appearances — reusing the
existing one is not a shortcut, it is the correct amount of work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.called_strike import (
    CATCHER_COLUMN,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_CALLED_STRIKE,
    UMPIRE_COLUMN,
)
from bbml.features.run_value import RunValue

log = get_logger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}


@dataclass
class CalledStrikeModel:
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    best_iteration: int | None = None

    # --- encoding --------------------------------------------------------------

    def _fit_cat_maps(self, df: pl.DataFrame) -> None:
        self.cat_maps = {}
        for col in CATEGORICAL_FEATURES:
            vals = [v for v in df[col].unique().to_list() if v is not None]
            self.cat_maps[col] = {str(v): i for i, v in enumerate(sorted(map(str, vals)))}

    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.select(FEATURE_NAMES)
        return out.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .replace_strict(self.cat_maps.get(col, {}), default=None, return_dtype=pl.Int32)
            .alias(col)
            for col in CATEGORICAL_FEATURES
        )

    # --- training ----------------------------------------------------------------

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        params: dict | None = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> CalledStrikeModel:
        self._fit_cat_maps(train)
        dtrain = lgb.Dataset(
            self._encode(train).to_pandas(),
            label=train[TARGET_CALLED_STRIKE].to_numpy(),
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        p = {**DEFAULT_PARAMS, **(params or {})}
        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets, valid_names = [dtrain], ["train"]

        if val is not None and val.height:
            dval = lgb.Dataset(
                self._encode(val).to_pandas(),
                label=val[TARGET_CALLED_STRIKE].to_numpy(),
                categorical_feature=CATEGORICAL_FEATURES,
                reference=dtrain,
                free_raw_data=False,
            )
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            p,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration or num_boost_round
        log.info("called-strike model trained to iteration %d", self.best_iteration)
        return self

    # --- inference -----------------------------------------------------------

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(
            self.booster.predict(self._encode(df).to_pandas(), num_iteration=self.best_iteration)
        )

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> Path:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"), num_iteration=self.best_iteration)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "features": FEATURE_NAMES,
                    "cat_maps": self.cat_maps,
                    "best_iteration": self.best_iteration,
                },
                indent=1,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> CalledStrikeModel:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(
                "Saved called-strike model's feature list does not match the current "
                "schema. Retrain rather than silently scoring with mismatched inputs."
            )
        model = cls()
        model.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        model.cat_maps = meta["cat_maps"]
        model.best_iteration = meta["best_iteration"]
        return model


# --- evaluation ----------------------------------------------------------------


@dataclass(frozen=True)
class CalledStrikeEval:
    log_loss: float
    auc: float
    ece: float
    n: int


def evaluate(model: CalledStrikeModel, test: pl.DataFrame, *, bins: int = 10) -> CalledStrikeEval:
    y = test[TARGET_CALLED_STRIKE].to_numpy().astype(float)
    p = model.predict_proba(test)
    eps = 1e-12
    pc = np.clip(p, eps, 1 - eps)
    return CalledStrikeEval(
        log_loss=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        auc=_auc(y, p),
        ece=binary_ece(y, p, bins=bins),
        n=len(y),
    )


def binary_ece(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Reliability of the predicted PROBABILITY, not of an argmax call.

    `framing_runs` sums `(actual - p)` over every take, so a probability that
    is off by a constant amount manufactures fake framing value league-wide
    even when the model separates strikes from balls perfectly by rank. That
    is why this — not accuracy or AUC — is the hard validation gate for using
    this model to grade catchers or umpires.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    n = len(p)
    if n == 0:
        return 0.0
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def _auc(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U form — avoids adding scikit-learn as a dependency here."""
    pos, neg = p[y_true == 1], p[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --- framing runs ----------------------------------------------------------------


def framing_runs(
    df: pl.DataFrame,
    model: CalledStrikeModel,
    run_value: RunValue,
    *,
    group_col: str = CATCHER_COLUMN,
    min_pitches: int = 500,
) -> pl.DataFrame:
    """`Sum (actual_strike - P(strike)) * strike_value(count)`, grouped by catcher
    (or by `UMPIRE_COLUMN` for the umpire side of the same residual).

    Positive means more strikes were called than an average umpire/catcher/
    pitcher combination would produce at that location and count, weighted by
    how much a strike was worth there — the standard framing-runs
    construction, credited to whichever column is grouped on.
    """
    p = model.predict_proba(df)
    sv = run_value.marginal_strike_value(df["balls"], df["strikes"])
    actual = df[TARGET_CALLED_STRIKE].cast(pl.Float64).to_numpy()
    credit = (actual - p) * sv
    return (
        df.with_columns(pl.Series("_credit", credit))
        .filter(pl.col("_credit").is_not_null())
        .group_by(group_col)
        .agg(pl.col("_credit").sum().alias("framing_runs"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_pitches)
        .sort("framing_runs", descending=True)
    )


def umpire_zone_rate(df: pl.DataFrame, model: CalledStrikeModel, *, min_pitches: int = 500) -> pl.DataFrame:
    """Per-umpire actual vs. expected called-strike rate on borderline takes.

    Feeds the umpire zone map (viz #13): the spatial companion to
    `framing_runs` grouped by `UMPIRE_COLUMN`. Restricted to
    `0.2 <= P(strike) <= 0.8` — pitches nowhere near the edge of the zone are
    called correctly by every umpire and would just dilute the signal with a
    huge n of uninformative agreement.
    """
    p = model.predict_proba(df)
    borderline = df.with_columns(pl.Series("_p", p)).filter(
        (pl.col("_p") >= 0.2) & (pl.col("_p") <= 0.8)
    )
    return (
        borderline.group_by(UMPIRE_COLUMN)
        .agg(
            pl.col(TARGET_CALLED_STRIKE).mean().alias("actual_rate"),
            pl.col("_p").mean().alias("expected_rate"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_pitches)
        .with_columns((pl.col("actual_rate") - pl.col("expected_rate")).alias("edge"))
        .sort("edge", descending=True)
    )
