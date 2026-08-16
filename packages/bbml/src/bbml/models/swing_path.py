"""Swing-path optimality: is this bat plane a good one *for the pitches it meets*?

Two heads over the same swing frame (`features/swing.py`):

  * `whiff`   — P(whiff | swing). Where nearly all of the geometry effect lives.
  * `contact` — E[xwOBA | contact]. Same interaction, much weaker.

WHY A PER-BATTER AVERAGE WOULD NOT ANSWER THE QUESTION
-------------------------------------------------------
The obvious metric — average a hitter's whiff rate, or his attack angle — cannot
separate two very different hitters: one whose swing plane genuinely suits the
pitches he sees, and one who simply sees flatter pitches. Pitch selection and
pitcher quality are baked into any raw average, and they are not properties of
the swing.

`plane_value` is a counterfactual instead. For every swing, the model is scored
twice against the *same pitch* — once at the batter's actual attack angle, once
at a **matched league-median swing** — and the difference is attributed to his
plane. Pitch context (approach angle, location, count, pitch type) is held
fixed, so what comes out is the contribution of swing geometry alone, measured
over exactly the pitches that hitter chose to swing at.

Sign is always **positive = better for the batter**: whiffs avoided for the
whiff head, xwOBA added for the contact head.

WHY "MATCHED", NOT JUST ATTACK_ANGLE FROZEN
--------------------------------------------
The first version of this counterfactual swapped only `attack_angle` to the
league median and froze the rest of the swing (bat speed, tilt, contact point)
at the *actual hitter's own* values. That is not a real swing — a hitter with a
25-degree attack angle also meets the ball in a different place than a hitter
with a 9-degree one (mean contact point +2.5in pull-side vs -7.0in, measured),
and "9-degree attack angle with a 25-degree hitter's contact point" is a
combination the training data barely contains. The model was extrapolating into
that empty space, and on real held-out data it gave the wrong sign on a slice
where the raw whiff-rate gap is unambiguous (11.2% vs 21.6%, every season
2023-2026) — see `test_plane_value_is_positive_when_the_plane_helps`.

`matched_neutral` fixes this: at `fit` time, small single-feature regressors
(`CORRELATED_SWING_FEATURES` ~ `attack_angle`) are fit on the training
population and evaluated at `league_plane`, giving the bat speed/tilt/contact
point a league-median-attack-angle swing typically has. The counterfactual sets
*all* of them together, not `attack_angle` alone. On the same held-out slice
this recovers a plane_value of +0.092 (t~28), matching the raw effect.

A caveat worth stating rather than burying. This is still "what his plane is
worth against this pitch mix, compared to a typical league swing" — not "what
he personally would gain by changing it", since a real re-plane wouldn't
necessarily land him on the population-typical version of everything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.swing import (
    CATEGORICAL_FEATURES,
    CORRELATED_SWING_FEATURES,
    FEATURE_NAMES,
    PLANE_FEATURE,
    TARGET_CONTACT,
    TARGET_WHIFF,
)

log = get_logger(__name__)

ROLES: tuple[str, ...] = ("whiff", "contact")

_OBJECTIVES: dict[str, dict] = {
    "whiff": {"objective": "binary", "metric": "binary_logloss"},
    "contact": {"objective": "regression", "metric": "l2"},
}

BASE_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "num_threads": 0,
}


def target_for(role: str) -> str:
    if role not in ROLES:
        raise KeyError(f"Unknown swing-path role {role!r}; expected one of {ROLES}.")
    return TARGET_WHIFF if role == "whiff" else TARGET_CONTACT


def rows_for(role: str, df: pl.DataFrame) -> pl.DataFrame:
    """The contact head is only defined where contact happened."""
    target = target_for(role)
    if role == "contact":
        df = df.filter(pl.col("is_in_play"))
    return df.filter(pl.col(target).is_not_null())


def _fit_matched_neutral(train: pl.DataFrame, league_plane: float) -> dict[str, float]:
    """The rest of the swing, matched to what's typical at `league_plane`.

    One tiny single-feature regressor per correlated feature — not the main
    booster's job, just "what does bat speed/tilt/contact point look like for a
    swing with this attack angle", evaluated once at the reference point. See
    the module docstring for why this replaces simply freezing the hitter's own
    values.
    """
    x = train.select(PLANE_FEATURE).to_pandas()
    query = pl.DataFrame({PLANE_FEATURE: [league_plane]}).to_pandas()
    params = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_data_in_leaf": 500,
        "verbosity": -1,
        "num_threads": 0,
    }
    matched = {}
    for feat in CORRELATED_SWING_FEATURES:
        ds = lgb.Dataset(x, label=train[feat].to_numpy(), free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=150)
        matched[feat] = float(booster.predict(query)[0])
    return matched


@dataclass
class SwingPathModel:
    role: str = "whiff"
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    best_iteration: int | None = None
    # The league-median swing plane, stored so a grade is reproducible against
    # the same reference it was computed with.
    league_plane: float = 0.0
    # The rest of CORRELATED_SWING_FEATURES, each matched to its typical value
    # at league_plane rather than left at the individual hitter's own — see the
    # module docstring for why freezing them breaks the counterfactual's sign.
    matched_neutral: dict[str, float] = field(default_factory=dict)

    # --- encoding ------------------------------------------------------------

    def _fit_cat_maps(self, df: pl.DataFrame) -> None:
        self.cat_maps = {
            col: {
                str(v): i
                for i, v in enumerate(
                    sorted(str(x) for x in df[col].unique().to_list() if x is not None)
                )
            }
            for col in CATEGORICAL_FEATURES
        }

    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(FEATURE_NAMES).with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .replace_strict(self.cat_maps.get(col, {}), default=None, return_dtype=pl.Int32)
            .alias(col)
            for col in CATEGORICAL_FEATURES
        )

    def _dataset(self, df: pl.DataFrame, *, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        return lgb.Dataset(
            self._encode(df).to_pandas(),
            label=df[target_for(self.role)].to_numpy(),
            categorical_feature=CATEGORICAL_FEATURES,
            reference=reference,
            free_raw_data=False,
        )

    # --- training ------------------------------------------------------------

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        params: dict | None = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
    ) -> SwingPathModel:
        train = rows_for(self.role, train)
        self._fit_cat_maps(train)
        self.league_plane = float(train[PLANE_FEATURE].median())
        self.matched_neutral = _fit_matched_neutral(train, self.league_plane)

        dtrain = self._dataset(train)
        p = {**BASE_PARAMS, **_OBJECTIVES[self.role], **(params or {})}
        callbacks = [lgb.log_evaluation(period=200)]
        valid_sets, valid_names = [dtrain], ["train"]

        if val is not None:
            val = rows_for(self.role, val)
            if val.height:
                valid_sets.append(self._dataset(val, reference=dtrain))
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
        log.info(
            "%s head trained to iteration %d (league plane %.2f deg)",
            self.role,
            self.best_iteration,
            self.league_plane,
        )
        return self

    # --- inference -----------------------------------------------------------

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(
            self.booster.predict(self._encode(df).to_pandas(), num_iteration=self.best_iteration)
        )

    def plane_value(self, df: pl.DataFrame) -> np.ndarray:
        """Per-swing value of this batter's plane vs a matched league-median swing.

        Positive is better for the batter in both roles. See the module
        docstring for what this does and does not license as an inference, and
        for why the reference swing matches bat speed/tilt/contact point to
        `league_plane` rather than freezing the hitter's own.
        """
        actual = self.predict(df)
        overrides = {PLANE_FEATURE: self.league_plane, **self.matched_neutral}
        neutral_df = df.with_columns([pl.lit(v).alias(k) for k, v in overrides.items()])
        neutral = self.predict(neutral_df)
        # Fewer whiffs is better; more xwOBA on contact is better.
        return neutral - actual if self.role == "whiff" else actual - neutral

    def feature_importance(self) -> pl.DataFrame:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        return pl.DataFrame(
            {
                "feature": self.booster.feature_name(),
                "gain": self.booster.feature_importance("gain"),
            }
        ).sort("gain", descending=True)

    # --- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> Path:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"), num_iteration=self.best_iteration)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "role": self.role,
                    "features": FEATURE_NAMES,
                    "cat_maps": self.cat_maps,
                    "best_iteration": self.best_iteration,
                    "league_plane": self.league_plane,
                    "matched_neutral": self.matched_neutral,
                },
                indent=1,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> SwingPathModel:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(
                f"Saved {meta['role']} swing-path model's feature list does not match the "
                "current schema. Retrain rather than silently scoring with mismatched inputs."
            )
        model = cls(role=meta["role"])
        model.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        model.cat_maps = meta["cat_maps"]
        model.best_iteration = meta["best_iteration"]
        model.league_plane = meta["league_plane"]
        model.matched_neutral = meta["matched_neutral"]
        return model


# --- evaluation ---------------------------------------------------------------


def evaluate(model: SwingPathModel, test: pl.DataFrame) -> dict[str, float]:
    part = rows_for(model.role, test)
    y = part[target_for(model.role)].to_numpy().astype(float)
    pred = model.predict(part)
    out = {"n": float(part.height)}
    if model.role == "whiff":
        eps = 1e-9
        p = np.clip(pred, eps, 1 - eps)
        out["log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out["auc"] = _auc(y, pred)
        out["base_rate"] = float(y.mean())
    else:
        resid = y - pred
        out["rmse"] = float(np.sqrt(np.mean(resid**2)))
        var = float(np.var(y))
        out["r2"] = float(1 - np.var(resid) / var) if var > 0 else 0.0
    return out


def plane_value_by_batter(
    model: SwingPathModel, df: pl.DataFrame, *, min_swings: int = 200
) -> pl.DataFrame:
    """Aggregate the counterfactual per batter-season, in per-100-swing units."""
    part = rows_for(model.role, df)
    return (
        part.with_columns(pl.Series("_pv", model.plane_value(part)))
        .group_by(["batter", "season"])
        .agg(
            (100 * pl.col("_pv").mean()).round(3).alias("plane_value_per_100"),
            pl.col(PLANE_FEATURE).mean().round(2).alias("attack_angle"),
            pl.len().alias("swings"),
        )
        .filter(pl.col("swings") >= min_swings)
        .sort("plane_value_per_100", descending=True)
    )


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC — no sklearn import for four lines of arithmetic."""
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
