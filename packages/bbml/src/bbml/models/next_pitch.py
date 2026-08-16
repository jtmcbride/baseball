"""The next-pitch model: one global LightGBM multiclass model for all pitchers.

WHY ONE GLOBAL MODEL RATHER THAN ONE PER PITCHER
-------------------------------------------------
Personalization here comes from FEATURES, not from separate models. `pitcher` is
deliberately not an input; instead each row carries that pitcher's expanding-window
usage rates, so the model conditions on who is throwing without memorizing an id
it cannot generalize from.

This was measured, not assumed. `UsageRateBaseline` is effectively a per-pitcher
model — P(pitch type | pitcher, count bucket) — and on the same test split it
scores 1.5942 log-loss against this model's 1.2736. Per-pitcher modelling loses by
~20% for three reasons:

  * Data. ~740 pitchers over 245k training pitches is ~330 pitches each for a
    15-class problem. Starters get a few thousand a season, relievers under a
    thousand. That is nowhere near enough to fit per-arm.
  * Cold start. Debut pitchers and midseason callups have no history at all, and
    a per-pitcher model has nothing to fall back on.
  * Shared structure. "0-2 counts get breaking balls away, 3-0 gets a fastball"
    is true league-wide. Every per-pitcher model would have to relearn it from
    its own thin slice.

The statistical framing is partial pooling: per-pitcher models are no pooling
(high variance), a global model with no pitcher features is complete pooling
(high bias), and this — global structure plus per-pitcher priors — is the middle
that wins. `blend_with` below adds an optional shrinkage step for more.

WHY THE ARSENAL MASK IS OFF BY DEFAULT
---------------------------------------
Masking predictions to the pitcher's known arsenal sounds obviously right and
measured worse. Against the identical booster: no mask 1.2736 log-loss / 0.0193
ECE; hard mask 1.7433 / 0.0251; even the best soft leak never beat no mask. The
per-pitch-type priors already teach the model each arsenal, so the mask only adds
false certainty about pitches a pitcher rarely-but-genuinely throws. It is kept
as an option, off by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, PITCH_TYPES

log = get_logger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 96,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}

# A pitch must appear this often in a pitcher's history before the mask treats it
# as part of the arsenal. One misclassified pitch in a season is a labeling
# artifact, not a pitch.
ARSENAL_MIN_COUNT = 5

# The mask is a SOFT prior, not a hard constraint — masked classes are scaled
# down by this factor rather than zeroed.
#
# Hard-zeroing was measured and it was much worse: test log-loss went 1.56 -> 1.89
# and ECE 0.026 -> 0.081 against the same booster. Zero asserts *impossible*, and
# when a pitcher throws a pitch outside his learned arsenal — a first changeup of
# the season, a Savant misclassification — log-loss charges the full price of
# that false certainty. Top-1 improves because the argmax gets cleaner, which is
# exactly how this kind of mistake hides: the accuracy metric rewards it while
# the probabilities the UI actually displays get worse.
ARSENAL_LEAK = 0.03


@dataclass
class NextPitchModel:
    classes: list[str] = field(default_factory=lambda: list(PITCH_TYPES))
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    arsenals: dict[int, list[int]] = field(default_factory=dict)
    best_iteration: int | None = None

    # --- encoding ------------------------------------------------------------

    def _fit_cat_maps(self, df: pl.DataFrame) -> None:
        self.cat_maps = {}
        for col in CATEGORICAL_FEATURES:
            vals = [v for v in df[col].unique().to_list() if v is not None]
            self.cat_maps[col] = {str(v): i for i, v in enumerate(sorted(map(str, vals)))}

    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.select(FEATURE_NAMES)
        exprs = []
        for col in CATEGORICAL_FEATURES:
            mapping = self.cat_maps.get(col, {})
            # Unseen categories map to null, which LightGBM handles natively.
            # Mapping them to a shared integer would assert that a rookie's team
            # and an unknown pitch type are the same thing.
            exprs.append(
                pl.col(col)
                .cast(pl.Utf8)
                .replace_strict(mapping, default=None, return_dtype=pl.Int32)
                .alias(col)
            )
        return out.with_columns(exprs)

    def _label(self, y: pl.Series) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes)}
        return np.array([idx.get(v, -1) for v in y.to_list()])

    # --- arsenal -------------------------------------------------------------

    def _fit_arsenals(self, df: pl.DataFrame, target: str) -> None:
        idx = {c: i for i, c in enumerate(self.classes)}
        counts = (
            df.filter(pl.col(target).is_not_null())
            .group_by(["pitcher", target])
            .len()
            .filter(pl.col("len") >= ARSENAL_MIN_COUNT)
        )
        arsenals: dict[int, list[int]] = {}
        for pitcher, pt, _ in counts.iter_rows():
            if pt in idx:
                arsenals.setdefault(int(pitcher), []).append(idx[pt])
        self.arsenals = arsenals
        log.info("arsenals learned for %d pitchers", len(arsenals))

    def _apply_mask(self, proba: np.ndarray, pitchers: list[int]) -> np.ndarray:
        """Down-weight pitches outside the pitcher's arsenal, then renormalize."""
        out = proba.copy()
        for i, p in enumerate(pitchers):
            allowed = self.arsenals.get(int(p))
            # An unknown pitcher keeps the unmasked distribution: we have no
            # evidence about his arsenal, and inventing one would be worse than
            # admitting uncertainty.
            if not allowed:
                continue
            mask = np.zeros(out.shape[1], dtype=bool)
            mask[allowed] = True
            out[i, ~mask] *= ARSENAL_LEAK
            total = out[i].sum()
            if total > 0:
                out[i] /= total
            else:
                out[i] = proba[i]
        return out

    # --- training ------------------------------------------------------------

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        target: str = "target_pitch_type",
        params: dict | None = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 50,
    ) -> NextPitchModel:
        self._fit_cat_maps(train)
        self._fit_arsenals(train, target)

        X, y = self._encode(train), self._label(train[target])
        keep = y >= 0
        dtrain = lgb.Dataset(
            X.filter(keep).to_pandas(),
            label=y[keep],
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )

        p = {**DEFAULT_PARAMS, "num_class": len(self.classes), **(params or {})}
        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if val is not None and val.height:
            Xv, yv = self._encode(val), self._label(val[target])
            keepv = yv >= 0
            dval = lgb.Dataset(
                Xv.filter(keepv).to_pandas(),
                label=yv[keepv],
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
        log.info("trained to iteration %d", self.best_iteration)
        return self

    # --- inference -----------------------------------------------------------

    def predict_proba(self, df: pl.DataFrame, *, mask: bool = False) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        proba = self.booster.predict(
            self._encode(df).to_pandas(), num_iteration=self.best_iteration
        )
        proba = np.asarray(proba)
        if mask and "pitcher" in df.columns:
            proba = self._apply_mask(proba, df["pitcher"].to_list())
        return proba

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
                    "classes": self.classes,
                    "cat_maps": self.cat_maps,
                    "arsenals": {str(k): v for k, v in self.arsenals.items()},
                    "best_iteration": self.best_iteration,
                    "features": FEATURE_NAMES,
                },
                indent=1,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> NextPitchModel:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(
                "Saved model's feature list does not match the current schema. "
                "Retrain rather than silently scoring with mismatched inputs."
            )
        m = cls(classes=meta["classes"])
        m.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        m.cat_maps = meta["cat_maps"]
        m.arsenals = {int(k): v for k, v in meta["arsenals"].items()}
        m.best_iteration = meta["best_iteration"]
        return m
