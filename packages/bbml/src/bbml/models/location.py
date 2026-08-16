"""The location model: where in (and around) the zone the next pitch goes.

Shares its shape with `NextPitchModel` — same feature set, same LightGBM
multiclass setup — but predicts the 26-class location grid (`schema.py`)
instead of pitch type, and carries no arsenal mask: "which corner a pitcher
misses to" isn't a per-pitcher categorical fact the way "does he throw a
splitter" is, so there is nothing analogous to gate on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, N_LOCATION_CLASSES

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


@dataclass
class LocationModel:
    n_classes: int = N_LOCATION_CLASSES
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    best_iteration: int | None = None

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
            exprs.append(
                pl.col(col)
                .cast(pl.Utf8)
                .replace_strict(mapping, default=None, return_dtype=pl.Int32)
                .alias(col)
            )
        return out.with_columns(exprs)

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        target: str = "target_location",
        params: dict | None = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 50,
    ) -> LocationModel:
        self._fit_cat_maps(train)

        X, y = self._encode(train), train[target].to_numpy()
        keep = ~np.isnan(y.astype(float))
        y = y.astype(np.int32)
        dtrain = lgb.Dataset(
            X.filter(pl.Series(keep)).to_pandas(),
            label=y[keep],
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )

        p = {**DEFAULT_PARAMS, "num_class": self.n_classes, **(params or {})}
        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if val is not None and val.height:
            Xv = self._encode(val)
            yv = val[target].to_numpy()
            keepv = ~np.isnan(yv.astype(float))
            yv = yv.astype(np.int32)
            dval = lgb.Dataset(
                Xv.filter(pl.Series(keepv)).to_pandas(),
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
        log.info("location model trained to iteration %d", self.best_iteration)
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        proba = self.booster.predict(
            self._encode(df).to_pandas(), num_iteration=self.best_iteration
        )
        return np.asarray(proba)

    def save(self, directory: Path) -> Path:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"), num_iteration=self.best_iteration)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "n_classes": self.n_classes,
                    "cat_maps": self.cat_maps,
                    "best_iteration": self.best_iteration,
                    "features": FEATURE_NAMES,
                },
                indent=1,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> LocationModel:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(
                "Saved model's feature list does not match the current schema. "
                "Retrain rather than silently scoring with mismatched inputs."
            )
        m = cls(n_classes=meta["n_classes"])
        m.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        m.cat_maps = meta["cat_maps"]
        m.best_iteration = meta["best_iteration"]
        return m
