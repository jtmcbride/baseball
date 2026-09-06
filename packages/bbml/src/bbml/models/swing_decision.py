"""Model #6: value-maximising swing decisions.

The swing head predicts batting run value conditional on offering.  Taking is
not estimated from outcomes (that would confuse called balls with good takes):
it is the expected count value under model #5's calibrated P(called strike).
The difference grades the observed choice against the better counterfactual.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from bbml.features.run_value import RunValue
from bbml.features.swing_decision import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_SWING,
    TARGET_VALUE,
)

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "num_threads": 0,
}


@dataclass
class SwingDecisionModel:
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    best_iteration: int | None = None

    def _fit_cat_maps(self, df: pl.DataFrame) -> None:
        self.cat_maps = {
            col: {
                str(v): i
                for i, v in enumerate(
                    sorted(map(str, (v for v in df[col].unique().to_list() if v is not None)))
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

    def fit(
        self, train: pl.DataFrame, val: pl.DataFrame | None = None, *, num_boost_round: int = 1500
    ) -> SwingDecisionModel:
        train = train.filter(pl.col(TARGET_SWING) & pl.col(TARGET_VALUE).is_not_null())
        self._fit_cat_maps(train)
        dtrain = lgb.Dataset(
            self._encode(train).to_pandas(),
            label=train[TARGET_VALUE].to_numpy(),
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        valid_sets, valid_names = [dtrain], ["train"]
        callbacks: list[Callable[..., Any]] = [lgb.log_evaluation(period=200)]
        if val is not None:
            val = val.filter(pl.col(TARGET_SWING) & pl.col(TARGET_VALUE).is_not_null())
            if val.height:
                valid_sets.append(
                    lgb.Dataset(
                        self._encode(val).to_pandas(),
                        label=val[TARGET_VALUE].to_numpy(),
                        categorical_feature=CATEGORICAL_FEATURES,
                        reference=dtrain,
                        free_raw_data=False,
                    )
                )
                valid_names.append("val")
                callbacks.append(lgb.early_stopping(50, verbose=False))
        self.booster = lgb.train(
            DEFAULT_PARAMS,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration or num_boost_round
        return self

    def predict_swing_value(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(
            self.booster.predict(self._encode(df).to_pandas(), num_iteration=self.best_iteration)
        )

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
    def load(cls, directory: Path) -> SwingDecisionModel:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError(
                "Saved swing-decision model's feature list does not match the current schema. Retrain it."
            )
        return cls(
            lgb.Booster(model_file=str(directory / "model.txt")),
            meta["cat_maps"],
            meta["best_iteration"],
        )


def score_decisions(
    df: pl.DataFrame, model: SwingDecisionModel, called_strike_model, run_value: RunValue
) -> pl.DataFrame:
    """Attach counterfactual values and non-positive decision value per pitch."""
    swing_value = model.predict_swing_value(df)
    take_value = run_value.expected_take_value(
        df["balls"], df["strikes"], called_strike_model.predict_proba(df)
    )
    best = np.maximum(swing_value, take_value)
    observed = np.where(df[TARGET_SWING].to_numpy(), swing_value, take_value)
    return df.with_columns(
        pl.Series("expected_swing_value", swing_value),
        pl.Series("expected_take_value", take_value),
        pl.Series("decision_value", observed - best),
    )


def decision_value_by_batter(scored: pl.DataFrame, *, min_pitches: int = 200) -> pl.DataFrame:
    return (
        scored.filter(pl.col("decision_value").is_not_null())
        .group_by(["batter", "season"])
        .agg(
            pl.col("decision_value").sum().alias("decision_runs"),
            (100 * pl.col("decision_value").mean()).alias("decision_value_per_100"),
            pl.len().alias("pitches"),
            pl.col(TARGET_SWING).sum().alias("swings"),
        )
        .filter(pl.col("pitches") >= min_pitches)
        .sort("decision_value_per_100", descending=True)
    )
