"""Stuff+ / Location+ / Pitching+ — one regressor, three feature sets, one scale.

Each model regresses `run_value.rv_pitcher` on the feature set named by its role
(`features/stuff.py`), then maps its prediction onto the familiar `+` scale where
100 is average and higher is better.

WHY A TINY R-SQUARED IS THE EXPECTED RESULT, NOT A FAILURE
-----------------------------------------------------------
A single pitch's run value is mostly not about the pitch. The batter swings or
doesn't, the ball finds a glove or a gap. Even a perfect model of pitch quality
would explain a few percent of per-pitch variance, so R^2 here lands near 0.001
for the stuff head and the number is close to meaningless as a quality signal —
chasing it upward would mean adding context features and quietly rebuilding a
situation model. It was chased anyway, to check: 500 / 1500 / 2000 rounds and a
finer-grained booster all made test R^2 NEGATIVE and drove `stability()` down
from 0.86 to 0.70. The early-stopped ~36-iteration fit is not undertrained; the
signal genuinely saturates there and everything after it is batted-ball luck
being memorized. Do not "fix" the small iteration count.

What matters is whether the *ranking* is real and holds up out of sample on the
aggregate anyone actually reads. Two evaluations answer that, and they are the
headline rather than R^2:

  * `stability()` — the year-over-year correlation of the grade against the
    year-over-year correlation of the results it grades. This is the entire
    reason a stuff metric exists. Measured on 2015 -> 2016 (319 pitchers, 500+
    pitches each): Stuff+ 0.86, Pitching+ 0.75, Location+ 0.65, against 0.38 for
    observed run value itself. The grade says the same thing about a pitcher two
    years running more than twice as reliably as his own results do.
  * `predictive_validity()` — does this season's grade beat this season's results
    at calling next season's results? On the same pairs Stuff+ runs about level
    with past run value (0.37 vs 0.38) rather than beating it. Worth stating
    plainly: on this partial lake, Stuff+ matches past performance as a predictor
    and is far more stable; it does not yet dominate it. The available split is
    also a 9-year jump (train <=2016, test 2025) across an era in which the
    sweeper was invented, so re-measure on a contiguous split once the full
    backfill lands before drawing a conclusion from this.

THE `+` SCALE
-------------
`plus = 100 + 10 * z(predicted rv)`, with the mean and SD taken over
**pitcher x season x pitch_type groups of at least `CALIBRATION_MIN_PITCHES`
pitches** in the training data, not over individual pitches. That choice is what
makes the number readable the way published Stuff+ is: "his slider is a 120"
means a slider a fair way above the average pitch-type-season, and pitcher totals
land in the familiar 80-130 band. Calibrating on individual pitches instead would
shrink every aggregate toward 100 and make a genuinely elite arsenal read as 103.

The constants live in the saved artifact, so scores from different runs are only
comparable when they came from the same version — which is the honest position,
since a retrain on more seasons moves the reference population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.stuff import TARGET_RUN_VALUE, categorical_features, feature_names

log = get_logger(__name__)

# Regularized harder than the classification heads. The target is dominated by
# irreducible noise, so a booster given a long leash fits batted-ball luck: deep
# trees on 2M rows will happily carve out leaves that "explain" where 40 ground
# balls happened to go.
DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "num_threads": 0,
}

CALIBRATION_GROUP = ["pitcher", "season", "pitch_type"]
CALIBRATION_MIN_PITCHES = 100


@dataclass
class PitchQualityModel:
    role: str = "stuff"
    booster: lgb.Booster | None = None
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    best_iteration: int | None = None
    plus_mean: float = 0.0
    plus_sd: float = 1.0

    @property
    def features(self) -> list[str]:
        return feature_names(self.role)

    @property
    def categoricals(self) -> list[str]:
        return categorical_features(self.role)

    # --- encoding ------------------------------------------------------------

    def _fit_cat_maps(self, df: pl.DataFrame) -> None:
        self.cat_maps = {}
        for col in self.categoricals:
            vals = [v for v in df[col].unique().to_list() if v is not None]
            self.cat_maps[col] = {str(v): i for i, v in enumerate(sorted(map(str, vals)))}

    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.select(self.features)
        # Unseen categories become null, which LightGBM handles natively. Folding
        # them into a shared code would assert that a newly named pitch type is
        # the same thing as some arbitrary existing one.
        return out.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .replace_strict(self.cat_maps.get(col, {}), default=None, return_dtype=pl.Int32)
            .alias(col)
            for col in self.categoricals
        )

    def _dataset(
        self, df: pl.DataFrame, target: str, *, reference: lgb.Dataset | None = None
    ) -> lgb.Dataset:
        part = df.filter(pl.col(target).is_not_null())
        return lgb.Dataset(
            self._encode(part).to_pandas(),
            label=part[target].to_numpy(),
            categorical_feature=self.categoricals,
            reference=reference,
            free_raw_data=False,
        )

    # --- training ------------------------------------------------------------

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        target: str = TARGET_RUN_VALUE,
        params: dict | None = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
    ) -> PitchQualityModel:
        self._fit_cat_maps(train)
        dtrain = self._dataset(train, target)
        p = {**DEFAULT_PARAMS, **(params or {})}
        callbacks = [lgb.log_evaluation(period=200)]
        valid_sets, valid_names = [dtrain], ["train"]

        if val is not None and val.height:
            valid_sets.append(self._dataset(val, target, reference=dtrain))
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
        log.info("%s model trained to iteration %d", self.role, self.best_iteration)
        return self.calibrate(train)

    def calibrate(self, df: pl.DataFrame) -> PitchQualityModel:
        """Set the `+` scale from the reference population. See module docstring."""
        groups = (
            df.with_columns(pl.Series("_pred", self.predict_rv(df)))
            .group_by(CALIBRATION_GROUP)
            .agg(pl.col("_pred").mean().alias("rv"), pl.len().alias("n"))
            .filter(pl.col("n") >= CALIBRATION_MIN_PITCHES)
        )
        if groups.height < 30:
            raise ValueError(
                f"Only {groups.height} pitcher-season-pitch_type groups clear "
                f"{CALIBRATION_MIN_PITCHES} pitches — too few to define a stable "
                "reference population for the + scale."
            )
        self.plus_mean = float(groups["rv"].mean())
        sd = float(groups["rv"].std())
        self.plus_sd = sd if sd > 0 else 1.0
        log.info(
            "%s+ calibrated on %d groups (mean rv %.5f, sd %.5f)",
            self.role,
            groups.height,
            self.plus_mean,
            self.plus_sd,
        )
        return self

    # --- inference -----------------------------------------------------------

    def predict_rv(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(
            self.booster.predict(self._encode(df).to_pandas(), num_iteration=self.best_iteration)
        )

    def plus(self, df: pl.DataFrame) -> np.ndarray:
        """Per-pitch `+` score. Average it to grade an arsenal or a pitcher."""
        return 100.0 + 10.0 * (self.predict_rv(df) - self.plus_mean) / self.plus_sd

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
                    "features": self.features,
                    "cat_maps": self.cat_maps,
                    "best_iteration": self.best_iteration,
                    "plus_mean": self.plus_mean,
                    "plus_sd": self.plus_sd,
                },
                indent=1,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> PitchQualityModel:
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(role=meta["role"])
        if meta["features"] != model.features:
            raise ValueError(
                f"Saved {meta['role']} model's feature list does not match the current "
                "schema. Retrain rather than silently scoring with mismatched inputs."
            )
        model.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        model.cat_maps = meta["cat_maps"]
        model.best_iteration = meta["best_iteration"]
        model.plus_mean = meta["plus_mean"]
        model.plus_sd = meta["plus_sd"]
        return model


# --- evaluation --------------------------------------------------------------


@dataclass(frozen=True)
class QualityEval:
    rmse: float
    r2: float
    # Spearman between per-pitch prediction and the observed target. Robust to the
    # target's heavy tails in a way Pearson is not.
    rank_corr: float
    n: int


def evaluate(model: PitchQualityModel, test: pl.DataFrame) -> QualityEval:
    part = test.filter(pl.col(TARGET_RUN_VALUE).is_not_null())
    y = part[TARGET_RUN_VALUE].to_numpy()
    pred = model.predict_rv(part)
    resid = y - pred
    var = float(np.var(y))
    return QualityEval(
        rmse=float(np.sqrt(np.mean(resid**2))),
        r2=float(1.0 - np.var(resid) / var) if var > 0 else 0.0,
        rank_corr=_spearman(pred, y),
        n=part.height,
    )


def aggregate_correlation(
    model: PitchQualityModel, test: pl.DataFrame, *, min_pitches: int = 100
) -> tuple[float, int]:
    """Does the grade track observed run value at the level the UI displays it?

    Per-pitch correlation is swamped by outcome noise. This groups the test set
    into pitcher x pitch_type buckets — the grain of the arsenal table — and asks
    whether the graded order matches the observed order there.
    """
    part = test.filter(pl.col(TARGET_RUN_VALUE).is_not_null())
    grouped = (
        part.with_columns(pl.Series("_pred", model.predict_rv(part)))
        .group_by(["pitcher", "pitch_type"])
        .agg(
            pl.col("_pred").mean().alias("grade"),
            pl.col(TARGET_RUN_VALUE).mean().alias("actual"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_pitches)
    )
    return (
        _spearman(grouped["grade"].to_numpy(), grouped["actual"].to_numpy()),
        grouped.height,
    )


def _pitcher_seasons(model: PitchQualityModel, df: pl.DataFrame, min_pitches: int) -> pl.DataFrame:
    part = df.filter(pl.col(TARGET_RUN_VALUE).is_not_null())
    return (
        part.with_columns(pl.Series("grade", model.predict_rv(part)))
        .group_by(["pitcher", "season"])
        .agg(
            pl.col("grade").mean(),
            pl.col(TARGET_RUN_VALUE).mean().alias("actual"),
            pl.len().alias("pitches"),
        )
        .filter(pl.col("pitches") >= min_pitches)
    )


def _consecutive_pairs(seasonal: pl.DataFrame) -> pl.DataFrame:
    nxt = seasonal.select(
        pl.col("pitcher"),
        (pl.col("season") - 1).alias("season"),
        pl.col("grade").alias("next_grade"),
        pl.col("actual").alias("next_actual"),
    )
    return seasonal.join(nxt, on=["pitcher", "season"], how="inner")


def stability(
    model: PitchQualityModel, df: pl.DataFrame, *, min_pitches: int = 500
) -> pl.DataFrame:
    """Year-over-year reliability of the grade, beside that of the results it grades.

    The number a stuff metric lives or dies on. A grade that swings as wildly
    season to season as the run value it is meant to explain has added nothing;
    the whole claim is that it sees a stable property the results only sample.
    """
    return _corr_by_season(
        _consecutive_pairs(_pitcher_seasons(model, df, min_pitches)),
        {f"{model.role}_yoy": ("grade", "next_grade"), "own_rv_yoy": ("actual", "next_actual")},
    )


def predictive_validity(
    model: PitchQualityModel, df: pl.DataFrame, *, min_pitches: int = 500
) -> pl.DataFrame:
    """Does this season's grade beat this season's results at calling next season?

    For every pitcher with enough pitches in consecutive seasons, correlate
    season N's grade against season N+1's ACTUAL run value, beside the same
    correlation for season N's own actual run value. A metric that cannot at
    least match "he was good last year" is not adding anything.
    """
    return _corr_by_season(
        _consecutive_pairs(_pitcher_seasons(model, df, min_pitches)),
        {
            f"{model.role}_vs_next": ("grade", "next_actual"),
            "own_rv_vs_next": ("actual", "next_actual"),
        },
    )


def _corr_by_season(
    paired: pl.DataFrame, pairs: dict[str, tuple[str, str]], *, min_pitchers: int = 20
) -> pl.DataFrame:
    """Spearman per season-pair. Spearman, not Pearson: pitcher-season run value
    has a long left tail that a handful of blow-up seasons would otherwise own."""
    rows = []
    for (season,), part in paired.group_by("season", maintain_order=True):
        if part.height < min_pitchers:
            continue
        row: dict[str, object] = {"season": season, "pitchers": part.height}
        for label, (a, b) in pairs.items():
            row[label] = _spearman(part[a].to_numpy(), part[b].to_numpy())
        rows.append(row)
    return pl.DataFrame(rows).sort("season") if rows else pl.DataFrame()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — scipy's `rankdata` without the dependency."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks
