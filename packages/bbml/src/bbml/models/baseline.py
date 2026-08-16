"""The baseline: each pitcher's own usage rate, bucketed by count.

This exists so that "the model works" is a falsifiable claim.

A next-pitch model that reports 48% top-1 accuracy sounds impressive until you
notice that always guessing the pitcher's most-used pitch gets 45%. Pitch
selection is dominated by two things a lookup table already knows — who is
throwing and what the count is — and a gradient-boosted model with sixty features
can spend all its capacity rediscovering that while adding nothing.

So the baseline is computed first, its log-loss is published, and
`test_beats_baseline` asserts the trained model improves on it. That turns "did
this model learn anything" from a judgement call into a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import polars as pl

from bbcore.logging import get_logger
from bbml.features.schema import PITCH_TYPES

log = get_logger(__name__)

# Counts collapse into behavioural groups. Pitchers treat 3-0 and 3-1 almost
# identically (get a strike), and 0-2 and 1-2 identically (chase pitch); keeping
# all twelve counts separate just thins the per-cell sample.
COUNT_BUCKETS: dict[tuple[int, int], str] = {
    (0, 0): "first",
    (0, 1): "ahead",
    (0, 2): "putaway",
    (1, 2): "putaway",
    (2, 2): "even_deep",
    (1, 0): "behind",
    (2, 0): "behind",
    (3, 0): "must_strike",
    (3, 1): "must_strike",
    (1, 1): "even",
    (2, 1): "ahead",
    (3, 2): "full",
}

LAPLACE_ALPHA = 1.0


def count_bucket(balls: int, strikes: int) -> str:
    return COUNT_BUCKETS.get((int(balls), int(strikes)), "even")


def count_bucket_expr() -> pl.Expr:
    expr = pl.lit("even")
    for (b, s), name in COUNT_BUCKETS.items():
        expr = (
            pl.when((pl.col("balls") == b) & (pl.col("strikes") == s))
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return expr.alias("count_bucket")


@dataclass
class UsageRateBaseline:
    """P(pitch type | pitcher, count bucket), smoothed, with fallbacks.

    Three tiers, most specific first: pitcher x count bucket, then pitcher
    overall, then league. Falling all the way back matters for a debut pitcher,
    where the specific tiers have no data at all.
    """

    classes: list[str] = field(default_factory=lambda: list(PITCH_TYPES))
    _by_pitcher_count: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    _by_pitcher: dict[int, np.ndarray] = field(default_factory=dict)
    _league: np.ndarray | None = None

    def _index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def fit(self, df: pl.DataFrame, target: str = "target_pitch_type") -> UsageRateBaseline:
        idx = self._index()
        d = df.with_columns(count_bucket_expr()).filter(pl.col(target).is_not_null())

        league = np.full(len(self.classes), LAPLACE_ALPHA)
        for pt, n in d.group_by(target).len().iter_rows():
            if pt in idx:
                league[idx[pt]] += n
        self._league = league / league.sum()

        for (pitcher,), grp in d.group_by(["pitcher"]):
            counts = np.full(len(self.classes), LAPLACE_ALPHA)
            for pt, n in grp.group_by(target).len().iter_rows():
                if pt in idx:
                    counts[idx[pt]] += n
            self._by_pitcher[int(pitcher)] = counts / counts.sum()

        for (pitcher, bucket), grp in d.group_by(["pitcher", "count_bucket"]):
            counts = np.full(len(self.classes), LAPLACE_ALPHA)
            for pt, n in grp.group_by(target).len().iter_rows():
                if pt in idx:
                    counts[idx[pt]] += n
            self._by_pitcher_count[(int(pitcher), str(bucket))] = counts / counts.sum()

        log.info(
            "baseline fitted: %d pitchers, %d pitcher-count cells",
            len(self._by_pitcher),
            len(self._by_pitcher_count),
        )
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self._league is None:
            raise RuntimeError("Baseline is not fitted.")
        d = df.with_columns(count_bucket_expr())
        out = np.empty((d.height, len(self.classes)))
        for i, (pitcher, bucket) in enumerate(
            zip(d["pitcher"].to_list(), d["count_bucket"].to_list(), strict=True)
        ):
            probs = self._by_pitcher_count.get((int(pitcher), str(bucket)))
            if probs is None:
                probs = self._by_pitcher.get(int(pitcher), self._league)
            out[i] = probs
        return out


# --- metrics -----------------------------------------------------------------


def log_loss(y_true: list[str | None], proba: np.ndarray, classes: list[str]) -> float:
    """Multiclass log-loss. Lower is better; this is the headline comparison."""
    idx = {c: i for i, c in enumerate(classes)}
    eps = 1e-15
    total, n = 0.0, 0
    for i, y in enumerate(y_true):
        if y is None or y not in idx:
            continue
        total -= np.log(max(proba[i, idx[y]], eps))
        n += 1
    return total / max(n, 1)


def top_k_accuracy(
    y_true: list[str | None], proba: np.ndarray, classes: list[str], k: int = 1
) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    hits, n = 0, 0
    order = np.argsort(-proba, axis=1)[:, :k]
    for i, y in enumerate(y_true):
        if y is None or y not in idx:
            continue
        if idx[y] in order[i]:
            hits += 1
        n += 1
    return hits / max(n, 1)


def expected_calibration_error(
    y_true: list[str | None], proba: np.ndarray, classes: list[str], bins: int = 10
) -> float:
    """ECE over the predicted-class confidence.

    Calibration matters more than accuracy for this product: the UI shows
    "slider 41%", so 41% has to mean 41%. A model can gain accuracy while
    becoming overconfident, which makes the interface actively misleading.
    """
    idx = {c: i for i, c in enumerate(classes)}
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct, confs = [], []
    for i, y in enumerate(y_true):
        if y is None or y not in idx:
            continue
        correct.append(1.0 if pred[i] == idx[y] else 0.0)
        confs.append(conf[i])
    if not confs:
        return 0.0
    correct_arr, conf_arr = np.array(correct), np.array(confs)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (conf_arr > lo) & (conf_arr <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf_arr)) * abs(correct_arr[m].mean() - conf_arr[m].mean())
    return float(ece)


@dataclass
class Evaluation:
    log_loss: float
    top1: float
    top2: float
    ece: float
    n: int

    def __str__(self) -> str:
        return (
            f"log_loss={self.log_loss:.4f}  top1={self.top1:.3f}  "
            f"top2={self.top2:.3f}  ece={self.ece:.4f}  n={self.n:,}"
        )


def evaluate(y_true: list[str | None], proba: np.ndarray, classes: list[str]) -> Evaluation:
    return Evaluation(
        log_loss=log_loss(y_true, proba, classes),
        top1=top_k_accuracy(y_true, proba, classes, 1),
        top2=top_k_accuracy(y_true, proba, classes, 2),
        ece=expected_calibration_error(y_true, proba, classes),
        n=sum(1 for y in y_true if y is not None),
    )
