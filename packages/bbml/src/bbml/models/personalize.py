"""Per-pitcher personalization on top of the global model.

THE QUESTION THIS ANSWERS
-------------------------
Should there be one model for everyone, or one per pitcher? Measured on the same
test split:

    per-pitcher only (count-bucket lookup)   log-loss 1.5942   top1 0.412   ECE 0.0466
    global model only                        log-loss 1.2736   top1 0.457   ECE 0.0193
    global + shrinkage blend (k=1500)        log-loss 1.2840   top1 0.460   ECE 0.0061

Per-pitcher modelling loses outright — see the `next_pitch` module docstring for
why. But the blend is genuinely interesting: it costs 0.8% log-loss and buys a
**3x improvement in calibration**, plus a little top-1.

That trade is worth taking for this product specifically. The UI's main surface
is a probability bar reading "Slider 41%", and ECE is the metric that says
whether 41% means 41%. A model that is marginally better at ranking but visibly
overconfident makes the interface lie. Use the blend for anything user-facing;
use the raw global model when you are optimizing log-loss for its own sake.

WHY SHRINKAGE RATHER THAN A FIXED WEIGHT
-----------------------------------------
How much to trust a pitcher's own history depends on how much of it there is.
`n / (n + k)` leans on the league-wide model for a pitcher with 40 pitches on
record and on his own tendencies once he has thousands — the standard
empirical-Bayes shrinkage, and the reason this is partial pooling rather than a
hand-tuned constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from bbml.models.baseline import UsageRateBaseline

# Half-weight point: a pitcher needs k prior pitches before his own rates carry
# as much weight as the global model. Chosen by sweeping k over the val split;
# 1500 gave the best calibration without a meaningful log-loss cost.
DEFAULT_SHRINKAGE_K = 1500.0

# Ceiling on the personal term. Even a pitcher with a full season on record does
# not get to override the global model entirely — the count/sequence structure it
# has learned is still worth more than his marginal rates.
MAX_PERSONAL_WEIGHT = 0.4


@dataclass
class PersonalizedBlend:
    """Blends a global model's predictions with each pitcher's own empirical rates."""

    baseline: UsageRateBaseline
    shrinkage_k: float = DEFAULT_SHRINKAGE_K
    max_weight: float = MAX_PERSONAL_WEIGHT

    @classmethod
    def fit(
        cls,
        train: pl.DataFrame,
        *,
        target: str = "target_pitch_type",
        shrinkage_k: float = DEFAULT_SHRINKAGE_K,
        max_weight: float = MAX_PERSONAL_WEIGHT,
    ) -> PersonalizedBlend:
        return cls(
            baseline=UsageRateBaseline().fit(train, target=target),
            shrinkage_k=shrinkage_k,
            max_weight=max_weight,
        )

    def weights(self, df: pl.DataFrame) -> np.ndarray:
        """Per-row weight on the personal term, from how much history exists."""
        if "prior_pitches_seen" in df.columns:
            n = df["prior_pitches_seen"].fill_null(0).to_numpy().astype(float)
        else:
            n = np.zeros(df.height)
        return (n / (n + self.shrinkage_k)) * self.max_weight

    def apply(self, global_proba: np.ndarray, df: pl.DataFrame) -> np.ndarray:
        personal = self.baseline.predict_proba(df)
        w = self.weights(df)[:, None]
        blended = (1.0 - w) * global_proba + w * personal
        return blended / blended.sum(axis=1, keepdims=True)
