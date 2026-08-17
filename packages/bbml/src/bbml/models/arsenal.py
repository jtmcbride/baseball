"""Arsenal re-classification (M3 model #2): what does this pitcher actually throw?

Savant's own `pitch_type` comes from an automated classifier, and it is known
to blur boundaries between physically similar shapes — a slider and a sweeper
differ mainly by degree (horizontal break, spin axis) rather than by kind, so a
pitcher who throws one real, blended shape can get split across both labels,
and conversely two pitchers' genuinely different sliders can get merged under
one. This module re-checks each pitcher-season's arsenal against Savant's own
labels via pairwise Gaussian-mixture evidence tests, and reports where and how
much they disagree — see `marts.py::build_arsenal_cluster_mart` for where that
comparison lands.

**Measured, not assumed — the first version of this was unsupervised from
scratch, and it was wrong.** Fitting a GMM with k = 1..6 chosen by BIC over
each pitcher-season's *own* pitches (no anchor to Savant's labels) essentially
never converged to a small k: 632 of 640 qualifying 2025 pitcher-seasons hit
the component ceiling exactly, and raising the ceiling just moved where BIC
saturated. Switching to `covariance_type="full"` (pricing the real
velocity/movement correlation within one pitch type, which diagonal
covariance was letting BIC "explain" by adding components) and capping the
fit to a bounded subsample (countering BIC's complexity penalty growing only
with log(n) while a starter's pitch count does not) narrowed it, but did not
fix it: real pitchers still came back with e.g. a single fastball split into
two near-identical clusters (94.97mph/18.5in IVB vs 95.45mph/17.6in IVB —
the same pitch, not two) because *any* two-component fit on thousands of
points finds a small, real, spurious likelihood gain from day-to-day
variation that a free-for-all k-sweep has no reason to resist. The problem
was the search space, not just the covariance shape or the sample cap:
letting BIC discover an unconstrained number of clusters from nothing gives
it too much room to reward noise.

WHY THIS RE-CHECKS SAVANT'S LABELS INSTEAD OF DISCOVERING ARSENALS FROM SCRATCH
----------------------------------------------------------------------------------
The fix is to anchor the search: start from Savant's own `pitch_type` groups
for a pitcher-season (already a strong, mostly-correct starting partition —
the point of this model is to catch its mistakes, not to distrust all of it),
then ask two narrow, well-powered questions instead of one wide, noisy one:

  1. **Merge:** for every pair of Savant-labeled groups, does a single
     Gaussian actually fit their union about as well as two do? If so, Savant
     over-split one real pitch into two labels — merge them.
  2. **Split:** for each resulting group, does it clearly contain two
     sub-populations? If so, Savant under-split (or a genuine second pitch
     is hiding under one label) — split it.

Both questions are decided by the same test (`_prefers_two_components`): fit
one component and two components on the same points, and require the
two-component solution to beat the one-component BIC by
`BIC_EVIDENCE_THRESHOLD` — not merely lower, clearly lower. That threshold
(10 BIC units) is Kass & Raftery's (1995) conventional cutoff for "very
strong evidence" from a Bayes-factor-style comparison, not a tuned magic
number: it is deliberately a high bar, because the earlier failure mode was
BIC rewarding *any* improvement, however small. A pairwise, anchored test
also can't run away to six clusters for a two-pitch reliever the way an
unconstrained 1-6 sweep did — there is no path to "discover" a cluster that
didn't start as one of Savant's own labels (via merge) or a strongly
evidenced split of one.

**Measured again — the anchored merge/split design still over-split, for a
different reason.** Re-checking a real pitcher's single Savant-labeled `FF`
group found overwhelming evidence for two components (a BIC gap of 200, twenty
times `BIC_EVIDENCE_THRESHOLD`) at nearly identical shape (94.98mph/18.5in IVB
vs 95.43mph/17.5in IVB — not a distinguishable pitch to begin with). The two
"clusters" turned out to be almost perfectly separated by *date*: one ran
2025-03-27 to 2025-06-10, the other 2025-06-07 to 2025-07-29, essentially
non-overlapping. That is a within-season drift (mechanical tweak, warm-weather
velocity gain, fatigue) showing up as if it were a second concurrently-thrown
pitch, and no amount of raising `BIC_EVIDENCE_THRESHOLD` fixes it — the
statistical evidence for "two Gaussians" is real, it is just evidence for the
wrong thing. `_split_group` now rejects a split whose two halves are
separated by `game_date` about as well as a rank-based AUC of
`DATE_CONFOUND_AUC` (measured on this exact case: 0.002, i.e. essentially
perfect chronological separation) — a genuinely concurrent second pitch
should not line up with the calendar that cleanly. This is deliberately
applied only to splits, not merges: a merge can only ever collapse a
distinction, never invent one, so an unlucky date correlation between two
already-Savant-labeled groups is not a risk worth guarding against the same
way.

READING THE OUTPUT AGAINST SAVANT, NOT REPLACING IT
------------------------------------------------------
Each final cluster's `label` is Savant's own majority pitch_type among the
pitches in it (disambiguated with a `-2`/`-3` suffix, ordered hardest first,
when a split leaves two clusters under one original label) — so the output
reads like an ordinary arsenal table, not an opaque cluster id, while
`purity` (the fraction of a cluster that actually carries its majority label)
and `n_savant_labels` (how many distinct Savant labels got folded into it)
say exactly where and how much this model disagrees. `arsenal_size_diff`
(final cluster count minus Savant's own count of pitch types thrown at
meaningful usage) is the one-number summary: positive means a split happened
(Savant under-split, likely a blurred boundary); negative means a merge
happened (Savant over-split one real pitch into two labels).

CAVEAT: MORE SAVANT LABELS MEANS MORE CHANCES FOR A FALSE POSITIVE
-----------------------------------------------------------------------
`BIC_EVIDENCE_THRESHOLD` is a per-test bar, not corrected for how many pairwise
merge tests a pitcher-season runs — a two-pitch reliever gets one merge test,
an eight-pitch kitchen-sink starter gets 28 (`C(8,2)`) plus 8 split tests. That
is a real multiple-comparisons exposure, not hypothetical: a spot-check of a
real 2025 eight-pitch-type pitcher found two additional splits (a changeup and
a slider each into two clean, well-separated, non-date-confounded clusters)
that are individually well-evidenced but plausible as either genuine shape
variants or as the kind of borderline case more tests makes more likely to
surface at all. No correction (Bonferroni or otherwise) is applied — with
usually well under ten labels per pitcher-season the inflation is modest, and
a stricter per-test bar would just re-introduce the under-splitting failure
mode this design already fixed once. Read a lone split/merge on an otherwise
unremarkable pitcher-season with a little more skepticism than one that is
part of a pattern (e.g. `mart_pitcher_arsenal_clusters` rows should agree
year over year for a real pitch, the same reliability check the called-strike
model's YoY catcher-framing test already establishes as this codebase's
standard for "is this measuring something real" — not yet run here).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from bbcore.logging import get_logger
from bbml.features.arsenal import CLUSTER_FEATURE_NAMES

log = get_logger(__name__)

# A pitcher-season below this many pitches doesn't have enough data to trust
# any merge/split evidence test — matches the season-grain qualifier every
# other per-entity-season mart in this codebase uses (`MIN_CATCHER_PITCHES`,
# `MIN_BATTER_SWINGS`).
MIN_PITCHES = 200

# Measured, not assumed: mean |arsenal_size_diff| is 2.8-3.2 for 2016-2019
# (frac within +-1 of Savant: 14-23%) and drops to a clean step-change of
# 0.8-1.2 for 2020-2026 (frac within +-1: 65-83%) -- exactly the boundary
# where Statcast unified every park onto one Hawk-Eye tracking system. Full
# story in `features/arsenal.py`'s "USABLE RANGE IS 2020+" section.
# `build_arsenal_cluster_mart` defaults to this floor when no explicit season
# range is given; the clustering functions themselves stay season-agnostic
# (an earlier season CAN be requested explicitly), since the unreliability is
# a data-quality fact about the input, not something this model's logic can
# fix.
MIN_RELIABLE_SEASON = 2020

# Kass & Raftery (1995) "very strong evidence" cutoff for a BIC gap between a
# 1-component and 2-component fit on the same points. Deliberately a high
# bar — see the module docstring for why a lower one over-splits.
BIC_EVIDENCE_THRESHOLD = 10.0

# A merge/split evidence test is fit on at most this many points (a
# deterministic random subsample), never an uncapped group — countering BIC's
# complexity penalty growing only with log(n) while a real pitcher-season's
# pitch count does not (see module docstring).
MAX_FIT_PITCHES = 500

# A proposed split is rejected if either resulting sub-group would be smaller
# than this — otherwise a strongly-evidenced but tiny sliver (a handful of
# pitches thrown differently on purpose, e.g. a show-me changeup) reports as
# a whole extra pitch.
MIN_SPLIT_SHARE = 0.10
MIN_SPLIT_PITCHES = 30

# If `game_date` alone separates a candidate split's two halves about this
# well (rank-based AUC, either direction), the split is rejected as
# within-season drift rather than a second concurrently-thrown pitch — see
# the module docstring's second "measured, not assumed" section.
DATE_CONFOUND_AUC = 0.85

# A Savant pitch_type thrown below this share of a pitcher-season's pitches
# doesn't count toward "how many pitches Savant says this pitcher threw" —
# otherwise a single mislabeled pitch would inflate the comparison this model
# is built to make.
SAVANT_MIN_USAGE = 0.05

N_INIT = 4
RANDOM_STATE = 0


def _subsample(x: np.ndarray, cap: int) -> np.ndarray:
    if x.shape[0] <= cap:
        return x
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(x.shape[0], size=cap, replace=False)
    return x[idx]


def _prefers_two_components(x: np.ndarray, means_init: np.ndarray | None = None) -> bool:
    """True if two components clearly beat one, by `BIC_EVIDENCE_THRESHOLD`.

    `means_init` seeds the two-component fit at the two groups actually being
    compared (merge test) rather than a random init, giving the split a fair
    chance to recover the exact boundary Savant already proposed instead of
    landing on some other two-way cut of the same points.
    """
    x_fit = _subsample(x, MAX_FIT_PITCHES)
    if x_fit.shape[0] < 2 * MIN_SPLIT_PITCHES:
        return False  # not enough evidence to support a second component at all

    one = GaussianMixture(
        n_components=1, covariance_type="full", reg_covar=1e-3, random_state=RANDOM_STATE
    ).fit(x_fit)
    if means_init is not None:
        two = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-3,
            means_init=means_init,
            n_init=1,
            random_state=RANDOM_STATE,
        ).fit(x_fit)
    else:
        two = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        ).fit(x_fit)
    return (one.bic(x_fit) - two.bic(x_fit)) > BIC_EVIDENCE_THRESHOLD


def _merge_groups(x: np.ndarray, groups: list[np.ndarray]) -> list[np.ndarray]:
    """Union-find over Savant's own label groups: merge any pair whose union
    is better explained by one Gaussian than two."""
    n_groups = len(groups)
    parent = list(range(n_groups))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            idx = np.concatenate([groups[i], groups[j]])
            if idx.shape[0] < 2 * MIN_SPLIT_PITCHES:
                # Too little combined evidence to justify keeping them apart.
                parent[find(i)] = find(j)
                continue
            means_init = np.array([x[groups[i]].mean(axis=0), x[groups[j]].mean(axis=0)])
            if not _prefers_two_components(x[idx], means_init=means_init):
                parent[find(i)] = find(j)

    merged: dict[int, list[int]] = {}
    for i in range(n_groups):
        merged.setdefault(find(i), []).append(i)
    return [np.concatenate([groups[i] for i in members]) for members in merged.values()]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — works on dates as readily as numbers."""
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


def _rank_auc(values: np.ndarray, group: np.ndarray) -> float:
    """Mann-Whitney-style AUC: P(a random `group` value > a random non-`group`
    value). 0.5 means `values` doesn't separate the two groups at all; near 0
    or 1 means it almost perfectly does."""
    ranks = _rankdata(values)
    n1 = int(group.sum())
    n2 = len(group) - n1
    if n1 == 0 or n2 == 0:
        return 0.5
    u1 = ranks[group].sum() - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n2))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _split_group(x: np.ndarray, dates: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    """One evidence test per group: split into two only on very strong,
    well-sized evidence that isn't better explained by within-season drift.
    Never recurses — real arsenals rarely need more than one correction per
    Savant label, and each split already costs a full evidence test's worth
    of statistical caution."""
    if idx.shape[0] < 2 * MIN_SPLIT_PITCHES:
        return [idx]
    sub = x[idx]
    if not _prefers_two_components(sub):
        return [idx]

    x_fit = _subsample(sub, MAX_FIT_PITCHES)
    two = GaussianMixture(
        n_components=2, covariance_type="full", reg_covar=1e-3, n_init=N_INIT, random_state=RANDOM_STATE
    ).fit(x_fit)
    sub_labels = two.predict(sub)  # assign every pitch in the group, not just the fit subsample
    a, b = idx[sub_labels == 0], idx[sub_labels == 1]
    share = min(a.shape[0], b.shape[0]) / idx.shape[0]
    if share < MIN_SPLIT_SHARE or min(a.shape[0], b.shape[0]) < MIN_SPLIT_PITCHES:
        return [idx]

    date_auc = _rank_auc(dates[idx], sub_labels.astype(bool))
    if max(date_auc, 1.0 - date_auc) > DATE_CONFOUND_AUC:
        return [idx]  # near-perfectly chronological -> drift, not a second pitch
    return [a, b]


def _circular_mean_deg(sin: np.ndarray, cos: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(sin.mean(), cos.mean())) % 360.0)


def _label_clusters(majority: list[str], velo: list[float]) -> list[str]:
    """Disambiguate clusters that share a majority Savant label.

    A single cluster per label keeps the bare label (reads exactly like an
    ordinary pitch_type). Clusters sharing a label are suffixed `-2`, `-3`, ...
    ordered hardest-first, since velocity is the most legible axis for a human
    scanning "two things both called SL" (e.g. a firm cutter-shaped slider vs.
    a slower sweeping one).
    """
    order_within: dict[str, list[int]] = {}
    for i, lab in enumerate(majority):
        order_within.setdefault(lab, []).append(i)

    out = [""] * len(majority)
    for lab, idxs in order_within.items():
        if len(idxs) == 1:
            out[idxs[0]] = lab
            continue
        for rank, i in enumerate(sorted(idxs, key=lambda j: -velo[j]), start=1):
            out[i] = lab if rank == 1 else f"{lab}-{rank}"
    return out


def cluster_pitcher_season(df: pl.DataFrame) -> pl.DataFrame:
    """One pitcher-season's pitches in, one row per re-derived cluster out.

    `df` must already be a single (pitcher, season) slice with
    `CLUSTER_FEATURE_NAMES` plus `pitch_type` present — `build_arsenal_clusters`
    is the entry point that does that grouping; this function is the unit the
    tests exercise directly.
    """
    n = df.height
    x = StandardScaler().fit_transform(df.select(CLUSTER_FEATURE_NAMES).to_numpy())
    savant = df["pitch_type"].to_numpy()
    dates = df["game_date"].to_numpy()

    initial_groups = [np.where(savant == lab)[0] for lab in np.unique(savant)]
    merged = _merge_groups(x, initial_groups)
    final_groups: list[np.ndarray] = []
    for g in merged:
        final_groups.extend(_split_group(x, dates, g))

    velo = df["release_speed"].to_numpy()
    ivb = df["ivb_in"].to_numpy()
    hb = df["hb_arm_in"].to_numpy()
    ext = df["release_extension"].to_numpy()
    spin_sin = df["spin_axis_sin"].to_numpy()
    spin_cos = df["spin_axis_cos"].to_numpy()

    rows = []
    majority: list[str] = []
    velo_by_cluster: list[float] = []
    for idx in final_groups:
        types, counts = np.unique(savant[idx], return_counts=True)
        maj = str(types[np.argmax(counts)])
        majority.append(maj)
        velo_by_cluster.append(float(velo[idx].mean()))
        rows.append(
            {
                "n": int(idx.shape[0]),
                "savant_majority": maj,
                "n_savant_labels": len(types),
                "purity": float(counts.max() / idx.shape[0]),
                "velo_avg": float(velo[idx].mean()),
                "ivb_in": float(ivb[idx].mean()),
                "hb_arm_in": float(hb[idx].mean()),
                "release_extension_avg": float(ext[idx].mean()),
                "spin_axis_arm_deg": _circular_mean_deg(spin_sin[idx], spin_cos[idx]),
            }
        )

    labels = _label_clusters(majority, velo_by_cluster)
    for i, (row, label) in enumerate(zip(rows, labels, strict=True)):
        row["cluster_id"] = i
        row["label"] = label
        row["usage_pct"] = 100.0 * row["n"] / n

    out = pl.DataFrame(rows)

    _savant_types, savant_counts = np.unique(savant, return_counts=True)
    savant_pitch_types = int((savant_counts / n >= SAVANT_MIN_USAGE).sum())
    weighted_purity = float((out["purity"] * out["n"]).sum() / n)
    cluster_k = len(final_groups)

    return out.with_columns(
        pl.lit(cluster_k).alias("cluster_k"),
        pl.lit(savant_pitch_types).alias("savant_pitch_types"),
        pl.lit(cluster_k - savant_pitch_types).alias("arsenal_size_diff"),
        pl.lit(weighted_purity).alias("season_purity"),
    )


def build_arsenal_clusters(
    df: pl.DataFrame, *, min_pitches: int = MIN_PITCHES
) -> pl.DataFrame:
    """Every qualifying (pitcher, season)'s re-derived arsenal, concatenated.

    `df` is a raw multi-season pitch frame (`features/arsenal.py::build_arsenal_frame`
    output) — this does the grouping and the qualifier `cluster_pitcher_season`
    itself doesn't know about.
    """
    counts = df.group_by(["pitcher", "season"]).len().filter(pl.col("len") >= min_pitches)
    qualified = set(zip(counts["pitcher"].to_list(), counts["season"].to_list(), strict=True))
    log.info("arsenal re-classification: %d qualifying pitcher-seasons", len(qualified))

    parts = []
    for (pitcher, season), group in df.group_by(["pitcher", "season"], maintain_order=True):
        if (pitcher, season) not in qualified:
            continue
        result = cluster_pitcher_season(group)
        parts.append(result.with_columns(pl.lit(pitcher).alias("pitcher"), pl.lit(season).alias("season")))

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts).select(
        "pitcher",
        "season",
        "cluster_id",
        "label",
        "n",
        "usage_pct",
        "velo_avg",
        "ivb_in",
        "hb_arm_in",
        "release_extension_avg",
        "spin_axis_arm_deg",
        "savant_majority",
        "purity",
        "n_savant_labels",
        "cluster_k",
        "savant_pitch_types",
        "arsenal_size_diff",
        "season_purity",
    )


def yoy_stability(clusters: pl.DataFrame) -> dict[str, float]:
    """Year-over-year Spearman correlation of `cluster_k` and
    `arsenal_size_diff` for the same pitcher in consecutive seasons — evidence
    this measures a persistent property of the pitcher rather than
    single-season noise, the same standard `PitchQualityModel.stability` and
    the called-strike model's catcher-framing YoY check hold themselves to.

    `clusters` is `build_arsenal_clusters` output (or the mart read back);
    this collapses it to one row per pitcher-season first.

    Measured on 2023-2025: `cluster_k_yoy` 0.56-0.61 (in the 0.5-0.7 range
    published framing metrics report as "real, sticky"), `arsenal_size_diff_yoy`
    0.27-0.30 (weaker, expected — it's a difference of two already-noisy
    counts, not a raw count).
    """
    per_season = clusters.unique(subset=["pitcher", "season"]).select(
        "pitcher", "season", "cluster_k", "arsenal_size_diff"
    )
    nxt = per_season.select(
        pl.col("pitcher"),
        (pl.col("season") - 1).alias("season"),
        pl.col("cluster_k").alias("next_cluster_k"),
        pl.col("arsenal_size_diff").alias("next_arsenal_size_diff"),
    )
    pairs = per_season.join(nxt, on=["pitcher", "season"], how="inner")
    return {
        "n_pairs": float(pairs.height),
        "cluster_k_yoy": _spearman(
            pairs["cluster_k"].to_numpy().astype(float),
            pairs["next_cluster_k"].to_numpy().astype(float),
        ),
        "arsenal_size_diff_yoy": _spearman(
            pairs["arsenal_size_diff"].to_numpy().astype(float),
            pairs["next_arsenal_size_diff"].to_numpy().astype(float),
        ),
    }
