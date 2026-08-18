"""Pitcher-season embedding + archetypes (M3 model #11, backing viz #12).

Turns `mart_pitcher_arsenal_clusters` (model #2's re-derived arsenals, one row
per re-derived cluster) into one feature vector per pitcher-season, reduces it
to 2D for the UMAP arsenal map, and clusters that feature space into a small
number of named archetypes ("who does this pitcher resemble?"). This is the
follow-up the model #2 docstring flagged as needing the CLUSTER mart as its
input rather than Savant's raw `pitch_type` groups.

ENCODING: FIXED USAGE-SORTED SLOTS, MEASURED AGAINST A HISTOGRAM ALTERNATIVE
------------------------------------------------------------------------------
Each pitcher-season's re-derived clusters (model #2's output, already sorted
by how much they disagree with Savant) are sorted by `usage_pct` descending
into `N_SLOTS` = 6 fixed slots (covers all but a hundred-odd kitchen-sink
pitcher-seasons in the mart; the rest silently drop their smallest clusters).
Each slot carries 6 features: usage, velocity, IVB, HB, and spin as a
sin/cos pair recovered from `spin_axis_arm_deg` (the same unit-circle trick
`features/arsenal.py` uses to dodge the 359/1 degree discontinuity) — 36 dims
total, absent slots filled with usage 0 and league-mean shape, then
league-standardized (`StandardScaler`) column-wise.

Measured against one alternative, per this project's measure-don't-assume
standard: a usage-weighted 2D histogram over (velocity, movement angle),
8x8=64 dims, which is invariant to how many clusters a pitcher has (a
9-cluster kitchen-sink starter and a simplified 4-cluster summary of the same
repertoire land in similar bins) rather than privileging the biggest few
clusters the way fixed slots do.

**Measured on the full 2020-2026 mart (4,212 pitcher-seasons) and it is NOT
a clean win either way — the two validation axes disagree, and named
spot-checks broke the tie.** Aggregate YoY neighbour rank (t-SNE) actually
favours the histogram: 0.086 vs. slots' 0.220 (lower is better, 0 is
"lands exactly where it was last year"). But the named spot-checks — the
kind of check that caught all four of model #2's real bugs, where aggregate
metrics caught none — go the other way, hard: under the histogram encoding,
Matt Waldron (663362, a knuckleballer, physically the most unusual arm in
the sport) has a nearest-*other*-pitcher distance at only the 20th
percentile of the whole population — a knuckleballer reading as an
unremarkable, replaceable arsenal is disqualifying for a map whose entire
point is "who does this pitcher resemble". Under the slot encoding, Waldron
sits at the 99th percentile (correctly isolated), and Tyler Rogers (643511,
a submariner) also lands at the 99th. The histogram's better YoY number is
explained by what it's insensitive to, not by better signal: it smooths over
exactly the kind of arsenal-shape distinctiveness a submariner or
knuckleballer needs to register as different, which is also what keeps it
stable when a pitcher's *cluster count* wobbles year to year (model #2's own
`arsenal_size_diff_yoy` is a weak 0.27-0.30 for the same reason — see
`models/arsenal.py::yoy_stability`). Slots is the default (`DEFAULT_ENCODING`);
the histogram function and its numbers stay so this isn't re-litigated from
scratch.

`arsenal_size_diff` (model #2's headline, positive = Savant under-split) is
deliberately NOT a clustering input — the map colours points by it, and an
embedding that used it as a feature would be circular (points that are close
in colour would trivially also be close in space).

REDUCER BAKE-OFF
-----------------
`reduce()` is pluggable across PCA / t-SNE / UMAP, all seeded
(`RANDOM_STATE`). Scored on:

  - `sklearn.manifold.trustworthiness` at k=15 — neighbourhood preservation.
  - Same-pitcher year-over-year 2D neighbour rank (`yoy_neighbor_rank`): for
    every pitcher with consecutive qualifying seasons, the percentile rank of
    their own next season among all pitcher-seasons by embedded distance.
    Median near 0 (vs. 0.5 random) means "a pitcher's arsonal moves smoothly
    year to year", the same standard `models.arsenal.yoy_stability` and
    `PitchQualityModel.stability` hold every other mart to.
  - Named spot-checks (`NAMED_SPOT_CHECKS`) — aggregates alone missed all four
    of model #2's real bugs, so this checks specific real pitchers by name:
    Tyler Rogers (621242, submariner) and Matt Waldron (676596, knuckleballer)
    should be far-flung outliers; Emmanuel Clase (661403) should sit among
    hard cutter/slider relievers.

**Measured (slot encoding, full mart): t-SNE wins outright — trustworthiness
0.980 vs. UMAP's 0.968 vs. PCA's 0.776; YoY neighbour rank 0.220 vs. UMAP's
0.230 vs. PCA's 0.216 (PCA's YoY looks competitive but its trustworthiness is
far worse — a linear projection preserves global season-to-season drift
cheaply while badly distorting local neighbourhoods, so it is not actually a
contender).** Named spot-checks confirm t-SNE: excluding same-pitcher
matches, Rogers and Waldron both land at the 99th percentile of
nearest-other-pitcher distance (correctly extreme outliers) and Clase
(661403) at the 94th, among hard-throwing cutter/slider relievers (Louis
Varland, Braydon Fisher) rather than starters. Despite viz #12's name in the
architecture plan ("UMAP arsenal map"), UMAP measured a close but clear
second on every axis here — `DEFAULT_REDUCER` is `"tsne"`, not `"umap"`;
`reduce()` still supports all three, and the plan's colloquial name is kept
for the UI since it names the general technique, not literally the `umap`
package. Re-run `bake_off()` to reproduce these numbers or if the mart
changes meaningfully.

ARCHETYPES ARE FIT ON THE FEATURE SPACE, NEVER THE 2D COORDINATES
-------------------------------------------------------------------
Clustering a 2D projection compounds whatever distortion the reducer already
introduced. `fit_archetypes` runs KMeans on the same standardized feature
matrix the reducer takes as input, with k chosen by silhouette over 4-12
(losing silhouettes recorded in the returned dict, not just the winner).
Archetype names are derived deterministically from each member pitcher-
season's *own* primary/secondary re-derived cluster labels (majority-vote
within the archetype) plus a velocity tier, so the legend reads like
baseball ("FF/SL · 95+") rather than "archetype 3".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bbcore.logging import get_logger

log = get_logger(__name__)

SLOT_SHAPE_FEATURES = ["velo_avg", "ivb_in", "hb_arm_in", "spin_axis_sin", "spin_axis_cos"]
SLOT_FEATURES = ["usage_pct", *SLOT_SHAPE_FEATURES]
N_SLOTS = 6

HIST_BINS = 8

RANDOM_STATE = 0
TRUSTWORTHINESS_K = 15
ARCHETYPE_K_RANGE = range(4, 13)
N_NEIGHBORS = 10

Reducer = Literal["pca", "tsne", "umap"]
Encoding = Literal["slot", "histogram"]

# Measured (see module docstring + `bake_off`): slot encoding + t-SNE won the
# bake-off on the real 2020-2026 mart -- NOT UMAP, despite viz #12's name.
# Update together if re-measured.
DEFAULT_ENCODING: Encoding = "slot"
DEFAULT_REDUCER: Reducer = "tsne"

# Pitchers whose known physical shape gives a hard pass/fail spot-check that
# no aggregate metric would catch — see module docstring.
NAMED_SPOT_CHECKS = {
    643511: "Tyler Rogers — submariner, should be a far outlier",
    663362: "Matt Waldron — knuckleballer, should be a far outlier",
    661403: "Emmanuel Clase — should sit among hard cutter/slider relievers",
}


def _with_spin_trig(df: pl.DataFrame) -> pl.DataFrame:
    rad = pl.col("spin_axis_arm_deg") * (np.pi / 180.0)
    return df.with_columns(rad.sin().alias("spin_axis_sin"), rad.cos().alias("spin_axis_cos"))


def _primary_secondary(df: pl.DataFrame) -> pl.DataFrame:
    """One row per pitcher-season: its top-usage and second-usage re-derived
    cluster labels/velocities, for archetype naming and the mart's
    `primary_label`/`primary_velo` columns."""
    ranked = df.sort(["mlbam_id", "season", "usage_pct"], descending=[False, False, True]).with_columns(
        pl.int_range(pl.len()).over(["mlbam_id", "season"]).alias("_rank")
    )
    primary = ranked.filter(pl.col("_rank") == 0).select(
        "mlbam_id", "season",
        pl.col("label").alias("primary_label"),
        pl.col("velo_avg").alias("primary_velo"),
    )
    secondary = ranked.filter(pl.col("_rank") == 1).select(
        "mlbam_id", "season", pl.col("label").alias("secondary_label")
    )
    return primary.join(secondary, on=["mlbam_id", "season"], how="left")


def slot_encoding(df: pl.DataFrame) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Fixed usage-sorted slots -> (36,) standardized vector per pitcher-season.

    See module docstring for why slots beat the histogram alternative.
    """
    df = _with_spin_trig(df)
    league_mean = {c: float(df[c].mean()) for c in SLOT_SHAPE_FEATURES}

    ids: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    # A single global sort by usage_pct descending also sorts every group's
    # own rows descending (stable sort preserves relative order), so one sort
    # + a maintain_order groupby avoids a per-group sort.
    sorted_df = df.sort("usage_pct", descending=True)
    for (mlbam_id, season), g in sorted_df.group_by(["mlbam_id", "season"], maintain_order=True):
        vec = np.zeros(N_SLOTS * len(SLOT_FEATURES), dtype=float)
        present = min(g.height, N_SLOTS)
        for slot, row in enumerate(g.head(N_SLOTS).iter_rows(named=True)):
            base = slot * len(SLOT_FEATURES)
            vec[base + 0] = row["usage_pct"]
            for k, feat in enumerate(SLOT_SHAPE_FEATURES, start=1):
                vec[base + k] = row[feat]
        for slot in range(present, N_SLOTS):
            base = slot * len(SLOT_FEATURES)
            vec[base + 0] = 0.0
            for k, feat in enumerate(SLOT_SHAPE_FEATURES, start=1):
                vec[base + k] = league_mean[feat]
        ids.append((int(mlbam_id), int(season)))
        rows.append(vec)

    x = StandardScaler().fit_transform(np.vstack(rows))
    return ids, x


def histogram_encoding(df: pl.DataFrame, *, n_bins: int = HIST_BINS) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Usage-weighted 2D histogram over (velocity, movement angle) -> standardized
    vector per pitcher-season. The measured alternative to `slot_encoding` — see
    module docstring for the comparison."""
    velo = df["velo_avg"].to_numpy()
    angle = np.degrees(np.arctan2(df["ivb_in"].to_numpy(), df["hb_arm_in"].to_numpy()))
    velo_edges = np.quantile(velo, np.linspace(0, 1, n_bins + 1))
    velo_edges[0] -= 1e-6
    velo_edges[-1] += 1e-6
    angle_edges = np.linspace(-180.0, 180.0, n_bins + 1)

    tagged = df.with_columns(pl.Series("_velo", velo), pl.Series("_angle", angle))

    ids: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    for (mlbam_id, season), g in tagged.group_by(["mlbam_id", "season"], maintain_order=True):
        hist, _, _ = np.histogram2d(
            g["_velo"].to_numpy(),
            g["_angle"].to_numpy(),
            bins=[velo_edges, angle_edges],
            weights=g["usage_pct"].to_numpy(),
        )
        ids.append((int(mlbam_id), int(season)))
        rows.append(hist.ravel())

    x = StandardScaler().fit_transform(np.vstack(rows))
    return ids, x


def build_encoding(df: pl.DataFrame, encoding: Encoding = DEFAULT_ENCODING) -> tuple[list[tuple[int, int]], np.ndarray]:
    if encoding == "slot":
        return slot_encoding(df)
    if encoding == "histogram":
        return histogram_encoding(df)
    raise ValueError(f"unknown encoding {encoding!r}")


def reduce(x: np.ndarray, reducer: Reducer = DEFAULT_REDUCER) -> np.ndarray:
    """2D projection of a standardized feature matrix, seeded for reproducibility."""
    if reducer == "pca":
        return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x)
    if reducer == "tsne":
        return TSNE(
            n_components=2, random_state=RANDOM_STATE, init="pca", perplexity=min(30, x.shape[0] - 1)
        ).fit_transform(x)
    if reducer == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=1).fit_transform(x)
    raise ValueError(f"unknown reducer {reducer!r}")


def yoy_neighbor_rank(ids: list[tuple[int, int]], embedding: np.ndarray) -> float:
    """Median percentile rank (0 = nearest, ~1 = farthest, 0.5 = random) of a
    pitcher's own next season among every pitcher-season by embedded distance,
    for every (pitcher, season) with a qualifying (pitcher, season+1) also in
    the mart. The domain check for "does 2D distance mean something real"."""
    index = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    if n < 3:
        return float("nan")
    dist = np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=-1)
    percentiles = []
    for i, (pid, season) in enumerate(ids):
        j = index.get((pid, season + 1))
        if j is None:
            continue
        order = np.argsort(dist[i])
        rank = int(np.where(order == j)[0][0])
        percentiles.append((rank - 1) / (n - 2))  # rank 0 is always self (distance 0)
    return float(np.median(percentiles)) if percentiles else float("nan")


def named_spot_check(
    ids: list[tuple[int, int]], embedding: np.ndarray
) -> dict[int, list[tuple[tuple[int, int], float]]]:
    """For each `NAMED_SPOT_CHECKS` pitcher, their nearest 5 neighbours (any
    season they qualify) by embedded distance — for a human to eyeball against
    the known physical read, not scored automatically."""
    index = {}
    for i, (pid, season) in enumerate(ids):
        index.setdefault(pid, []).append((season, i))

    out: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for pid in NAMED_SPOT_CHECKS:
        seasons = index.get(pid)
        if not seasons:
            continue
        _, i = seasons[-1]  # most recent qualifying season
        dist = np.linalg.norm(embedding - embedding[i], axis=1)
        order = np.argsort(dist)
        neighbors = [(ids[j], float(dist[j])) for j in order if j != i][:5]
        out[pid] = neighbors
    return out


def bake_off(df: pl.DataFrame) -> pl.DataFrame:
    """Runs every (encoding, reducer) combination and scores each on
    trustworthiness (k=15) and YoY neighbour rank. Slow-ish (t-SNE/UMAP fit
    six times) but 4,212 rows finishes in well under a minute; meant for
    reproducing the module docstring's measurement, not routine calls.
    """
    rows = []
    for encoding in ("slot", "histogram"):
        ids, x = build_encoding(df, encoding)
        for reducer in ("pca", "tsne", "umap"):
            emb = reduce(x, reducer)
            tw = trustworthiness(x, emb, n_neighbors=TRUSTWORTHINESS_K)
            yoy = yoy_neighbor_rank(ids, emb)
            rows.append({"encoding": encoding, "reducer": reducer, "trustworthiness": tw, "yoy_neighbor_rank": yoy})
            log.info(
                "bake-off %s/%s: trustworthiness=%.4f yoy_neighbor_rank=%.4f", encoding, reducer, tw, yoy
            )
    return pl.DataFrame(rows)


def _velocity_tier(velo: float) -> str:
    if velo >= 99:
        return "99+"
    if velo >= 95:
        return "95-98"
    if velo >= 90:
        return "90-94"
    return "80s-"


def fit_archetypes(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """KMeans on the feature space (never the 2D projection — see module
    docstring). k chosen by silhouette over `ARCHETYPE_K_RANGE`; every
    candidate's silhouette is returned, not just the winner."""
    scores: dict[int, float] = {}
    best_k, best_score, best_labels = None, -1.0, None
    for k in ARCHETYPE_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
        score = float(silhouette_score(x, km.labels_))
        scores[k] = score
        if score > best_score:
            best_k, best_score, best_labels = k, score, km.labels_
    return best_labels, {"k": best_k, "silhouette": best_score, "silhouette_by_k": scores}


def archetype_labels(
    archetype_id: np.ndarray, primary: list[str], secondary: list[str | None], velo: list[float]
) -> dict[int, str]:
    """One name per archetype id: modal primary label / modal secondary label
    among members, plus a velocity tier off the mean primary velocity.

    Two feature-space archetypes can legitimately share the same modal
    primary/secondary/tier (e.g. two different FF/CH clusters both landing in
    the same broad velocity band) — real 2020-2026 data hits this at k=6.
    Disambiguated the same way `models.arsenal._label_clusters` handles a
    repeated Savant label within one pitcher-season: a bare name for the only
    archetype with it, `-2`/`-3`... ordered hardest-first for the rest.
    """
    base: dict[int, str] = {}
    mean_velo: dict[int, float] = {}
    for a in np.unique(archetype_id):
        mask = archetype_id == a
        prim = [primary[i] for i in np.where(mask)[0]]
        sec = [secondary[i] for i in np.where(mask)[0] if secondary[i] is not None]
        mode_prim = max(set(prim), key=prim.count)
        mode_sec = max(set(sec), key=sec.count) if sec else None
        v = float(np.mean([velo[i] for i in np.where(mask)[0]]))
        mean_velo[int(a)] = v
        tier = _velocity_tier(v)
        base[int(a)] = f"{mode_prim}/{mode_sec} · {tier}" if mode_sec else f"{mode_prim} · {tier}"

    by_base: dict[str, list[int]] = {}
    for a, name in base.items():
        by_base.setdefault(name, []).append(a)

    names: dict[int, str] = {}
    for name, archetypes in by_base.items():
        if len(archetypes) == 1:
            names[archetypes[0]] = name
            continue
        for rank, a in enumerate(sorted(archetypes, key=lambda a: -mean_velo[a]), start=1):
            names[a] = name if rank == 1 else f"{name}-{rank}"
    return names


def knn_neighbors(ids: list[tuple[int, int]], x: np.ndarray, *, k: int = N_NEIGHBORS) -> pl.DataFrame:
    """Top-k nearest pitcher-seasons in the pre-embedding FEATURE space (not
    the 2D map — the map is for eyeballing, this is for the actual answer).
    A separate table (not a list column) so the API can `LEFT JOIN dim_player`
    in plain SQL like every other leaderboard route."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, x.shape[0])).fit(x)
    dist, idx = nn.kneighbors(x)
    rows = []
    for i, (mlbam_id, season) in enumerate(ids):
        rank = 0
        for d, j in zip(dist[i], idx[i], strict=True):
            if j == i:
                continue
            rank += 1
            n_id, n_season = ids[j]
            rows.append(
                {
                    "mlbam_id": mlbam_id,
                    "season": season,
                    "rank": rank,
                    "neighbor_id": n_id,
                    "neighbor_season": n_season,
                    "distance": float(d),
                }
            )
            if rank >= k:
                break
    return pl.DataFrame(rows)


def build_embedding(
    df: pl.DataFrame,
    *,
    encoding: Encoding = DEFAULT_ENCODING,
    reducer: Reducer = DEFAULT_REDUCER,
    n_neighbors: int = N_NEIGHBORS,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """The full pipeline: `df` is `mart_pitcher_arsenal_clusters` (one row per
    re-derived cluster). Returns (embedding rows, neighbor rows, validation
    numbers) — validation numbers are for the CLI to print, not written to
    either mart.
    """
    ids, x = build_encoding(df, encoding)
    embedding = reduce(x, reducer)
    tw = float(trustworthiness(x, embedding, n_neighbors=min(TRUSTWORTHINESS_K, x.shape[0] - 1)))
    yoy = yoy_neighbor_rank(ids, embedding)

    labels, arch_info = fit_archetypes(x)

    ps = _primary_secondary(df)
    ps_index = {(r["mlbam_id"], r["season"]): r for r in ps.iter_rows(named=True)}
    primary = [ps_index[i]["primary_label"] for i in ids]
    secondary = [ps_index[i]["secondary_label"] for i in ids]
    primary_velo = [ps_index[i]["primary_velo"] for i in ids]
    names = archetype_labels(labels, primary, secondary, primary_velo)

    meta = (
        df.group_by(["mlbam_id", "season"])
        .agg(
            pl.col("cluster_k").first(),
            pl.col("savant_pitch_types").first(),
            pl.col("arsenal_size_diff").first(),
            pl.col("season_purity").first(),
            pl.col("n").sum().alias("n_pitches"),
        )
    )
    meta_index = {(r["mlbam_id"], r["season"]): r for r in meta.iter_rows(named=True)}

    emb_rows = []
    for i, (mlbam_id, season) in enumerate(ids):
        m = meta_index[(mlbam_id, season)]
        emb_rows.append(
            {
                "mlbam_id": mlbam_id,
                "season": season,
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]),
                "archetype_id": int(labels[i]),
                "archetype_label": names[int(labels[i])],
                "cluster_k": m["cluster_k"],
                "savant_pitch_types": m["savant_pitch_types"],
                "arsenal_size_diff": m["arsenal_size_diff"],
                "season_purity": m["season_purity"],
                "n_pitches": m["n_pitches"],
                "primary_label": primary[i],
                "primary_velo": primary_velo[i],
                "reducer": reducer,
            }
        )
    embedding_df = pl.DataFrame(emb_rows)
    neighbors_df = knn_neighbors(ids, x, k=n_neighbors)

    validation = {
        "encoding": encoding,
        "reducer": reducer,
        "trustworthiness": tw,
        "yoy_neighbor_rank": yoy,
        "n_pitcher_seasons": len(ids),
        **arch_info,
        "named_spot_check": named_spot_check(ids, embedding),
    }
    return embedding_df, neighbors_df, validation
