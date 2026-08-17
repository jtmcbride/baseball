"""Arsenal re-classification features: the pitch shape, without Savant's label.

WHY THIS EXISTS
----------------
`pitch_type` everywhere else in this codebase (`mart_pitcher_arsenal.sql`,
Location+'s feature set, the frontend arsenal table) is Savant's own label, and
that label comes from an automated classifier with known failure modes at the
boundary between similar shapes: a slider and a sweeper differ mainly by
degree (horizontal break, spin axis), not by kind, and a pitcher who throws
one blended shape can get split across both labels, or two pitchers'
genuinely different sliders can get merged under one. Model #2 re-derives
arsenals directly from the physical measurements, per pitcher-season, and
compares the result back against Savant's own labels rather than replacing
them everywhere — see `models/arsenal.py` for the actual clustering.

FEATURES ARE BORROWED FROM STUFF+, NOT REDERIVED
--------------------------------------------------
`add_pitch_quality_features` (`features/stuff.py`) already does the exact
handedness normalization pitch-shape clustering needs: `hb_arm_in` is
arm-side-positive out of the transform layer, `release_pos_x_arm` mirrors
release point the same way, and `spin_axis` is mirrored for LHP and put on
the unit circle so 359 degrees and 1 degree are neighbours rather than
opposites. Re-deriving any of that here would risk a second, silently
different answer to "which way is arm-side" from the one Stuff+ already
uses and has validated. `velo_diff_fb`/`ivb_diff_fb`/`hb_diff_fb` are
deliberately NOT reused here, though — those are relative to Savant's own
notion of "the primary fastball" (grouped by `pitch_type`), which is exactly
the labeling this model is trying to check rather than assume.

`arm_angle` is excluded from the clustering features even though Stuff+
carries it: it's 2025-only in most of the lake (see that module's docstring),
so including it would silently limit re-classification to one season while
looking like a full-history feature.

USABLE RANGE IS 2020+ FOR RELIABLE OUTPUT, NOT THE FULL 2015-2026 BACKFILL
-------------------------------------------------------------------------------
`spin_axis` has 0% coverage in 2015 (`bb check --coverage`) — Statcast didn't
publish it that first season. `CLUSTER_FEATURES` needs it (spin axis is part
of what makes a pitch shape a pitch shape, not an optional extra), so every
2015 pitch fails the null-filter below and the season contributes zero rows.

That alone would only push the floor to 2016 — the real floor is 2020, and it
took a full-history mart build to find out why. A first pass over 2016-2026
produced its ten worst disagreements with Savant *entirely* from 2016-2019 —
one pitcher with 5 Savant pitch types reported as 13 clusters, all at 100%
purity. Checked season-by-season, `mean |arsenal_size_diff|` is 2.8-3.2 for
2016-2019 (only 14-23% of pitcher-seasons within +-1 of Savant's count) and
drops in a clean step, not a gradual trend, to 0.8-1.2 from 2020 on (65-83%
within +-1) — exactly the boundary where Statcast unified every park onto one
Hawk-Eye tracking system. Spot-checking one 2019 pitcher's doubled changeup
confirmed it wasn't a bug already fixed elsewhere (date AUC 0.34, no
chronological pattern; the two "changeups" differed in velocity, movement,
*and* spin axis simultaneously, spread over 20+ different games each) — the
signature of genuinely less consistent pre-Hawk-Eye measurement, not a
confound this model's logic can correct for. `bbml.marts.build_arsenal_cluster_mart`
defaults to 2020+ for exactly this reason; the loader and clustering
functions here stay season-agnostic so an explicit request for 2016-2019 is
still possible; unlike every other mart in this codebase,
`mart_pitcher_arsenal_clusters` does not cover the full backfill by
default — this is expected, not a gap to fix.

RELEASE POINT WAS ALSO EXCLUDED — MEASURED, NOT ASSUMED
------------------------------------------------------------
The plan for this model listed "release" among the clustering inputs, and the
first version included `release_extension`, `release_pos_x_arm`, and
`release_pos_z` alongside the shape features. On real 2025 data that produced
a pitcher whose fastball, sinker, and curveball *each* split into two
near-identical clusters (e.g. one fastball pair at 92.02mph/17.86in IVB vs
92.12mph/17.86in IVB — not a physically distinguishable pitch). Comparing the
two clusters feature-by-feature showed velocity, movement, and spin axis were
all essentially equal; only `release_pos_x_arm` differed meaningfully (1.06ft
vs 0.45ft, arm-side). This pitcher has a bimodal release point — a real
mechanical property of his delivery, not a different pitch — and including it
as a clustering input let pure arm-slot variance masquerade as a whole extra
pitch, on every pitch type at once (a same-pitcher-wide pattern is itself a
tell: a genuinely different pitch shows up in one pitch type, not all of
them). Dropping the three release-point columns removed the spurious split on
every affected pitch type while leaving a separately-found, plausible
two-cutter split on the same pitcher intact (its two clusters differed by
3.6in of IVB and 1.9in of horizontal break — an actual shape difference, not
release noise) — evidence the fix removed the right thing rather than just
suppressing splits generally. `velo`, `ivb_in`, `hb_arm_in`, and spin axis are
what a human means by "which pitch is this"; where a pitcher happened to be
standing when he let go of it is not.
"""

from __future__ import annotations

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbml.features.schema import Feature
from bbml.features.stuff import add_pitch_quality_features, load_pitch_frame

log = get_logger(__name__)

# The pitch's physical shape, handedness-normalized — velocity, movement, and
# spin axis, and nothing else. Deliberately excludes `arm_angle` (2025-only
# coverage), the fastball-differential columns (those presuppose the Savant
# `pitch_type` grouping this model re-derives), and release point (real but
# unrelated bimodality in where a pitcher stands to throw — see "RELEASE
# POINT WAS ALSO EXCLUDED" above for the measured example).
CLUSTER_FEATURES: list[Feature] = [
    Feature("release_speed", "numeric", "Velocity out of hand, mph."),
    Feature("ivb_in", "numeric", "Induced vertical break, inches."),
    Feature("hb_arm_in", "numeric", "Horizontal break, arm-side positive, inches."),
    Feature("spin_axis_sin", "numeric", "Spin axis, mirrored for LHP, sine component."),
    Feature("spin_axis_cos", "numeric", "Spin axis, mirrored for LHP, cosine component."),
]

CLUSTER_FEATURE_NAMES: list[str] = [f.name for f in CLUSTER_FEATURES]


def build_arsenal_frame(
    *,
    seasons: list[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Tracked pitches with clustering features attached, one row per pitch.

    Carries `pitcher`, `season`, `pitch_type` (Savant's own label, kept only
    so the model can grade itself against it — never a clustering input) and
    `pitch_name` (for readable fallback labels) alongside the physical shape.
    """
    s = settings or get_settings()
    df = load_pitch_frame(seasons=seasons, settings=s)
    out = add_pitch_quality_features(df)
    # `pitch_type` null crashed a full-history build (`np.unique` can't sort
    # None against strings) — `is_tracked_pitch & is_competitive` does not
    # guarantee a pitch got classified at all. `mart_pitcher_arsenal.sql`
    # already filters this same case; missing it here is what a from-scratch
    # Python rebuild of a SQL filter looks like when it isn't copied exactly.
    # A handful of pitches per season are separately missing one physical
    # measurement (tracking gaps, not a labeling issue) — GMM has no native
    # way to handle a NaN, so those are dropped too rather than imputed.
    # Together: under 0.3% of pitches on a real recent season, concentrated
    # in `spin_axis`; a larger, one-time share on the full 2015-2026 history
    # from early seasons with less complete classification.
    before = out.height
    out = out.filter(
        pl.col("pitch_type").is_not_null()
        & pl.all_horizontal([pl.col(c).is_not_null() for c in CLUSTER_FEATURE_NAMES])
    )
    dropped = before - out.height
    log.info(
        "loaded %d pitches for arsenal re-classification (%d dropped for missing measurements)",
        out.height,
        dropped,
    )
    return out
