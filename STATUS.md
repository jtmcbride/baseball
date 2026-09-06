# Development bookmark

**Paused:** 2026-09-06 · **M1 and M2 complete and visually verified
end-to-end. M3 models #2, #3, #4, #5, and #11 are all done, each with a full
API + UI surface, and every dedicated visualization those models unlocked
(viz #20 catcher framing map, viz #13 umpire zone map, viz #12 UMAP arsenal
map) is built too. Viz #8 (spray chart) and #19 (swing path) are built AND
now Playwright-verified against real data** (2026-09-06 — the check STATUS
had been carrying as outstanding since 2026-08-17; it found three real bugs,
see "Viz #8/#19 verification" below). **A regression on `main` that silenced
the whole model #2/#11 API surface has been fixed** — see "Arsenal router"
below. The full 2015-2026 backfill has landed (9,202,082 pitches, contiguous)
and every model has been retrained on it. Officials data (umpire per game) is
fully ingested (11,154 games) and materialized as `dim_official`.

Read this first in a new session, then `README.md` for how the thing works,
`HISTORY.md` for the full dated write-up of how each piece got built (bugs
found, numbers measured, design dead-ends) — this file stays lean and
current, HISTORY.md is the archive it points back to — and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M3 backlog.

---

## Where we are

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite, `transforms/officials.py` (`dim_official`). Complete. |
| `packages/bbml` | Feature builder (batch+live, parity-tested), datasets/splits, `UsageRateBaseline`, `NextPitchModel` (pitch type), `LocationModel` (26-class grid), `PersonalizedBlend`, arsenal re-classification (M3 model #2 — pairwise GMM merge/split tests against Savant's `pitch_type`), arsenal embedding + archetypes (M3 model #11, backs viz #12 — `models/arsenal_embed.py`), `RunValue` + `PitchQualityModel` (Stuff+/Location+/Pitching+, M3 model #3), `SwingPathModel` (whiff + contact heads, M3 model #4), `CalledStrikeModel` (binary, `framing_runs` + `umpire_zone_rate`, M3 model #5), `registry.py` (versioned artifacts + optional MLflow), `marts.py` (every mart below plus the catcher/umpire spatial grids feeding `mart_zone_profile`), `bb-ml` CLI. Depends on `bbetl`. |
| `apps/api` | `/predict/next-pitch`, `/games/{game_pk}/replay`, `/players/{id}/games`, `/pitches/trajectory`, `/stuff/*`, `/swing/*` (+ `/swing/{id}/pitches` per-swing Arrow, viz #19), `/framing/*`, `/zones/{id}` (roles: batter/pitcher/catcher/umpire), `/arsenal/{id}`, `/arsenal/embedding`, `/arsenal/{id}/similar`, `/spray/*` (`battedballs` Arrow, `contour` JSON, `extent` — viz #8), `/pitches/arsenal-trajectories` (one real exemplar flight per pitch type) and `/atbats` + `/atbats/{game_pk}/{at_bat_number}` (whole plate appearance, physics per pitch — pitch comparison + 3D at-bat). 34 routes total (`app.openapi()` path count; the previously recorded 31 was measured while the arsenal router was unmounted — see "Arsenal router"), JSON + Arrow IPC. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table + re-derived-arsenal panel, at-bat replay strip (viz #9), 3D pitch trajectory (viz #6), pitch quality panel (model #3), swing-plane panel for batters (model #4) plus a swing-path scatter + length histogram (viz #19), spray chart over a real park outline with a smoothed xwOBA contour (viz #8), catcher-framing panel with embedded zone map (viz #20), standalone Umpires tab (viz #13), standalone Arsenal map tab (viz #12 — pan/zoom scatter, archetype hulls, "who does this pitcher resemble?"), pitch comparison (multi-trajectory 3D: one pitcher's arsenal overlaid, or two pitchers side by side, with a commit-point tunnel readout) and a 3D at-bat panel (every pitch of one plate appearance over the batter's own zone heat map). Visually verified light + dark, viz #8/#19 included as of 2026-09-06. |

**Verification status:** 235+ backend Python tests (bbcore/bbetl/bbml/api).
The full backend suite was NOT run to completion on 2026-09-06 — it was
stopped at 27% (135 tests, no failures in that prefix); the targeted
`TestMeta`/`TestArsenal*` selection passes apart from the deliberate ABS
failure below. **Re-run `uv run pytest` before trusting the backend state.**
48 frontend tests (was 43;
+`histogram.test.ts`; now 64, +4 in `viewport.test.ts` for `axisStretch`),
`tsc --noEmit`, `oxlint`, `ruff check`, `bb check`
(data quality — all error-level checks pass after the lake rebuild), a
frontend production build (`npm run build`). `bb-ml status` unchanged by this
session — `mart_batter_spray` is a direct-from-`fact_pitch` mart with no
trained model behind it, same shape as the arsenal-cluster/embedding marts,
so it isn't a ninth `bb-ml status` entry either. All eight registered models
still trained on the full 9,202,082-pitch 2015-2026 lake (unchanged row
counts after the rebuild), saved to
`data/models/{next_pitch,location,stuff_plus,location_plus,pitching_plus,swing_whiff,swing_contact,called_strike}/`
— arsenal re-classification and arsenal embedding have no registered
artifact (small per-pitcher-season / whole-mart fits, not `bbml.registry`
artifacts) so they aren't a ninth/tenth entry here. UI visually verified with
Playwright (light + dark) across every panel/tab — viz #8/#19 included as of
2026-09-06 (Judge, Ohtani, Arraez, against the real API, not mocks) — see
`HISTORY.md` for the specific pitchers/catchers/umpires checked on each panel
and what each check found.

**One backend test fails on `main`, deliberately left failing:**
`TestArsenalTrajectories::test_flight_lands_on_the_pitch_own_plate_crossing`
reconstructs an exemplar's flight to `17/12` ft and gets a 0.098 ft vertical
residual when the sampled exemplar comes from an ABS game. This is the ABS
plate-reference canary doing its job — it fires exactly when a 2026 (or 2025
spring/All-Star) pitch is drawn — and it stays red until the plane question
is decided. Do not "fix" it by widening the tolerance; that deletes the only
automated detector for the shift. See "Open decisions" below.

---

## Current local data

- **9,202,082 pitches**, seasons 2015-2026, contiguous — no season gaps.
- Marts: `mart_pitcher_stuff` 40,539 rows (pitcher x season x pitch_type + an
  `ALL` rollup); `mart_batter_swing` 1,611 rows (batter x season);
  `mart_catcher_framing` 1,048 rows (catcher x season); `mart_umpire_zone`
  230 rows (umpire x season); `mart_zone_profile` holds 1,048 catcher-season
  and 362 umpire-season grids alongside the batter/pitcher grids;
  `mart_pitcher_arsenal_clusters` 19,855 rows / 4,212 pitcher-seasons,
  **2020-2026 only** (see "Decisions already made" below — reliable range,
  not the full backfill); `mart_arsenal_embedding` 4,212 rows (one per
  pitcher-season) and `mart_arsenal_neighbors` 42,120 rows (top-10 nearest
  per pitcher-season). `mart_pitcher_arsenal` / the batter/pitcher share of
  `mart_zone_profile` not re-counted recently — re-run `bb check --coverage`
  before trusting old figures. `mart_batter_spray` (new, viz #8): 4,518 rows
  (batter x season, min 100 batted balls), full 2015-2026 range.
- Officials: **11,154 games** with a home-plate umpire, materialized as
  `dim_official`, full coverage.

---

## Resume paths

**M1, M2 done. M3 models #2/#3/#4/#5/#11 done, each with full API + UI +
their unlocked visualizations (viz #12/#13/#20).** Plans for #2/#11 and #5
are archived in the assistant's project memory
(`baseball-model2-arsenal-plan`, `baseball-model5-called-strike-plan`) if
design-choice detail is needed — both are fully executed, not just scoped.

**Viz #8/#19 (2026-08-17):** both built per
`~/.claude/plans/plan-the-implementation-of-recursive-hinton.md`. #19 (swing
path) needed only a new per-swing Arrow route
(`GET /swing/{id}/pitches`, calling `load_swing_frame()` directly — no new
mart) plus a scatter + length histogram; all its underlying data already
existed. #8 (spray chart) was new plumbing end to end: `x_ft`/`y_ft`/
`spray_angle_deg`/`hit_distance_derived_ft` derived from `hc_x`/`hc_y` in
`bbetl.transforms.statcast.enrich` (constants MEASURED against real
`hit_distance_sc`, not the community-published defaults — see that module's
comment; origin confirmed within a foot, scale corrected 2.495→2.339, MAE
~28ft/r=0.89 even at best fit because `hc_x`/`hc_y` is a charted fielding
location, not a trajectory endpoint), a shared `kernel_regress_2d` smoothing
core extracted from `zones.py` into `bbetl.transforms.smoothing` (both
modules now call it, regression-tested to reproduce `zones.py`'s pre-refactor
output exactly), the new `mart_batter_spray` mart, a `/spray/*` router, and
30 MLB park wall polygons (`apps/web/src/data/parks.ts`, Catmull-Rom-smoothed
through 5 publicly documented distance markers per park — LF/LF-alley/CF/
RF-alley/RF, not survey-grade fence data).

**Arsenal router (fixed 2026-09-06):** `apps/api/src/bbapi/main.py` never
called `app.include_router(arsenal.router)`, so on every commit through
`46b3837` the whole model #2/#11 surface 404'd — `/arsenal/embedding`,
`/arsenal/{id}`, `/arsenal/{id}/similar`, and with them the Arsenal map tab
(viz #12), the re-derived-arsenal panel and "who does this pitcher resemble?".
The suite stayed green because `_needs()` in `test_api.py` skips off
`health["tables"]`, and the same handler in the same file also omitted the
three arsenal marts from its hand-written table list — so `.get()` returned
None and every arsenal test, including the one written as a route-matching
regression test, skipped silently. Three fixes, not one:
`app.include_router(arsenal.router)`; `/health`'s table list now DERIVED from
`LAKE_TABLES | SQL_MARTS` (`main.KNOWN_TABLES`) instead of hand-maintained;
`_needs()` now RAISES on a table `/health` has never heard of and skips only
on one it reports as unbuilt. Plus `TestMeta::test_every_router_module_is_mounted`,
which walks `bbapi.routers` and asserts every route each module defines is in
`app.openapi()["paths"]` — verified to fail with the include commented out.
Two more latent holes closed on the way: `mart_arsenal_embedding` and
`mart_arsenal_neighbors` were missing from `bbetl.warehouse.LAKE_TABLES` (they
self-register at write time, so a plain `bb build register` would not have
recovered them — the documented gotcha, live), and `apps/api` now declares its
`bbetl` dependency explicitly rather than leaning on `bbml`'s.

**Viz #8/#19 verification (2026-09-06):** the outstanding Playwright pass, run
against the real local API with real data (Judge, Ohtani, Arraez; light +
dark). Three real bugs, all of the kind unit tests structurally cannot see —
the components rendered, the data was right, the pictures were wrong:
- **Viz #19's x axis collapsed to 10% of the plot.** `SwingPathScatter` reused
  `fitViewport`, whose single `scale` is deliberate — feet-by-feet on the
  spray chart, where an anisotropic park is a wrong park — but wrong for a
  scatter of two unrelated angles. Descent angle spans ~12.5 deg against
  attack angle's ~91.4, so x got 49 of 480 pixels and every batter drew as the
  same vertical ribbon; the fastball/breaking-ball separation the viz exists
  to show was invisible. Fixed with `axisStretch()` in `lib/viewport.ts`,
  which pre-stretches x into y's units before fitting, leaving the shared
  pan/zoom math and the raw `vaa_deg` on the row for the readout. 4 tests.
- **Viz #8 opened on an arbitrary park.** `defaultTeam` was
  `battedBallRows[0]?.home_team` — row order, not the batter. A career query
  for a Yankee opened on Citizens Bank Park. Now the modal `home_team`.
- **Viz #8 had no foul lines and no home plate.** `parkPolygon` returns the
  wall arc only, and `wallPath` does not close it, so the chart was a bare
  dome floating over the points with nothing marking the origin everything is
  measured from. Added dashed foul lines home-plate-to-each-pole plus a plate
  marker.

**What the pass confirmed correct** (worth not re-deriving): viz #8's
coordinate transform and park geometry are right, measured rather than
eyeballed. Against Judge's career 3,286 batted balls at Yankee Stadium, only
**0.42% of non-home-runs (12 balls) plot beyond the wall**, which is the check
that a scale or origin error would fail loudly; 5.17% fall outside the foul
lines, which is foul-territory batted-ball events, as expected. 61.5% of home
runs plot INSIDE the wall — that is the documented `hc_x`/`hc_y` caveat (a
charted fielding location, ~28ft MAE, not a landing point), not a rendering
bug. The `parkPolygon` spline reproduces its five measured markers exactly and
is monotone in angle. The dark-mode diverging ramp inverts lightness
(extremes light, midpoint dark) — that is the intended dark-theme design in
`theme.css`, not a reversed legend; it looks wrong beside the light shot and
is not.
- Live game-feed mode, `PostgresWarehouse`.
- Model #6 (swing decision, needs #5's P(strike) as RV(take) — now unblocked)
  and model #15 (ABS counterfactual, also now unblocked).
- Viz 7, 10, 14, 15-18 (viz #8 and #19 done and now verified — see above).
- Retrosheet backfill.
- A location arsenal-style prior (where a pitcher tends to miss) as a
  next-pitch/location feature.
- `save_model` doesn't persist a `metrics.json` beside each artifact (MLflow
  isn't installed here, so training metrics evaporate after the console
  print) — fix before trying to answer the Stuff+ predictive-validity
  question on a proper contiguous split.
- `ArsenalTable.tsx`'s `key={r.pitch_type}` collides across seasons when no
  season filter is set (React duplicate-key warning, not a crash) —
  surfaced during viz #12 Playwright verification, pre-existing, not yet
  fixed. See `HISTORY.md`'s "Arsenal embedding, API, UI, viz #12" section.

---

## Open decisions — blocking, decide before the work they gate

- **The ABS plate-reference plane.** In ABS games Savant reports
  `plate_x`/`plate_z` at the plate MIDPOINT (y = 8.5in), not the front edge
  (y = 17in): 2026 in every game type, 2025 only `game_type` S and A. The
  shift is pitch-type dependent (CU 1.44in down to FF 0.70in — it is
  `vz_plate x dt`, so steep pitches move furthest), so it cannot be absorbed
  as a constant, and **4.07% of 2026 competitive pitches (~28k) flip
  `is_in_zone` depending on which plane you pick.** Normalizing at ingest is
  agreed; the plane is not chosen. Front edge rewrites 829,381 rows (9% of the
  lake) and keeps every existing mart and model meaning what it meant;
  midpoint rewrites 8.37M rows (91%), forces a full retrain, and cannot move
  2015-2016 at all (those seasons fit NO plane — 0.20-0.35ft residual — which
  is already why arsenal clusters start at 2020). Recommended: front edge,
  preserving the raw Savant values, plus a per-row `plate_ref_y` column so
  tests assert against the row's own plane instead of a hardcoded `17/12`.
  The counterweight is future intent: if ABS-challenge modelling is on the
  roadmap, midpoint is the honest target and is cheapest to migrate now.
  Gates: model #15 (ABS counterfactual), any trustworthy 2026 zone/framing
  number, and the failing canary test noted under "Verification status".
  Full measured detail in the assistant's project memory,
  `baseball-abs-plate-reference-shift`.

## Decisions already made — don't relitigate

- **Colour encodes pitch *family* (3 hues), shape encodes pitch type.** The
  all-pairs CVD gate: the validated palette clears it with three slots, nine
  pitch types cannot take nine hues. Centroid labels + table views supply the
  required contrast relief. Applies to every scatter, including the arsenal
  map (archetypes get hull outlines, not their own hue).
- **Arrow IPC for pitch-level routes, JSON for everything else.** Measured
  6.8x smaller on real data.
- **`season` is written into the Parquet files, not just the directory
  name,** and `hive_partitioning=false` everywhere. DuckDB 1.5.5 throws an
  InternalException when a query projects *only* a synthesized partition
  column. Do not "simplify" this back to hive synthesis.
- **`bb ingest dims` must run AFTER `bb build pitches`** — `dim_player` is
  populated from the ids present in `fact_pitch`. The Makefile `pipeline`
  target encodes the right order.
- **Cutters are ranked below four-seams/sinkers when picking the baseline
  fastball** for velo/movement deltas — a cutter is its own pitch class.
- **`pitcher` is deliberately not a next-pitch feature.** Personalization is
  via expanding-window per-pitch-type priors, not a pitcher ID or a
  per-pitcher model — measured, not assumed (see `next_pitch.py`).
- **The arsenal mask defaults off** — hard-zeroing pitches outside a
  pitcher's learned arsenal made log-loss and ECE both worse.
- **The pitch-quality feature sets deliberately break the next-pitch leakage
  rule.** Stuff+/Location+ are *grading a pitch already thrown*, so columns
  describing the pitch are the entire input, not leakage.
  `auto_split(..., check_features=False)` is the correct opt-out.
- **MLflow uses a sqlite backend**, not the plain file store. Tracking URI is
  `sqlite:///data/models/mlruns/mlflow.db`.
- **`mart_pitcher_arsenal_clusters` and its embedding default to 2020+, not
  the full 2015-2026 backfill** — pre-Hawk-Eye tracking (2016-2019) measures
  the clustering features inconsistently enough to produce spurious splits.
  See `HISTORY.md`'s "Arsenal re-classification" section for the measured
  numbers behind this.
- **The UMAP arsenal map (viz #12) does not actually default to UMAP.**
  t-SNE measurably won the reducer bake-off (trustworthiness, YoY neighbor
  rank, and the named-pitcher spot-check all favor it). The tab keeps its
  plan-given name since it names the general technique;
  `DEFAULT_REDUCER` in `arsenal_embed.py` is `"tsne"`. Don't "fix" this back
  to UMAP without re-running `bb-ml arsenal-bakeoff` first.

## Gotchas that cost time to rediscover

- **Savant truncates silently at 25,000 rows** — HTTP 200, no marker, data
  just stops mid-day. Guarded, but never widen the date partition without
  re-checking.
- **Savant revises published data** after the fact. `bb ingest refresh`
  re-pulls a trailing window; append-only ingest goes stale invisibly.
- **Statcast's `umpire` column is empty in every season.** Umpires come from
  the Stats API boxscore (`bb ingest officials`), landed as raw JSON only —
  `bb build officials` is the separate step that turns that into the
  queryable `dim_official` lake table.
- **A new lake table isn't queryable until it's in
  `bbetl.warehouse.LAKE_TABLES`.** Writing the Parquet and registering it
  with the warehouse are two different steps for every `dim_*`/`mart_*`
  table. If a new table 503s despite the build command succeeding, check
  `LAKE_TABLES` before checking anything else.
- **`dim_official` only covers 2023+.** Every take from an earlier game has a
  null umpire id. Any `group_by` on a column that can be null for a
  structural reason (not just missing data) MUST filter the null out
  explicitly — polars groups null as its own bucket rather than dropping it.
- **`bbml` depends on `bbetl`** (`pyproject.toml`) — it reuses
  `bbetl.transforms.zones`'s grid-smoothing machinery. Declare cross-package
  dependencies explicitly; the workspace root installing everything into one
  shared venv will mask a missing declaration until it doesn't.
- **DuckDB persists a view's resolved schema.** Rebuilding the lake with a
  changed column set leaves stale views that fail confusingly. `build
  pitches` re-registers automatically; keep it that way.
- **Bat tracking / swing path is 2023H2+, not 2025+.** Savant backfilled it
  (0% before July 2023, ~95% after). Nullable before 2023H2 and still
  nullable per-row after. `bb check --coverage` reports per-season
  availability — read it rather than assuming.
- **A single-feature counterfactual that freezes correlated features at
  their actual values can flip sign on real data**, not just add noise. If a
  counterfactual metric perturbs one feature while holding others fixed at
  an individual's own values, check whether those held-fixed features are
  themselves correlated with the perturbed one before trusting the sign.
- **`_prior_sum` needs `fill_null(0)` on the counted expression**, not just
  on the result — without it, every prior comes out null in live inference.
  Caught by the parity test; don't remove the `fill_null`.
- **An all-null-for-the-day column infers as `String`, not `Float64`.** Any
  measurement column with zero non-null values in one day's raw CSV makes
  polars pick `String` for that file; `diagonal_relaxed` concat then upcasts
  the whole column. Every physics/measurement column must be pinned in
  `SCHEMA_OVERRIDES`.
- **`open_warehouse` takes an exclusive DuckDB lock** — `bb build`/`bb
  ingest` will fail with `IOException: Could not set lock` if the API server
  (or anything else holding a `DuckDBWarehouse`) is still running. Stop it
  first.

## Quick sanity check after `git pull` / fresh session

```bash
uv sync --python 3.13 && cd apps/web && npm install && cd ../..
uv run pytest && uv run bb check
uv run bb status          # ingest manifest
uv run bb-ml status       # registered model versions
```
