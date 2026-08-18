# Development bookmark

**Paused:** 2026-08-17 · **M1 and M2 complete and visually verified
end-to-end. M3 models #2, #3, #4, #5, and #11 are all done, each with a full
API + UI surface, and every dedicated visualization those models unlocked
(viz #20 catcher framing map, viz #13 umpire zone map, viz #12 UMAP arsenal
map) is built too. Viz #19 (swing path) and viz #8 (spray chart) are also now
built** — see "Viz #8/#19" below; #8's `x_ft`/`y_ft` hit-coordinate transform
required rebuilding the full lake (row counts unchanged, 9,202,082 pitches)
and has NOT yet had a Playwright visual pass (no browser tool available in
that session — API-level smoke checks only). The full 2015-2026 backfill has
landed (9,202,082 pitches, contiguous) and every model has been retrained on
it. Officials data (umpire per game) is fully ingested (11,154 games) and
materialized as `dim_official`.

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
| `apps/api` | `/predict/next-pitch`, `/games/{game_pk}/replay`, `/players/{id}/games`, `/pitches/trajectory`, `/stuff/*`, `/swing/*` (+ `/swing/{id}/pitches` per-swing Arrow, viz #19), `/framing/*`, `/zones/{id}` (roles: batter/pitcher/catcher/umpire), `/arsenal/{id}`, `/arsenal/embedding`, `/arsenal/{id}/similar`, `/spray/*` (`battedballs` Arrow, `contour` JSON, `extent` — viz #8). 31 routes total (`app.openapi()` operation count), JSON + Arrow IPC. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table + re-derived-arsenal panel, at-bat replay strip (viz #9), 3D pitch trajectory (viz #6), pitch quality panel (model #3), swing-plane panel for batters (model #4) plus a swing-path scatter + length histogram (viz #19), spray chart over a real park outline with a smoothed xwOBA contour (viz #8), catcher-framing panel with embedded zone map (viz #20), standalone Umpires tab (viz #13), standalone Arsenal map tab (viz #12 — pan/zoom scatter, archetype hulls, "who does this pitcher resemble?"). Visually verified light + dark EXCEPT viz #8/#19 — see the paused-state note above. |

**Verification status:** 235+ backend Python tests (bbcore/bbetl/bbml/api;
exact count not re-tallied this session — targeted suites for every file
touched all pass, see "Viz #8/#19" below), 48 frontend tests (was 43;
+`histogram.test.ts`), `tsc --noEmit`, `oxlint`, `ruff check`, `bb check`
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
Playwright (light + dark) across every panel/tab EXCEPT viz #8/#19 (this
session had no browser tool available) — see `HISTORY.md` for the specific
pitchers/catchers/umpires checked on the older panels and what each check
found.

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
RF-alley/RF, not survey-grade fence data). **Not yet Playwright-verified** —
that session had no browser tool available; do this before trusting the
chart's visual correctness (park outline orientation, contour rendering,
light/dark).
- Live game-feed mode, `PostgresWarehouse`.
- Model #6 (swing decision, needs #5's P(strike) as RV(take) — now unblocked)
  and model #15 (ABS counterfactual, also now unblocked).
- Viz 7, 10, 14, 15-18 (viz #8 and #19 now done — see above).
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
