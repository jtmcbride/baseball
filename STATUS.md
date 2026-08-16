# Development bookmark

**Paused:** 2026-08-16 · **M1 and M2 complete and visually verified end-to-end.
Two M3 items pulled forward and finished: the 3D pitch trajectory (viz #6) and
Stuff+ / Location+ / Pitching+ (model #3, the flagship analytical feature).** A
partial backfill (2015-2017 + 2025) is what every model is currently trained on;
the full 2015-present run was still in flight when this was written and
everything needs retraining once it lands.

Read this first in a new session, then `README.md` for how the thing works and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M3 backlog.

---

## Where we are

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite. Complete. |
| `packages/bbml` | Feature builder (batch+live, parity-tested), datasets/splits, `UsageRateBaseline`, `NextPitchModel` (pitch type), `LocationModel` (26-class grid), `PersonalizedBlend`, **`RunValue` + `PitchQualityModel` (Stuff+/Location+/Pitching+, M3 model #3 — see below)**, `registry.py` (versioned artifacts + optional MLflow), `marts.py`, `bb-ml` CLI. |
| `apps/api` | `/predict/next-pitch` (what-if), `/games/{game_pk}/replay`, `/players/{id}/games`, `/pitches/trajectory`, `/stuff/{id}` + `/stuff` leaderboard added. 14 routes total, JSON + Arrow IPC. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table, at-bat replay strip (viz #9), **3D pitch trajectory** (viz #6), **pitch quality panel** (model #3). Visually verified. |

**Verification status:** 129 backend Python tests (bbcore/bbetl/bbml/api) + 18
frontend tests, `tsc --noEmit`, `oxlint`, `ruff check`, `bb check` (data quality
— all error-level checks pass), `bb-ml status` all pass/registered. Models
retrained on the expanded 2.4M-pitch lake (see below) and saved to
`data/models/{next_pitch,location,stuff_plus,location_plus,pitching_plus}/`.
**The rendered UI has been visually verified** (Playwright/Chromium screenshots,
light + dark, `Tarik Skubal`) — see below.

---

## Visual verification (2026-08-15) — found and fixed a real bug

Installed Playwright (already a devDependency with Chromium pre-cached, so no
new install) and screenshotted the player page in light and dark mode. Found: **Movement,
Release point, and Velocity-by-inning were all permanently blank** (stuck on
the loading skeleton) on every player page, while Location profile, the replay
strip, and the arsenal table worked fine.

**Root cause:** `apps/api/src/bbapi/arrow.py` wrote Arrow IPC batches with
`compression="zstd"`. The JS `apache-arrow` package (`apps/web`) has no codec
registered for zstd and throws `Record batch is compressed but codec not
found` the moment it tries to decode — silently, since React Query has no
`onError` handler here, so the three charts fed by `/pitches` just sat on
their skeleton forever with no console error. `/zones`, `/games/*/replay`, and
`/players/*/arsenal` are JSON, not Arrow, which is why those three kept
working and masked the bug.

**Fix applied:** dropped the `compression="zstd"` option in `arrow_response()`
(`arrow.py`) — plain uncompressed Arrow IPC decodes fine client-side. Payload
went from 101KB → 328KB for the same 1,480-row pitch set; still far smaller
than the JSON alternative. If size becomes a real problem later, the correct
fix is registering a JS zstd codec via `apache-arrow`'s `compressionRegistry`
(confirmed it exists in the installed 21.2.0), not reverting this. Verified
post-fix: all three charts render with real data in both light and dark mode,
92+14 tests still pass, `tsc --noEmit` clean.

Also removed a stray uncommitted `console.log(points)` debug line in
`VeloTrend.tsx` found during the same pass (not shipped — it wasn't in the
last commit, just sitting locally).

**Layout/geometry/dark-mode:** full-page screenshot checked for collisions,
overflow, and dark-mode contrast — none found. The at-bat replay strip (viz
#9) is long (one full game, ~87 pitches grouped by at-bat) but reads cleanly;
no visual issues to report there.

Both dev servers are running (`localhost:8000` API — restarted with `--reload`
this session so future backend edits hot-reload, `localhost:5173` UI,
unchanged). `make dev` will also work from a clean start.

---

## 3D pitch trajectory (2026-08-16) — viz #6 pulled forward from M3

Built while the full backfill ran in the background. New `GET
/pitches/trajectory?game_pk=&at_bat_number=&pitch_number=` (`pitches.py`)
returns one pitch's raw 9-parameter physics fit — `vx0/vy0/vz0/ax/ay/az` plus
`release_pos_x/y/z`, `sz_top/sz_bot`, `plate_x/plate_z` — straight off
`fact_pitch` (these columns were already in the lake, just never exposed by
an API route). `lib/trajectory.ts`'s `reconstructFlight()` does the actual
reconstruction client-side; `PitchTrajectory3D.tsx` (vanilla Three.js, no
react-three-fiber — matches the hand-rolled-SVG style of the other charts)
animates it from a fixed near-batter camera. Triggered by a "fly it in 3D"
link on every pitch card in the at-bat replay strip.

**The physics is exact, not approximate — this took real validation, not
just "the numbers look plausible":**
- `vx0/vy0/vz0/ax/ay/az` are valid at Statcast's fixed y=50ft reference, **not
  at the actual release point** (`release_pos_y`, typically ~54ft). Naively
  animating from `release_pos_*` using `vx0` etc. directly draws the wrong
  curve.
- Fix: solve the same quadratic backward from y=50 to `release_pos_y` to get
  velocity *at* release, then integrate forward from there to `y=17/12`
  (front of the plate — confirmed empirically to be where Savant's own
  `plate_x`/`plate_z` are measured, not `y=0`).
- Validated against 200 real pitches: mean error ~0.003ft on both `plate_x`
  and `plate_z`. Locked in as a regression test on **both** sides —
  `TestTrajectory` in `apps/api/tests/test_api.py` and
  `trajectory.test.ts` in the frontend, against the same real pitch. Don't
  change the y=50 or y=17/12 reference points without re-validating both.

**A genuine bug caught by visual verification, not by either test suite:**
`toThree(x,y,z) = Vector3(x,z,y)` swaps two axes, which silently flips
handedness (right-handed Statcast → left-handed Three.js scene) — Three.js
assumes right-handed throughout, so this mirrored the whole scene left-right.
A pitch that actually broke to the batter's right rendered as breaking left.
Fixed by negating x (`Vector3(-x,z,y)`) and re-deriving the camera's
batter-box offset sign to match. Caught by literally looking at where the
ball landed relative to the strike zone box for a known-location pitch and
noticing the horizontal side was wrong — no unit test would have caught this,
since the *shape* of the flight (which is what the physics tests check) is
unaffected by a global mirror.

Camera *starts* near the plate on the batter's box side (offset sign depends
on `stand`), aimed at the zone center, wide FOV (62°) so the pitcher and the
zone both fit in frame — a literal at-the-eyes position can't see its own
strike zone, so this is deliberately a few feet back, not a physically
literal eye position. From there it is user-controlled (OrbitControls: drag to
orbit, scroll to zoom). Playback defaults to real time with a speed slider down
to 0.1x; replay resets only the ball, deliberately **not** the camera — that is
why the animation state lives behind a ref instead of an effect dependency.

Not yet done: no test drives the actual Playwright/visual check for this
(all verification this session was manual, screenshot-and-look). If this
regresses, `bb-ml`-style CI won't catch a coordinate-mirror bug like the one
above — only rendering it and checking will.

---

## Stuff+ / Location+ / Pitching+ (2026-08-16) — M3 model #3, the flagship

Built while the backfill ran. Three LightGBM regressors on ONE target, three
disjoint feature sets: Stuff+ from physical characteristics only, Location+ from
placement and count only, Pitching+ from both. `features/stuff.py` declares the
sets and `assert_sets_are_disjoint()` enforces the wall between them — the whole
value of the triple is that "elite shape, poor command" is readable as a gap
between two numbers, and one leaked column silently turns Stuff+ into a worse
Pitching+.

**The target took more work than the models** (`features/run_value.py`).
`delta_run_exp` straight off Savant is the wrong thing to regress on, for two
reasons measured on our own lake:

- It is a *base-out* delta, so the same strikeout is worth more with the bases
  loaded. A stuff model can't see the base state, so that spread is pure noise.
- Grouping it by description: every non-BIP outcome has SD 0.03-0.09, while
  `hit_into_play` alone has SD **0.487** on 20% of pitches. Almost all of the
  variance is one bucket, and most of that is where the ball landed.

So we build the standard count-based construction from our own data — no
imported linear weights: context-averaged event values, balls in play de-noised
by regressing run value on `estimated_woba_using_speedangle` (rv = -0.253 +
0.829 * xwoba), a 12-cell count run-expectancy table, and pitch RV as the
telescoping difference. Target SD drops 0.231 -> 0.150 and, more importantly,
`E[rv | count] == 0` at every count to 1e-5 — the count carries no run value of
its own, so a model can't score points knowing 0-2 is a good count. Two
consequences that look like bugs and aren't: **a two-strike foul is worth
exactly zero**, and the per-pitch scale is tiny (±0.05).

**Measured, and the numbers argue for the design:**

| head | iters | test r2 | agg corr | yoy grade | yoy actual |
|---|---|---|---|---|---|
| Stuff+ | 77 | 0.0005 | 0.177 | **0.840** | 0.385 |
| Location+ | 286 | 0.0382 | 0.177 | 0.629 | 0.385 |
| Pitching+ | 247 | 0.0395 | 0.257 | 0.714 | 0.385 |

- **R^2 near zero is the expected result, not a failure**, and this was checked
  rather than assumed: 500 / 1500 / 2000 rounds and a finer booster all pushed
  test R^2 *negative* and dropped stability 0.86 -> 0.70. The ~40-80 iteration
  early-stopped fit is not undertrained. Don't "fix" it.
- **Stability is the headline.** The grade says the same thing about a pitcher
  two years running (0.84) more than twice as reliably as his own run value does
  (0.385). That is the entire reason a stuff metric exists.
- **Predictive validity is honest-but-unproven so far**: Stuff+ vs next season's
  run value is 0.37 against past run value's 0.38 — level, not better. The only
  split available is a 9-year jump (train <=2016, test 2025) across the invention
  of the sweeper. Re-measure on a contiguous split once the backfill lands.
- **`pitch_type` is in the Location+ set and not the Stuff+ set** — measured both
  ways (agg corr: stuff 0.173 -> 0.164 with it, location 0.130 -> 0.175 with it).
  It's a bare label, not a quality measurement, so it tells the location model
  *which* pitch is being located without smuggling in how good it is.
- Sanity check that the decomposition is real: José Soriano grades 115.2 stuff /
  94.4 location, deGrom 114.7 / 112.4. That matches the scouting reputations, and
  no single "pitch quality" number would show it.

Scale: `100 + 10 * z`, calibrated over pitcher x season x pitch_type groups of
100+ pitches (not over individual pitches — that would shrink every aggregate
toward 100). Constants live in the artifact, so grades are only comparable
within a training run.

`bb-ml stuff` trains all three and rebuilds `mart_pitcher_stuff` (pitcher x
season x pitch_type + an `ALL` rollup). That mart is written by bbml, not by a
SQL file, because a row means scoring every pitch through three boosters; it
reads the lake directly so it does **not** need the warehouse lock. `make train`
runs it.

---

## Current local data

A background backfill for 2015-2017 landed unattended during this session (not
launched interactively — discovered via a task-killed notification, already
partway through when found) and has been folded into the lake alongside the
original 2025 M1 slice. **Full 2015-present is still not done** — 2018-2024 and
the 2025 offseason gap remain.

- **2,402,136 pitches** across seasons 2015, 2016, 2017, 2025 (2018-2024 missing).
- Marts: `mart_pitcher_arsenal` 13,112 rows; `mart_zone_profile` 18,890 grids
  (8,955 batter / 9,935 pitcher).
- `dim_game`/`dim_player`/crosswalk rebuilt to match (32,862 games, 5,542
  players, 129,658-row Chadwick crosswalk).

**Bug found and fixed while building this:** `enrich()` crashed with `division
with 'String' datatypes is not allowed`. On any day where a physics column
(release speed, movement, `sz_top`/`sz_bot`, `spin_axis`, etc.) was entirely
null for every pitch that day, polars' CSV reader infers `String` rather than
`Float64` for that file — nothing numeric to infer from. `diagonal_relaxed`
concat across a season's ~180 day-files then upcasts the whole column to
`String` the moment one file disagrees. Fixed by pinning ~40 measurement
columns in `SCHEMA_OVERRIDES` (`savant.py`), not just the two that happened to
trigger the crash — the same failure would recur for any future all-null day on
an unpinned column. All error-level `bb check` checks pass post-fix.

Full 2015→present backfill (2018-2024 + rest of 2025) is still outstanding.
Measured cost: 6.4s per game-day sustained. Resumable — safe to start and
interrupt; `bb ingest statcast` will pick up where the manifest left off.
**Retrain both models again once it lands** — the current numbers below were
already retrained once (on 2015-16→17→25) and moved a lot from the single-season
baseline; a full decade will move them further, especially the location model
(13.6% top-1 on 26 classes barely moved between the two runs so far, suggesting
it's not history-starved the way pitch-type prediction is).

---

## Committed

`git log`: two commits — M1 (`6e0d555`) and the M2 model package
(`67e2e16`). The M2 API/frontend work from this session (predict router, replay
UI, registry) is staged for the next commit — check `git status` before assuming
it landed.

---

## Model numbers

Two runs so far — numbers moved a lot between them, which is itself informative.

**Run 1 — single 2025 slice** (train Apr1-Jun3 / val Jun4-17 / test Jun18-Aug5,
same season throughout):

| model | log_loss | top1 | ece |
|---|---|---|---|
| baseline (per-pitcher usage) | 1.5942 | - | - |
| next-pitch (global LightGBM) | 1.2736 | 0.457 | 0.0193 |
| next-pitch + personalized blend | 1.2840 | 0.460 | 0.0061 |
| location (26-class grid) | 2.9572 | 0.135 | - |

**Run 2 — current, multi-season** (train 2015-16 / val 2017 / test 2025 — an
8-year gap, forced by `auto_split` since those are the only 4 seasons present):

| model | log_loss | top1 | ece |
|---|---|---|---|
| baseline (per-pitcher usage) | 2.2788 | - | - |
| next-pitch (global LightGBM) | 1.3886 | 0.434 | 0.0438 |
| next-pitch + personalized blend | 1.3981 | 0.434 | 0.0282 |
| location (26-class grid) | 2.9659 | 0.136 | - |

The baseline got much worse across the 8-year gap (roster turnover — most 2025
pitchers have zero 2015-16 history for a pure per-pitcher lookup to use) while
the model held up, so **improvement over baseline jumped from 5% to 39%** — a
more convincing demonstration that the feature-based personalization
generalizes rather than memorizes. ECE is worse (0.0438 vs 0.0193, though still
under the 0.05 test gate) — expected, since the model is now predicting into a
pitch-type landscape 8 years removed from training (e.g. the sweeper "ST" barely
existed in 2015-16). **Re-run once the 2018-2024 gap fills in** and a
contiguous/nearer split becomes possible — this split is an artifact of which
seasons happen to be backfilled, not a deliberately chosen evaluation design.

Personalization is answered in full in conversation history: one global model,
`pitcher` deliberately not a feature, personalization via expanding-window
per-pitch-type usage priors. Per-pitcher models measured ~20% worse on the
single-season run, and the multi-season run's baseline collapse only reinforces
that conclusion. See `next_pitch.py` module docstring for the full writeup —
don't re-derive it.

---

## Resume paths

**Finish M2 properly:**
1. ~~Visually verify the replay UI~~ — done 2026-08-15, see above.
2. Run the full backfill, retrain both models (`bb-ml next-pitch`, `bb-ml
   location`), see if the location model's 13.5% top-1 improves with more data.
3. Consider a location arsenal-style prior (where a pitcher tends to miss) as a
   feature — the location model currently uses the same feature set as pitch
   type, which wasn't built with location-specific signal in mind.

4. Re-run `bb-ml stuff` on the full lake and re-check predictive validity on a
   contiguous split — the current 0.37-vs-0.38 result is not yet a conclusion.

**Or move to M3:** live game-feed mode; `PostgresWarehouse`; models 2 and 5
(arsenal re-classification, called-strike probability — model 3 is done, see
above); viz 7-8, 10-20; Retrosheet backfill.

---

## Decisions already made — don't relitigate

- **Colour encodes pitch *family* (3 hues), shape encodes pitch type.** Not
  cosmetic: the movement plot is a scatter, so it falls under the all-pairs CVD
  gate, which the validated palette clears with three slots. Nine pitch types
  cannot take nine hues. Centroid labels + the arsenal table supply the required
  contrast relief. The replay strip reuses the same family-color mapping.
- **Arrow IPC for pitch-level routes, JSON for everything else.** Measured 6.8x
  smaller on real data. Retrofitting would mean rewriting every chart's data path.
- **`season` is written into the Parquet files, not just the directory name,** and
  `hive_partitioning=false` everywhere. DuckDB 1.5.5 throws an InternalException
  when a query projects *only* a synthesized partition column. Do not "simplify"
  this back to hive synthesis.
- **`bb ingest dims` must run AFTER `bb build pitches`** — `dim_player` is
  populated from the ids present in `fact_pitch`. The Makefile `pipeline` target
  encodes the right order; running it wrong silently builds a partial dimension.
- **Cutters are ranked below four-seams/sinkers when picking the baseline
  fastball** for velo/movement deltas. A cutter is its own pitch class; anchoring
  on it reports a pitcher's sinker as +7mph "offspeed separation".
- **`pitcher` is deliberately not a next-pitch feature.** Personalization is via
  expanding-window per-pitch-type priors, not a pitcher ID or a per-pitcher model
  — measured, not assumed (see `next_pitch.py`).
- **The arsenal mask defaults off** — hard-zeroing pitches outside a pitcher's
  learned arsenal made log-loss and ECE both worse; it inflates top-1 by making
  argmax cleaner while quietly ruining the probabilities the UI shows.
- **The pitch-quality feature sets deliberately break the next-pitch leakage
  rule.** `schema.FORBIDDEN_FEATURE_COLUMNS` bans anything describing the pitch
  being predicted; Stuff+/Location+ are *grading a pitch already thrown*, so
  those columns are the entire input. `auto_split(..., check_features=False)`
  is the opt-out and it is correct — the temporal and plate-appearance checks
  still run.
- **MLflow uses a sqlite backend**, not the plain file store — the file store is
  in maintenance mode and now raises on `set_experiment`. Tracking URI is
  `sqlite:///data/models/mlruns/mlflow.db`.

## Gotchas that cost time to rediscover

- **Savant truncates silently at 25,000 rows** — HTTP 200, no marker, data just
  stops mid-day. Guarded, but never widen the date partition without re-checking.
- **Savant revises published data** after the fact. `bb ingest refresh` re-pulls a
  trailing window; append-only ingest goes stale invisibly.
- **Statcast's `umpire` column is empty in every season.** Umpires come from the
  Stats API boxscore (`bb ingest officials`), one request per game — needed for
  the framing/called-strike models, skipped so far.
- **DuckDB persists a view's resolved schema.** Rebuilding the lake with a changed
  column set leaves stale views that fail confusingly. `build pitches` now
  re-registers automatically; keep it that way.
- **Bat tracking is 2024+, swing path 2025+.** Nullable across the whole lake.
  `bb check --coverage` reports per-season availability — read it rather than
  assuming.
- **`_prior_sum` needs `fill_null(0)` on the counted expression**, not just on the
  result — without it, every prior comes out null in live inference because the
  pending pitch's own indicator is null. Caught by the parity test; don't remove
  the `fill_null` while "simplifying" that function.
- **An all-null-for-the-day column infers as `String`, not `Float64`.** Any
  measurement column with zero non-null values in one day's raw CSV (common in
  older seasons before a stat existed) makes polars pick `String` for that file;
  `diagonal_relaxed` concat across a season then upcasts the whole column to
  `String`. Every physics/measurement column must be pinned in
  `SCHEMA_OVERRIDES` — don't assume a new Statcast field is safe to leave
  uninferred just because recent seasons look fine.
- **`open_warehouse` takes an exclusive DuckDB lock** — `bb build`/`bb ingest`
  will fail with `IOException: Could not set lock` if the API server (or
  anything else holding a `DuckDBWarehouse`) is still running. Stop it first.

## Quick sanity check after `git pull` / fresh session

```bash
uv sync --python 3.13 && cd apps/web && npm install && cd ../..
uv run pytest && uv run bb check
uv run bb status          # ingest manifest
uv run bb-ml status       # registered model versions
```
