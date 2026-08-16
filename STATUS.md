# Development bookmark

**Paused:** 2026-08-16 · **M1 and M2 complete and visually verified end-to-end.
Three M3 items pulled forward and finished: the 3D pitch trajectory (viz #6),
Stuff+ / Location+ / Pitching+ (model #3), and the swing-path model (model #4).**
The full 2015-2026 backfill has landed (9,202,082 pitches, contiguous) and
every model has been retrained on it — see "Full lake rebuild + retrain" below
for the numbers, several of which overturned earlier guesses in this file.
Officials data (umpire per game) is also fully ingested now (11,154 games),
unblocking model #5 (called-strike probability / framing) whenever that's
picked up.

Read this first in a new session, then `README.md` for how the thing works and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M3 backlog.

---

## Where we are

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite. Complete. |
| `packages/bbml` | Feature builder (batch+live, parity-tested), datasets/splits, `UsageRateBaseline`, `NextPitchModel` (pitch type), `LocationModel` (26-class grid), `PersonalizedBlend`, `RunValue` + `PitchQualityModel` (Stuff+/Location+/Pitching+, M3 model #3), **`SwingPathModel` (whiff + contact heads, matched-counterfactual `plane_value`, M3 model #4 — see below)**, `registry.py` (versioned artifacts + optional MLflow), `marts.py`, `bb-ml` CLI. |
| `apps/api` | `/predict/next-pitch` (what-if), `/games/{game_pk}/replay`, `/players/{id}/games`, `/pitches/trajectory`, `/stuff/{id}` + `/stuff` leaderboard added. 14 routes total, JSON + Arrow IPC. **No swing-path route yet** — model #4 is trained and registered but not exposed via the API or UI. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table, at-bat replay strip (viz #9), **3D pitch trajectory** (viz #6), **pitch quality panel** (model #3). Visually verified. No swing-path surface yet. |

**Verification status:** 147 backend Python tests (bbcore/bbetl/bbml/api,
including 18 new swing-path tests) + 18 frontend tests, `tsc --noEmit`,
`oxlint`, `ruff check`, `bb check` (data quality — all error-level checks
pass), `bb-ml status` all pass/registered. All seven models retrained on the
full 9,202,082-pitch 2015-2026 lake (see "Full lake rebuild + retrain" below)
and saved to
`data/models/{next_pitch,location,stuff_plus,location_plus,pitching_plus,swing_whiff,swing_contact}/`.
**The rendered UI has been visually verified** (Playwright/Chromium
screenshots, light + dark, `Tarik Skubal`) — see below. That verification
predates the swing-path model, which has no UI yet to verify.

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

## Full lake rebuild + retrain (2026-08-16) — the backfill landed, and it moved things

The 2018-2024 gap and the 2025 offseason filled in while the swing-path model
(below) was being built. Rebuilt the lake and retrained everything on it —
9,202,082 pitches, seasons 2015-2026 contiguous, no gap. This also added
`vaa_deg`/`haa_deg` (pitch vertical/horizontal approach angle at the plate,
solved from the 9-parameter physics fit rather than read off the y=50
reference — see `bbetl.transforms.statcast` for why the shortcut reports the
angle 48ft early and flattens every pitch) and `plate_z_norm` where needed —
`swing_path.py`'s feature set depends on both.

**Next-pitch — the prior run's ECE regression was a split artifact, not a
calibration problem:**

| | 8-year-gap split (old) | full contiguous lake (now) |
|---|---|---|
| baseline log-loss | 2.2788 | 1.8326 |
| model log-loss | 1.3886 | 1.2973 |
| top-1 | 0.434 | 0.443 |
| ECE | 0.0438 | 0.0131 |
| ECE + personalized blend | 0.0282 | 0.0034 |

Calibration recovered 3.3x. The model was never miscalibrated — it was being
scored against a pitch-type landscape 8 years removed from its training data
(the old split trained on 2015-16, tested on 2025, because those were the only
seasons backfilled at the time). 29% better than baseline on a fair split.

**Location model — this retracts a claim earlier in this file.** Top-1 went
0.136 -> **0.159** (+2.3pp) on 6.7M training rows, up from a run that trained on
a fraction of that. The "Current local data" section used to say the location
model's top-1 "barely moved between the two runs, suggesting it's not
history-starved" — that inference was wrong; it was history-starved, just like
next-pitch. Corrected here rather than left for someone to re-trust later.

**Pitch quality — Stuff+ especially:**

| head | agg corr (partial -> full) | yoy stability (partial -> full) |
|---|---|---|
| Stuff+ | 0.177 -> 0.258 | 0.840 -> 0.864 |
| Location+ | 0.177 -> 0.200 | 0.629 -> 0.663 |
| Pitching+ | 0.257 -> 0.368 | 0.714 -> 0.745 |

Stuff+ is now 2.24x more stable year-to-year than the run value it grades
(0.864 vs the ~0.385 yoy stability of raw run value itself).

**Still open:** `predictive_validity` (Stuff+ vs next season's run value on a
contiguous split — the one open question flagged when model #3 shipped) is
computed in the training script but isn't actually persisted anywhere durable.
`save_model` only forwards metrics to MLflow, which isn't installed in this
environment (`registry.py` warns and continues), so the number is printed to
the console and then gone. The artifact-on-disk docstring claims "the artifact
on disk is the source of truth"; for metrics that isn't true yet. Fix is a
small one — write a `metrics.json` beside each artifact in `save_model` — and
it hasn't been done. Do that before trying to answer the contiguous-split
predictive-validity question for real.

Officials data (`bb ingest officials`, per-game home-plate umpire from the
Stats API boxscore) also finished during this session: **11,154 games**, full
coverage. Unblocks model #5 (called-strike probability -> catcher framing runs
+ umpire zone maps) whenever that gets picked up — see the plan already worked
out for it (below, and in the assistant's memory for this project).

An exploratory zone-expansion-by-count analysis was built and published as a
standalone artifact ("The Elastic Strike Zone") using a partial officials
sample (92 umpires, ~6,300 games ingested at the time) — headline: the
called-strike zone's *size*, not just its center, changes by count (122% of
rulebook in a 2-0 count, 34% in 0-2), and on the selection-controlled
borderline band strike rate swings 2.1x by count (30.5% -> 64.9%). The
per-umpire cut in that artifact is stale now that officials data is complete
(11,154 vs ~6,300 games) — queued to re-run, not yet done.

---

## Swing-path model (2026-08-16) — M3 model #4: is this batter's plane good against what he actually sees?

`features/swing.py` + `models/swing_path.py`. The finding it's built on,
measured on 1.04M tracked swings (2023H2-2026): whiff rate by swing plane
(attack angle) against pitch descent angle (approach angle) is a strong
*interaction*, not two main effects — a steep uppercut whiffs 11.6% against
flat pitches and 57.8% against steep ones (5x), while a flat swing runs the
opposite direction. A swing plane is not good or bad in itself; it is good or
bad against a particular pitch. See the module docstring for the full table.

**Two models, same swing frame:**
- `whiff` — P(whiff | swing), AUC **0.896**. Where nearly all of the geometry
  effect lives.
- `contact` — E[xwOBA | contact], R² **0.080**. Same interaction, much weaker,
  as the finding above predicts (contact quality varies less by plane/pitch
  match than whether contact happens at all).

**`plane_value` is a counterfactual, not a per-batter average**, because a raw
average can't separate "his plane suits the pitches he sees" from "he sees
flatter pitches" — pitch selection and pitcher quality are baked into any raw
number. Each swing is scored twice against the *same pitch*: once at the
batter's actual attack angle, once at a **matched league-median swing**, with
location/count/pitch type held fixed as controls (needed, or the whole result
would read as "uppercuts miss low breaking balls" — a location fact, not a
geometry one).

**A real bug was caught here, not just a design choice measured both ways.**
The first version of the counterfactual swapped only `attack_angle` to the
league median and froze bat speed / swing tilt / contact point at the
individual hitter's own values. That isn't a real swing: a 25-degree
attack-angle hitter meets the ball in a measurably different place than a
9-degree one (mean contact point +2.5in pull-side vs -7.0in), and "9-degree
attack angle with a 25-degree hitter's contact point" barely exists in
training data. The model was extrapolating into that gap, and on real held-out
data it returned the **wrong sign** — `plane_value` averaged -0.013 (t≈-7,
n=2,512, not sampling noise) on a slice where the raw whiff-rate gap is
unambiguous (11.2% vs 21.6%, every season 2023-2026) — caught by
`test_plane_value_is_positive_when_the_plane_helps`, which is pinned against
real data rather than a synthetic fixture for exactly this reason.

**Fix:** `matched_neutral` — small single-feature auxiliary regressors
(`CORRELATED_SWING_FEATURES` ~ `attack_angle`), fit once at training time and
evaluated at `league_plane`, give the reference swing the bat speed/tilt/contact
point a *typical* league-median-attack-angle swing actually has, instead of the
individual hitter's own. Recovers `plane_value = +0.092` (t≈28) on the same
slice — matches the raw effect. The 120-round test fixture also wasn't enough
model capacity to resolve this interaction reliably; bumped to 600 (measured:
still wrong sign at 120 and 300 rounds, stable and correct by 600).

**Also corrected while building this:** this file's gotchas section long said
swing path is 2025+. It's **2023H2+** — Savant backfilled it (rollout: 0%
before July 2023, 62% that month, ~95% after; attack-angle mean 8.2-8.5°, bat
speed 70.9-71.3mph, stable across all four seasons). That's ~4x the training
data. Fixed in the gotchas list below.

Not yet done: no API route, no UI panel. `bb-ml swing` trains and registers
both heads and prints a plane-value leaderboard; that's as far as it goes today.

---

## Current local data

- **9,202,082 pitches**, seasons 2015-2026, contiguous — the full backfill
  finally landed (see above). No more season gaps.
- Marts: `mart_pitcher_stuff` 40,539 rows (pitcher x season x pitch_type + an
  `ALL` rollup). `mart_pitcher_arsenal` / `mart_zone_profile` not re-counted
  this session — re-run `bb check --coverage` before trusting the old figures
  below.
- Officials: **11,154 games** with a home-plate umpire (`bb ingest
  officials`), full coverage.

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

The backfill is now complete (see "Full lake rebuild + retrain" above) — this
paragraph's cost/resumability notes are kept for the ingest machinery itself
(measured 6.4s per game-day sustained, resumable via `bb ingest statcast`
picking up where the manifest left off), useful if a future re-ingest is ever
needed, but there is no gap left to fill right now.

---

## Committed

`git log` (newest first): M3 model #3 (`f979595`), 3D pitch trajectory work
(`4b77e20`/`d3d9c2f`/`782a5b5`), the zstd Arrow fix (`cbf07a8`), an earlier
STATUS.md update (`4694c91`), the schema-overrides fix (`20d776b`), `make
train` (`3455af5`), M2 (`00d8ad9`), the first bbml package (`67e2e16`), and M1
(`6e0d555`). This session's work — `vaa_deg`/`haa_deg` in the transform layer,
the swing-path model (model #4) and its matched-counterfactual fix, and this
STATUS.md update — lands in the commit right after this one; check `git log`
rather than trusting this list to stay current.

---

## Model numbers

Three runs so far — numbers moved a lot between them, which is itself
informative. See "Full lake rebuild + retrain" above for the run-2 -> run-3
comparison and what it settled (the ECE regression was a split artifact, not
a calibration problem; the location model was history-starved after all).
Run 3's contiguous split (train up through some cutoff / val / test on the
most recent season) is the first fair evaluation — re-run `bb-ml next-pitch`
and pull the exact split boundaries from its output before quoting them
precisely; the headline deltas are recorded above.

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

**M2 is done.** Both items that used to sit here are resolved:
1. ~~Visually verify the replay UI~~ — done 2026-08-15, see above.
2. ~~Run the full backfill, retrain~~ — done 2026-08-16, see "Full lake
   rebuild + retrain" above. Location model's top-1 did improve with more
   data (0.136 -> 0.159); the "not history-starved" guess in an earlier
   version of this file was wrong.

Still open from that list:
- Consider a location arsenal-style prior (where a pitcher tends to miss) as a
  feature — the location model currently uses the same feature set as pitch
  type, which wasn't built with location-specific signal in mind.
- Fix `save_model` to persist a `metrics.json` beside each artifact (MLflow
  isn't installed here, so metrics currently evaporate — see "Full lake
  rebuild + retrain" above), then answer the Stuff+ predictive-validity
  question on a proper contiguous split.

**M3, in progress:** model #3 (Stuff+/Location+/Pitching+) and model #4
(swing-path) are both done — see above for both. Model #4 has no API route or
UI panel yet; that's the fastest next win if the goal is a shippable feature
rather than a new model. Model #5 (called-strike probability -> catcher
framing + umpire zone maps) is fully scoped and ready to start now that
officials data is complete — the plan is written up in the assistant's
project memory (`baseball-model5-called-strike-plan`), not re-derived here to
avoid drift between two copies. Also queued: re-run the zone-expansion
umpire analysis with the complete 11,154-game officials data (the published
artifact used a partial 92-umpire sample). Still not started: live game-feed
mode, `PostgresWarehouse`, model #2 (arsenal re-classification), viz 7-8,
10-20, Retrosheet backfill.

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
  the framing/called-strike models. Fully ingested now (11,154 games, model #5
  ready to start).
- **DuckDB persists a view's resolved schema.** Rebuilding the lake with a changed
  column set leaves stale views that fail confusingly. `build pitches` now
  re-registers automatically; keep it that way.
- **Bat tracking / swing path is 2023H2+, not 2025+ as this file used to
  claim.** Savant backfilled it (0% before July 2023, 62% that month, ~95%
  after — attack angle and bat speed distributions stable across all four
  seasons since). That wrong assumption would have cost ~4x the training data
  if it had gone unchecked into `swing_path.py`. Nullable before 2023H2 and
  still nullable per-row after. `bb check --coverage` reports per-season
  availability — read it rather than assuming.
- **A single-feature counterfactual that freezes correlated features at their
  actual values can flip sign on real data**, not just add noise — see the
  swing-path `matched_neutral` fix above. If a future counterfactual metric
  perturbs one feature while holding others fixed at an individual's own
  values, check whether those held-fixed features are themselves correlated
  with the perturbed one before trusting the sign.
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
