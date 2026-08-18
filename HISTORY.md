# Development history

Dated, narrative write-ups of past sessions — the "how we got here" and the
bugs found along the way. `STATUS.md` at the repo root is the lean, current
state; this file is the archive it points back to for detail. Read
`STATUS.md` first; come here when a "Decisions already made" or "Gotchas"
entry there says "see HISTORY.md" for the full story behind a decision.

Sections are in chronological order (oldest first).

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
data. Fixed in the gotchas list in `STATUS.md`.

Not yet done at the time this section was written: no API route, no UI panel.
Both landed later — see "API + UI for models #4/#5" below.

---

## Called-strike model (2026-08-16) — M3 model #5: framing runs, from scratch

`features/called_strike.py` + `models/called_strike.py`. Binary classifier:
P(called strike) on TAKES only (`is_swing=false`, description in
ball/called_strike/blocked_ball — a hit-by-pitch is excluded, it's not a
ball/strike decision). Inputs are the pitch as it crossed the plate — location
(mirrored to `plate_x_out`, same convention as `stuff.py`), `plate_z_norm`,
movement, velocity, count, `pitch_type` as a bare label. **Catcher and umpire
are deliberately not features** — same principle as keeping `pitcher` out of
next-pitch: the entity `framing_runs` grades can't be an input to the model
doing the grading, or the residual collapses toward the entity's own skill.

**Numbers on the full lake, contiguous split (train <=2024 / val 2025 / test
2026):** AUC **0.988**, log-loss **0.134**, ECE **0.017** on 283,652 held-out
takes. ECE — not AUC — is the metric that gates this model for actual use:
`framing_runs` sums `(actual - p)` per catcher, so a probability that's off by
a constant amount manufactures fake *uniform* framing value league-wide even
with perfect ranking. 0.017 clears the 0.03 test gate comfortably.

**Framing runs formula reuses `RunValue` rather than building a new table:**
`RunValue.marginal_strike_value(balls, strikes)` — a new method, not a new
model — is the run-expectancy swing between a take being called a ball vs a
strike at that count, built from the same count run-expectancy table
Stuff+/Location+/Pitching+ already fit. `framing_runs = Sum (actual_strike -
P(strike)) * marginal_strike_value(count)`, grouped by catcher or by umpire.
Terminal counts (3-2 ball -> walk, *-2 strike -> strikeout) route to
`event_value`, not a missing `count_re` entry — the one subtlety worth
remembering if this ever needs re-deriving.

**Validation, not just a metric:**
- **YoY catcher stability** (732 catcher-season pairs): Spearman **0.53** —
  inside the 0.5-0.7 range published framing metrics report. Evidence this is
  measuring a real, sticky catcher property rather than single-season noise.
- **Named-catcher sanity check** (career framing runs, >=3,000 takes,
  2015-2026): top 3 are **Yasmani Grandal, Austin Hedges, Tyler Flowers** —
  the actual best-known elite framers in the sport over this window. Bottom
  is led by **Salvador Perez, Elias Díaz, Shea Langeliers** — also matches
  public reputation. Not tuned to produce this; it's what the residual says.
- **Umpire zone-edge spread** (borderline takes, 0.2<=P(strike)<=0.8, 86
  umpires with >=1,000 such takes): edge (actual - expected borderline strike
  rate) SD **0.0385** — real, but smaller than the catcher spread, consistent
  with the earlier zone-expansion finding that umpires are the tightest of
  the three entities (batter/pitcher/umpire) on borderline calls.

**`dim_official` now exists** (`bbetl/transforms/officials.py`, `bb build
officials`) — the 11,154-game officials ingest from last session had never
been turned into a lake table before this; the raw JSON just sat in
`data/raw/officials/`. This is also the deferred zone-expansion umpire rerun:
`dim_official` is the input that analysis needed and didn't have.

**Two new marts**, scored by the registered model and rebuilt by `bb-ml
called-strike-mart` (also runs automatically at the end of `bb-ml
called-strike`): `mart_catcher_framing` (1,048 rows, catcher x season) and
`mart_umpire_zone` (230 rows, umpire x season — see the null-group bug fix
below for why this isn't 238 anymore) — the data viz #20 (catcher framing
map) and viz #13 (umpire zone map) need.

Not yet done at the time this model was built: no API route, no UI panel, no
viz #20/#13. All landed later — see "API + UI for models #4/#5" and "Viz
#20/#13" below. `bb-ml called-strike` trains, registers, and prints a
framing-runs leaderboard; `bb-ml called-strike-mart` rebuilds the two marts
standalone.

---

## API + UI for models #4/#5 (2026-08-16) — swing-path and called-strike get a surface

Both models were trained and registered with nothing reading them. This adds
the API routes, the marts model #4 never had, and player-page panels for
both — plus two real bugs the process of wiring it up surfaced.

**New mart: `mart_batter_swing`** (`bbml.marts.build_batter_swing_mart`,
1,611 rows, batter x season). Model #4 had no mart at all before this —
`plane_value_by_batter` existed but only ever ran inside `bb-ml swing` and
printed a top-5 table to the console. Runs both heads and joins them into one
row per batter-season; `bb-ml swing` now calls `bb-ml swing-mart`
automatically at the end, same as `stuff`/`called-strike` already did for
their marts.

**New API routes**, all reading marts directly (no live model inference —
same pattern as `/stuff`): `/swing/{id}` + `/swing` leaderboard,
`/framing/catchers/{id}` + `/framing/catchers` leaderboard, `/framing/umpires`
leaderboard. Umpires have no `dim_player` row (they aren't players), so the
umpire leaderboard joins a deduplicated `dim_official` subquery for the name
instead of the usual `dim_player` join every other leaderboard uses.

**New player-page panels**, both gated on the existing `role` filter rather
than a new concept: `SwingPanel` shows for `role === "batter"` (attack angle,
whiffs-avoided/100, contact-value/100, diverging color around zero — zero is
the actual meaningful reference point here, not a league average, since the
counterfactual is already relative to a neutral swing). `FramingPanel` shows
additionally gated on `player.primary_position === "C"` (framing runs, takes
received). Visually verified in the browser, light and dark, on `Luis Arraez`
(batter) and `Jose Trevino` (catcher) — numbers on screen cross-checked
against the training-run leaderboard and matched exactly (e.g. Arraez 2026:
attack angle 7.6°, +12.29 whiffs avoided/100, matching `bb-ml swing`'s own
printed leaderboard from the model #4 training run).

**Bug #1 — `dim_official` was written but never registered in the
warehouse.** `bb build officials` wrote the Parquet correctly, but nothing
called `wh.register_lake_table` for it — `dim_game`/`dim_team`/`dim_player`
and every mart go through `bbetl.warehouse.LAKE_TABLES` +
`register_all()`/`bb build register`, and `dim_official` was never added to
that dict. Silent failure mode: `/framing/umpires` 503'd with "table not
built" even though the file existed on disk and `bb build officials` had
reported success. Fixed by adding `dim_official` (and, for the same recovery
reason `mart_pitcher_stuff` already documents, `mart_batter_swing` /
`mart_catcher_framing` / `mart_umpire_zone`) to `LAKE_TABLES`, then running
`bb build register` once to pick up the already-built files.

**Bug #2 — a real one, not a wiring gap: `framing_runs`/`umpire_zone_rate`
grouped by a column that can be null, and null formed its own group.**
`dim_official` only covers 2023+ (see below); every take from an
uncovered game has `UMPIRE_COLUMN = null`. Grouping by it without filtering
put every pre-2023 take into one giant "unknown umpire" row — 50,000+ pitches
in some seasons — that swamped every real umpire by sample size alone
whenever a query didn't explicitly filter it out. Caught by hand while
building the API (`mart_umpire_zone` had exactly 8 more rows than it should,
one per season 2015-2022), not by a test, because no existing test
constructed a frame with a null group column. Fixed in both functions
(`models/called_strike.py`) by dropping null-group rows before aggregating,
plus two regression tests
(`test_a_null_group_column_is_dropped_not_grouped`,
`test_a_null_umpire_is_dropped_not_grouped`) that build exactly that
scenario. `mart_umpire_zone` went from 238 rows to 230 after the fix — the 8
phantom rows are gone.

**Also discovered while validating the API default:** the umpire leaderboard's
`min_pitches` default was set to 1,000 by analogy with the catcher
leaderboard's 500, without checking against real data — no umpire-season ever
clears 1,000 *borderline* takes (`mart_umpire_zone.n`, the borderline-only
count `umpire_zone_rate` restricts to; max observed is 803). Lowered the
default to 500 to match `MIN_UMPIRE_PITCHES`, the mart's own build floor —
anything stricter than the floor the mart itself used returns nothing, for
every umpire, every season.

---

## Viz #20/#13 (2026-08-16) — the two visualizations model #5 unlocked

Both reuse `StrikeZoneHeatmap` and `mart_zone_profile` exactly as the model #5
plan called for — no new chart component, no new API route. `/zones/{id}`
already took any `role`/`metric` pair; it just grew two more (`role=catcher`
+ `metric=framing`, `role=umpire` + `metric=strike_rate`), and
`mart_zone_profile` grew two more per-role Parquet files
(`catcher.parquet`, `umpire.parquet`) alongside the existing `batter.parquet`
/ `pitcher.parquet` — same directory, same glob, so the warehouse view picks
them up with no re-registration needed.

**The grid-building logic lives in `bbml.marts` now, not `bbetl.transforms.zones`**,
because it's the first zone grid that needs a model score rather than a raw
lake column. `_build_entity_grids` is the shared helper (id column, role,
`MetricSpec`, per-season pitch qualifier in, one grid row per qualifying
entity-season out) — it calls the *same* `build_grid`/`GRID_N` machinery
`bbetl.transforms.zones.build_zone_profiles` already uses for batter/pitcher
grids, imported across the package boundary (`bbml` now depends on `bbetl` in
`pyproject.toml` — it didn't need to before this). Built the null-id-column
filter in from the start rather than discovering it the way
`framing_runs`/`umpire_zone_rate` did (see "API + UI for models #4/#5" above)
— `UMPIRE_COLUMN` is structurally null pre-2023, and grouping by it unfiltered
would produce the same phantom-row bug in a grid instead of a scalar. Pinned
with `test_marts.py::TestBuildEntityGrids::test_a_null_id_column_is_dropped_not_grouped`.

- **`build_catcher_framing_grid`** (viz #20): weight per pitch is
  `actual_strike - P(strike)`, the same residual `framing_runs` sums, left
  un-aggregated by count so the surface reads as "where does this catcher's
  receiving gain or cost strikes" rather than a run total. **1,048 catcher-season
  grids** — same count as `mart_catcher_framing`, same 500-pitch qualifier.
  Rendered as a second card inside the existing `FramingPanel` (now takes an
  optional `grid` prop) rather than a new section — the scalar number and the
  map are the same fact at two resolutions, so they sit side by side. Verified
  on Jose Trevino: a strong "steals strikes low, especially low-and-away"
  signature (dark red band just below and outside the rulebook zone, blue
  above) — matches his public reputation as a low-ball framer, not tuned to
  produce this.
- **`build_umpire_zone_grid`** (viz #13): weight is the umpire's own raw
  `is_called_strike` (×100), no model score needed — unlike the catcher grid
  this metric doesn't need a residual, the actual call rate by location IS
  the thing being mapped. **362 umpire-season grids** (more than
  `mart_umpire_zone`'s 230 — that mart restricts to *borderline* takes only
  for its scalar edge calculation; the grid's own 500-total-pitch qualifier
  is looser). Umpires have no player page (no `dim_player` row), so this
  needed a new standalone view: a top-level "Players / Umpires" tab in
  `App.tsx` (plain `useState`, no router — the app never had one and still
  doesn't need one for two views) rendering a new `UmpiresPage.tsx` — a
  `/framing/umpires` leaderboard table on the left, the selected umpire's
  zone map on the right.
- **New: `lib/contour.ts`, a small marching-squares tracer**, `StrikeZoneHeatmap`'s
  new optional `contourAt` prop. The umpire zone map is unreadable as color
  alone near 50% (a smooth gradient through the diverging ramp's neutral
  band), so the client traces the surface's own 50% crossing as a line on top
  — visually distinct from the thin rulebook-zone rectangle the chart already
  draws, so both are readable at once. Segments are returned unstitched (a
  flat list of line pieces, not closed polygons) since a bare contour line
  renders identically either way and stitching would be real complexity for
  no visual gain. `StrikeZoneHeatmap`'s legend also generalized: `MetricDef`
  gained an optional `legendLabels` override (`"ball"`/`"strike"` for the
  umpire map, `"fewer strikes called"`/`"more strikes called"` for the
  catcher map) since the existing pitcher/batter-derived labels don't apply
  to either.

**Real numbers, not hypothetical.** Willie Traynor (2026, 554 borderline
takes) has the widest edge in the league at -14.6pp / -13.0 framing runs, and
his zone map shows why: a visibly tight, almost perfectly rectangular 50%
contour that undershoots the rulebook zone on the sides — a hitter's-count
umpire on borderline pitches, not noise. Verified in the browser, light and
dark, for both the Trevino framing map and the Traynor zone map; no console
errors beyond the pre-existing FA/EP/KN duplicate-React-key warning noted
above (still out of scope as of this session).

---

## Arsenal re-classification (2026-08-16) — M3 model #2: what does this pitcher actually throw?

`features/arsenal.py` + `models/arsenal.py`. Savant's own `pitch_type` comes
from an automated classifier with known failure modes at the boundary between
similar shapes — a slider and a sweeper differ mainly by degree, not by kind,
so one real blended pitch can get split across both labels, or two pitchers'
genuinely different sliders can get merged under one. This model re-checks
each pitcher-season's arsenal against Savant's labels and reports where and
how much they disagree — it does not replace `pitch_type` anywhere else in
the codebase.

**This one took four real, measured wrongs before the design was sound —
the fullest "measured, not assumed" story in this file.** All four are
written up in full in the module docstrings (`models/arsenal.py`,
`features/arsenal.py`); summarized here:

1. **Unsupervised GMM+BIC from scratch never converged.** Fitting k=1..6
   chosen by BIC over each pitcher-season's own pitches, no anchor to
   Savant's labels: 632 of 640 qualifying 2025 pitcher-seasons hit the
   component ceiling exactly. Switching to full covariance (diagonal was
   letting BIC "explain" real feature correlation by adding clusters) and
   capping the fit to a bounded subsample (BIC's penalty grows with log(n),
   a season's pitch count does not) narrowed it but didn't fix it — a real
   fastball still split into two near-identical clusters. **Redesigned** to
   anchor the search in Savant's own labels: pairwise merge tests (does a
   single Gaussian fit two Savant-labeled groups' union as well as two do?)
   and per-group split tests (does one label clearly contain two
   sub-populations?), both decided by the same 1-vs-2-component BIC test
   requiring Kass & Raftery's "very strong evidence" bar (10 BIC units), not
   merely "lower."
2. **The anchored design still over-split — a date confound.** A pitcher's
   single `FF` group still split with overwhelming evidence (BIC gap 200) at
   two near-identical shapes; the two "clusters" turned out to be almost
   perfectly separated by `game_date` (2025-03-27 to 2025-06-10 vs
   2025-06-07 to 2025-07-29) — within-season drift, not two concurrently
   thrown pitches. Fixed by rejecting any split whose two halves are
   separated by date about as well as a rank-based AUC of 0.85 (measured on
   this exact case: 0.002, i.e. essentially perfect chronological
   separation).
3. **A different pitcher's fastball, sinker, AND curveball all split the
   same way — a release-point confound.** Comparing the split clusters
   feature-by-feature: velocity, movement, and spin axis were all
   essentially equal; only `release_pos_x_arm` differed (1.06ft vs 0.45ft).
   A bimodal release point is real but is not a different pitch, and
   including it as a clustering input let arm-slot variance masquerade as
   an extra pitch on every pitch type a pitcher threw at once. Fixed by
   dropping `release_extension`/`release_pos_x_arm`/`release_pos_z` from
   the clustering features entirely — only velocity, movement, and spin
   axis define "which pitch is this."
4. **A full-history build's ten worst disagreements were *all* from
   2016-2019 — a tracking-era discontinuity, not a model bug.** One
   pitcher with 5 Savant pitch types came back as 13 clusters, 100% pure.
   Checked season-by-season: mean `|arsenal_size_diff|` is 2.8-3.2 for
   2016-2019 (only 14-23% of pitcher-seasons within +-1 of Savant) and drops
   in a clean step, not a gradual trend, to 0.8-1.2 from 2020 on (65-83%
   within +-1) — exactly the boundary where Statcast unified every park onto
   one Hawk-Eye tracking system. A spot-checked 2019 pitcher's doubled
   changeup wasn't explained by either earlier fix (date AUC 0.34, no
   chronological pattern; differed in velocity, movement, *and* spin axis at
   once, spread over 20+ games each side) — genuinely less consistent
   pre-Hawk-Eye measurement. `build_arsenal_cluster_mart` now defaults to
   2020+ (`MIN_RELIABLE_SEASON`) rather than the full backfill; 2016-2019 is
   still reachable with an explicit `--season` range, documented as
   unreliable rather than hidden.

**Validated the same way the called-strike model's catcher framing was:**
year-over-year Spearman correlation of `cluster_k` (arsenal size found),
2023-2025, is **0.56-0.61** — in the 0.5-0.7 range published framing metrics
report as real and sticky, not single-season noise. `arsenal_size_diff`
(disagreement with Savant specifically) is weaker but still clearly positive,
**0.27-0.30**, expected since it's a difference of two already-noisy counts.

**Reliable usable range is 2020+, not the full 2015-2026 backfill** — unlike
every other mart in this codebase; see finding #4 above. `spin_axis` also has
0% coverage in 2015 specifically (`bb check --coverage`), so 2015 contributes
zero rows on its own regardless (`build_arsenal_frame` filters it out
cleanly, no crash). A separate, real bug did crash a full-history build:
`pitch_type` can be null even after the tracked/competitive filter
(`is_tracked_pitch & is_competitive` doesn't guarantee a pitch was ever
classified), and `np.unique` can't sort `None` against strings —
`mart_pitcher_arsenal.sql` already filtered this case; the Python rebuild had
missed copying it. Fixed and pinned with a regression test on 2016 (the
earliest season that reproduces it).

**Caveat stated up front, not discovered:** `BIC_EVIDENCE_THRESHOLD` is a
per-test bar, not corrected for how many pairwise tests a pitcher-season
runs — an eight-pitch kitchen-sink starter gets 28 merge tests plus 8 split
tests, a two-pitch reliever gets one. A spot-check of a real eight-pitch-type
2025 pitcher found two additional splits that are individually
well-evidenced but more likely to include a borderline case simply because
more tests ran. No multiple-comparisons correction is applied (see the model
docstring for why a stricter one would re-introduce the under-splitting
failure mode the redesign already fixed).

**`mart_pitcher_arsenal_clusters`** (pitcher x season x re-derived cluster):
**19,855 rows, 4,212 pitcher-seasons, 2020-2026** — no registered model, no
`bb-ml`-trained artifact — a small model is fit per pitcher-season, the same
shape as the zone-profile grids (`bbetl.transforms.zones`), built by `bb-ml
arsenal`. 69.75% of pitcher-seasons land within +-1 of Savant's own count
(mean `|arsenal_size_diff|` 0.81-1.23 across every season 2020-2026, no
remaining discontinuity). API route, UI panel, and viz #12 followed in the
next session — see below.

---

## Arsenal embedding, API, UI, viz #12 (2026-08-17) — M3 model #11 closes

`models/arsenal_embed.py`. Turns `mart_pitcher_arsenal_clusters` into one
feature vector per pitcher-season, reduces it to 2D, and clusters the feature
space into named archetypes — the follow-up model #2's plan flagged as
needing the re-derived clusters as its input rather than Savant's raw
`pitch_type`. This also closes **model #11** (pitcher similarity embedding,
"who does this pitcher resemble?") since the architecture plan's viz #12
table attributes the map to model #11, not model #2 directly — discovered
mid-plan, confirmed with the user, folded into one build.

**Encoding: measured, and the intuitive pick lost.** Two encodings were
compared on the real 2020-2026 mart per this project's measure-don't-assume
standard: fixed usage-sorted slots (6 clusters x 6 features = 36 dims) vs. a
usage-weighted 2D histogram over (velocity, movement angle), 64 dims,
invariant to cluster count. The histogram's aggregate YoY neighbor-rank was
*better* (0.086 vs. slots' 0.220 — lower means "a pitcher's next season lands
near where he was"). But the named spot-check — the kind that caught all
four of model #2's real bugs, where aggregates caught none — broke hard for
the histogram: Matt Waldron (663362, a knuckleballer) landed at only the
20th percentile of nearest-other-pitcher distance, i.e. reading as an
unremarkable, replaceable arsenal, which is disqualifying for a map whose
whole point is spotting exactly that kind of outlier. Under slots, Waldron
and Tyler Rogers (643511, submariner) both land at the **99th percentile**
(correctly extreme). The histogram's better YoY number turned out to be
explained by insensitivity, not signal — it smooths over the same
arsenal-shape distinctiveness a submariner/knuckleballer needs to register,
which is also what keeps it stable when cluster count wobbles (model #2's
own `arsenal_size_diff_yoy` is a weak 0.27-0.30 for the same underlying
reason). **Slots is the default encoding.**

**Reducer bake-off (slot encoding, full mart, 4,212 pitcher-seasons):**

| reducer | trustworthiness (k=15) | YoY neighbor rank |
|---|---|---|
| t-SNE | **0.980** | **0.220** |
| UMAP | 0.968 | 0.230 |
| PCA | 0.776 | 0.216 |

t-SNE wins outright despite viz #12's name in the architecture plan ("UMAP
arsenal map") — PCA's YoY number looks close but its trustworthiness is far
worse (a linear projection cheaply preserves global drift while badly
distorting local neighborhoods, so it was never really a contender). Named
spot-check confirms t-SNE: excluding same-pitcher matches, Rogers and
Waldron both land at the 99th percentile of nearest-other-pitcher distance,
and Emmanuel Clase (661403) at the 94th, among hard-throwing cutter/slider
relievers (Louis Varland, Braydon Fisher) rather than starters. **`bb-ml
arsenal-embed` defaults to t-SNE, not UMAP** — the UI tab keeps the
colloquial "UMAP arsenal map" name from the architecture plan since it names
the general technique, not literally the `umap` package. `umap-learn` stays
an optional `embedding` extra on `bbml` (installed for the bake-off, not
promoted to a required dependency since it lost) so the comparison stays
reproducible via `bb-ml arsenal-bakeoff`.

**Archetypes** (KMeans on the feature space, never the 2D coordinates — k
chosen by silhouette over 4-12): **k=6, silhouette 0.143** (next-best k=5 at
0.142 — a genuinely flat curve, expected for a continuum like pitch
repertoires rather than tight natural clusters). Names come from each
archetype's own modal primary/secondary re-derived cluster label plus a
velocity tier (e.g. `FF/SL · 90-94`), disambiguated with a `-2`/`-3` suffix
the same way `models.arsenal._label_clusters` handles a repeated label —
real data hits two different feature-space archetypes sharing the same
coarse name at k=6.

**New marts**, both registered in `LAKE_TABLES`, inheriting model #2's 2020+
floor: **`mart_arsenal_embedding`** (4,212 rows, one per pitcher-season: x/y,
archetype, and the model #2 columns riding along for the map's colour-by
options) and **`mart_arsenal_neighbors`** (42,120 rows, top-10 nearest
pitcher-seasons per pitcher-season by pre-embedding FEATURE-space distance,
not 2D distance — the 2D map is for eyeballing, this table is the actual
"who does this resemble" answer). Built by `bb-ml arsenal-embed`.

**API**: `apps/api/src/bbapi/routers/arsenal.py` — `GET /arsenal/embedding`
(every point for the map), `GET /arsenal/{mlbam_id}` (that pitcher's
re-derived clusters), `GET /arsenal/{mlbam_id}/similar` (nearest neighbors,
joined to `dim_player` for names). `/embedding` is declared before
`/{mlbam_id}` — FastAPI matches route registration order, and the reverse
would 422 trying to parse `"embedding"` as an int; pinned with a regression
test (`test_embedding_route_is_matched_before_the_int_path_param`). 27 API
operations total now (was 24).

**UI**: `ArsenalClusterPanel` on the player page, right after the existing
Savant-labelled `ArsenalTable`, so the disagreement is legible side by side.
New standalone **Arsenal map tab** (`ArsenalMapPage` + `ArsenalMap`) — every
pitcher-season as a point, coloured by a selectable `ARSENAL_METRICS` key on
the same validated diverging ramp every other chart uses (archetypes get a
convex-hull outline + direct centroid label rather than their own hue, so
the all-pairs CVD gate — colour encodes at most three hues — is untouched).
Pan/zoom (`lib/viewport.ts`, new) is a single `<g transform>` over the points
group with a counter-scaled marker radius, not a per-point recompute at
~4,200 points. The global season filter narrows the cloud; unset (default)
shows every pitcher-season at once.

**Verification**: 8 new API tests (`TestArsenal`, 1.5s — read-only against
already-built marts, no retraining), 20 new frontend tests
(`viewport.test.ts` + `hull.test.ts`), `tsc --noEmit`/`oxlint` clean,
`ruff check` clean. **Playwright, light + dark**: the Arsenal map tab
(hulls + centroid labels render, click populates the detail pane, wheel-zoom
+ drag-pan actually move the view via the single `<g transform>`, the season
filter narrows the cloud), and the player-page panel for **Gerrit Cole
(543037)** — matches the on-disk spot-check, 9 re-derived clusters vs. 4
Savant types in 2024, `arsenal_size_diff` +5. One real layout bug caught and
fixed by the screenshot, not by the type-checker: the cluster table's 9
columns overflowed the narrower detail-pane card — wrapped in its own
`overflow-x: auto` container in `ArsenalClusterPanel`. Also surfaced, but
pre-existing and out of this session's scope: `ArsenalTable.tsx`'s
`key={r.pitch_type}` collides across seasons when no season filter is set
(a React duplicate-key warning, not a crash) — the Savant-arsenal table has
always shown a full career unfiltered; noted here so it isn't mistaken for
something this session introduced.

Note: whichever session picks up the `ArsenalTable.tsx` key-collision fix
found above should also remove this note once it's done.

---

## Viz #19 (swing path) + viz #8 (spray chart) (2026-08-17)

Implemented from `~/.claude/plans/plan-the-implementation-of-recursive-hinton.md`.
Built in the plan's suggested order: #19 first (small, all data already
existed), #8 second (new plumbing end to end).

**Viz #19 — swing path.** One new route, `GET /swing/{id}/pitches`
(`routers/swing.py`), calling `bbml.features.swing.load_swing_frame()`
directly and filtering to the batter — no new mart, so it structurally cannot
drift from `mart_batter_swing`'s own predicate. Arrow IPC (per-swing grain,
same rule as `/pitches`). Frontend: `SwingPathScatter.tsx`, attack angle (y)
vs. pitch descent angle (x), one point per tracked swing, hollow marker for a
whiff vs. filled for contact (colour still carries pitch family, shape still
carries pitch type — `familyColor`/`pitchShape` reused unchanged, the CVD
gate governs every scatter regardless of what else it's encoding), plus a
swing-length histogram using a new `lib/histogram.ts` (+ 5 tests). Templated
directly on `ArsenalMap.tsx`'s pan/zoom pattern.

**Viz #8 — spray chart.** The larger piece, four layers:

1. **Hit-coordinate transform** (`bbetl.transforms.statcast.enrich`):
   `hc_x`/`hc_y` → `x_ft`/`y_ft`/`spray_angle_deg`/`hit_distance_derived_ft`.
   The plan's suggested constants were the community-published ones (origin
   125.42/198.27, scale 2.495 px/ft) — **measured against this project's own
   data instead of trusted**, per the project's own "measure, don't assume"
   standard. A least-squares fit of `k * hypot(hc_x-x0, hc_y-y0)` against
   Savant's own `hit_distance_sc` over 158,098 real 2023-2024 balls in play
   confirmed the published ORIGIN (fit converged to 125.91/199.54, within a
   foot) but not the scale — 2.339, not 2.495. Shipped the fitted constants,
   not the published ones. Even at the best-fit scale, MAE against
   `hit_distance_sc` is ~28ft, r=0.886 — this is `hc_x`/`hc_y`'s inherent
   precision (a charted fielding location, not a trajectory endpoint), not a
   transform bug: no systematic offset (fitted origin within a foot of
   published) and no mirrored sign (RHB mean `x_ft` -3.8ft / LHB +9.6ft on
   2024 fly balls + line drives — pull side correctly opposite by handedness).
   Required a full `bb build pitches` lake rebuild (9,202,082 pitches, row
   counts unchanged, ~2min12s) since it's a derived column computed at
   ingest-transform time, not query time.
2. **Shared smoothing core**: `zones.py`'s `_smooth_ratio` extracted into
   `bbetl.transforms.smoothing.kernel_regress_2d`, parameterized by
   grid/bandwidth so `zones.py` and the new `transforms/spray.py` share the
   numeric core without sharing constants. Regression-tested
   (`test_smoothing.py`) to reproduce `zones.py`'s pre-refactor output
   exactly on `zones.py`'s own grid.
3. **`mart_batter_spray`** (new): batter x season, one smoothed
   xwOBA-on-contact grid per row (60x60, x∈[-350,350]ft, y∈[0,450]ft,
   bandwidth 18ft, `MIN_RELIABLE_N=12` — all measured/documented in
   `transforms/spray.py`, not copied from the strike-zone grid's constants).
   4,518 rows, min 100 batted balls/batter-season, full 2015-2026 range (no
   2020+ floor the way the arsenal marts need — batted-ball landing spot
   doesn't depend on Hawk-Eye-era bat-tracking precision). `bb-ml spray-mart`
   CLI command; registered in `LAKE_TABLES`.
4. **API + frontend**: `/spray/{id}/battedballs` (Arrow), `/spray/{id}/contour`
   (JSON, shaped like `/zones/{id}`'s response), `/spray/extent`. 31 API
   operations total now (was 27). `apps/web/src/data/parks.ts`: 30 MLB park
   wall polygons, Catmull-Rom-interpolated through 5 publicly documented
   distance markers per park (LF line/LF-CF alley/CF/CF-RF alley/RF line) —
   more detail than a 3-point approximation per the user's explicit
   direction, but these are commonly-published scoreboard figures, not
   surveyed fence data, and worth re-verifying if a specific park's shape
   ever looks wrong. `SprayChart.tsx`: pan/zoom over feet-from-plate
   coordinates (SVG y flipped so CF renders up), the wall outline, batted-ball
   points coloured by `divergingColor` on the new `ZONE_METRICS.spray`
   entry (mid/halfRange measured from `mart_batter_spray`'s own reliable
   cells: p1/p99 = 0.10/0.44 — the full grid including sparse near-empty
   deep-field cells runs to ~2.0 and would blow out the ramp), and the
   smoothed contour traced with `lib/contour.ts`'s existing `marchingSquares`
   at the metric's `mid` level.

**Health endpoint gap found and fixed in passing**: `/health`'s `tables`
dict is a separate hardcoded tuple in `main.py`, not derived from
`bbetl.warehouse.LAKE_TABLES` — a new mart is invisible to `_needs()`-gated
tests and to anyone checking pipeline state via `/health` until it's added to
*both* places. `mart_batter_spray` added to `main.py`'s list; worth
remembering for the next new mart too.

**Verification**: targeted test suites only (the user asked to stop a
full-repo `pytest -q` run mid-session and only run what this session's files
touch) — `apps/api/tests/test_api.py`, `packages/bbetl/tests/test_transforms.py`,
`test_zones.py`, `test_smoothing.py` (new), `packages/bbml/tests/test_marts.py`,
`test_spray_mart.py` (new, 4 tests against a fixture lake under `tmp_path`) —
all pass. Frontend: 48 tests (was 43), `tsc --noEmit`/`oxlint` clean,
`ruff check` clean, `npm run build` succeeds. Manually smoke-tested the three
new API routes against the real running dev server (`/health`,
`/swing/650333/pitches`, `/spray/650333/battedballs`, `/spray/650333/contour`
for Luis Arraez, 2024 — all 200, correct row/column shapes). **No Playwright
pass** — no browser tool was available in this session, so viz #8/#19 have
NOT had the visual check that caught real bugs (Arrow zstd, the coordinate-
mirror bug) on every other visualization in this app. Do that before trusting
the chart's on-screen correctness, especially the park outline orientation
and the contour rendering — this is exactly the failure class that check
exists to catch and it has not been run yet for this feature.

---

## Committed (as of 2026-08-17, arsenal-embedding session)

`git log` (newest first) at the time of writing: the API routes + player-page
panels for models #4/#5, `mart_batter_swing`, the `dim_official`
warehouse-registration fix, and the null-group-column fix in
`framing_runs`/`umpire_zone_rate` (`6ff1462`), the called-strike model +
`dim_official` (`6fc00fe`), the swing-path model + `vaa_deg`/`haa_deg`
(`a6a987e`), M3 model #3 (`f979595`), 3D pitch trajectory work
(`4b77e20`/`d3d9c2f`/`782a5b5`), the zstd Arrow fix (`cbf07a8`), an earlier
STATUS.md update (`4694c91`), the schema-overrides fix (`20d776b`), `make
train` (`3455af5`), M2 (`00d8ad9`), the first bbml package (`67e2e16`), and M1
(`6e0d555`). Always check `git log` rather than trusting this list to stay
current — it is a snapshot, not maintained after the fact.

---

## Model numbers — next-pitch, early runs

Three runs total; numbers moved a lot between them, which is itself
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
existed in 2015-16). Run 3 (see "Full lake rebuild + retrain" above) is the
resolution once the 2018-2024 gap filled in.

Personalization is answered in full in conversation history: one global model,
`pitcher` deliberately not a feature, personalization via expanding-window
per-pitch-type usage priors. Per-pitcher models measured ~20% worse on the
single-season run, and the multi-season run's baseline collapse only reinforces
that conclusion. See `next_pitch.py` module docstring for the full writeup —
don't re-derive it.
