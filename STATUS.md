# Development bookmark

**Paused:** 2026-08-16 · **M1 and M2 complete and visually verified end-to-end.
Four M3 items pulled forward and finished: the 3D pitch trajectory (viz #6),
Stuff+ / Location+ / Pitching+ (model #3), the swing-path model (model #4), and
the called-strike model (model #5) — models #4/#5 have a full API + UI
surface, and the two dedicated visualizations model #5 unlocked, viz #20
(catcher framing map) and viz #13 (umpire zone map), are now built too.**
The full 2015-2026 backfill has landed (9,202,082 pitches, contiguous) and
every model has been retrained on it — see "Full lake rebuild + retrain" below
for the numbers, several of which overturned earlier guesses in this file.
Officials data (umpire per game) is fully ingested (11,154 games) and
materialized into the lake as `dim_official` — see "Called-strike model",
"API + UI for models #4/#5", and "Viz #20/#13" below.

Read this first in a new session, then `README.md` for how the thing works and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M3 backlog.

---

## Where we are

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite. **Now includes `transforms/officials.py` (`dim_official`, home-plate umpire per game — see below).** Complete. |
| `packages/bbml` | Feature builder (batch+live, parity-tested), datasets/splits, `UsageRateBaseline`, `NextPitchModel` (pitch type), `LocationModel` (26-class grid), `PersonalizedBlend`, `RunValue` + `PitchQualityModel` (Stuff+/Location+/Pitching+, M3 model #3), `SwingPathModel` (whiff + contact heads, matched-counterfactual `plane_value`, M3 model #4), `CalledStrikeModel` (binary, `framing_runs` + `umpire_zone_rate`, M3 model #5), `registry.py` (versioned artifacts + optional MLflow), `marts.py` (`mart_batter_swing` / `mart_catcher_framing` / `mart_umpire_zone`, **plus the catcher/umpire spatial grids feeding `mart_zone_profile` — see "Viz #20/#13" below**), `bb-ml` CLI. Now depends on `bbetl` (reuses its zone-smoothing machinery) — added to `pyproject.toml`. |
| `apps/api` | `/predict/next-pitch` (what-if), `/games/{game_pk}/replay`, `/players/{id}/games`, `/pitches/trajectory`, `/stuff/{id}` + `/stuff` leaderboard, `/swing/{id}` + `/swing` leaderboard, `/framing/catchers/{id}` + `/framing/catchers` leaderboard, `/framing/umpires` leaderboard, **`/zones/{id}` now also serves `role=catcher`/`role=umpire`** (no new route — the existing generic zone endpoint just grew two more valid roles/metrics). 24 routes total (`app.openapi()` operation count), JSON + Arrow IPC. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table, at-bat replay strip (viz #9), 3D pitch trajectory (viz #6), pitch quality panel (model #3), swing-plane panel for batters (model #4), catcher-framing panel for catchers (model #5) **with an embedded zone map (viz #20)**, **a new standalone Umpires tab (viz #13) — umpire leaderboard + zone map with a marching-squares 50% contour overlay**. Visually verified, light + dark. |

**Verification status:** 206 backend Python tests (bbcore/bbetl/bbml/api,
including 25 called-strike tests, 3 `dim_official` transform tests, 10
swing/framing API contract tests, and **3 new `test_marts.py` tests for the
viz #20/#13 grid-building helper's null-id-column filter**) + 23 frontend
tests (**5 new for the marching-squares contour tracer**), `tsc --noEmit`,
`oxlint`, `ruff check`, `bb check` (data quality — all
error-level checks pass), `bb-ml status` all pass/registered. All eight
models trained on the full 9,202,082-pitch 2015-2026 lake and saved to
`data/models/{next_pitch,location,stuff_plus,location_plus,pitching_plus,swing_whiff,swing_contact,called_strike}/`.
**The rendered UI has been visually verified** (Playwright/Chromium
screenshots, light + dark) for the player page including the two new
model #4/#5 panels — `Luis Arraez` (batter, swing plane) and `Jose Trevino`
(catcher, swing plane + framing runs). Numbers on screen were cross-checked
against the training-run leaderboard and matched exactly. **Viz #20/#13 got
the same treatment** — Trevino's embedded framing map and the new Umpires
tab's zone map + contour, light and dark — see "Viz #20/#13" below.

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
map) and viz #13 (umpire zone map) need. Neither full visualization is built
yet, but both marts are now API-served — see "API + UI for models #4/#5"
below.

Not yet done at the time this model was built: no API route, no UI panel, no
viz #20/#13. Both the route and a player-page panel now exist — see below;
viz #20/#13 (dedicated catcher-map / umpire-map visualizations, as opposed to
a per-player number) are still not built. `bb-ml called-strike` trains,
registers, and prints a framing-runs leaderboard; `bb-ml called-strike-mart`
rebuilds the two marts standalone.

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

Not yet done at the time this section was written: viz #20/#13 — see below,
now built.

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
errors beyond the pre-existing FA/EP/KN duplicate-React-key warning noted in
the previous section (still out of scope — not touched by any file this
session either).

---

## Current local data

- **9,202,082 pitches**, seasons 2015-2026, contiguous — the full backfill
  finally landed (see above). No more season gaps.
- Marts: `mart_pitcher_stuff` 40,539 rows (pitcher x season x pitch_type + an
  `ALL` rollup); `mart_batter_swing` 1,611 rows (batter x season);
  `mart_catcher_framing` 1,048 rows (catcher x season); `mart_umpire_zone` 230
  rows (umpire x season — 238 minus 8 phantom null-umpire rows, one per
  season 2015-2022, dropped by the bug fix in "API + UI for models #4/#5"
  above). `mart_zone_profile` now also holds **1,048 catcher-season grids and
  362 umpire-season grids** (viz #20/#13, see above) alongside its existing
  batter/pitcher grids. `mart_pitcher_arsenal` / the batter/pitcher share of
  `mart_zone_profile` not re-counted this session — re-run `bb check
  --coverage` before trusting the old figures below.
- Officials: **11,154 games** with a home-plate umpire, now materialized as
  `dim_official` in the lake (`bb ingest officials` then `bb build
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

`git log` (newest first): the API routes + player-page panels for models
#4/#5, `mart_batter_swing`, the `dim_official` warehouse-registration fix,
and the null-group-column fix in `framing_runs`/`umpire_zone_rate`
(`6ff1462`), the called-strike model + `dim_official` (`6fc00fe`), the
swing-path model + `vaa_deg`/`haa_deg` (`a6a987e`), M3 model #3 (`f979595`),
3D pitch trajectory work (`4b77e20`/`d3d9c2f`/`782a5b5`), the zstd Arrow fix
(`cbf07a8`), an earlier STATUS.md update (`4694c91`), the schema-overrides fix
(`20d776b`), `make train` (`3455af5`), M2 (`00d8ad9`), the first bbml package
(`67e2e16`), and M1 (`6e0d555`). This session's work — viz #20/#13
(`bbml.marts`'s new grid builders, the two `/zones` role/metric additions,
`lib/contour.ts`, the `UmpiresPage` tab, the `FramingPanel` map, and this
STATUS.md update) — lands in the commit right after this one; check
`git log` rather than trusting this list to stay current.

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

**M3, in progress:** models #3 (Stuff+/Location+/Pitching+), #4 (swing-path),
and #5 (called-strike / framing) are all done, #4/#5 have a full API + UI
surface (see "API + UI for models #4/#5" above), and the two dedicated
visualizations model #5 unlocked — viz #20 (catcher framing map) and viz #13
(umpire zone map) — are now built too (see "Viz #20/#13" above). That closes
out everything model #5's plan called for. The previously-queued
zone-expansion umpire rerun is superseded: `dim_official` now exists and the
called-strike model's own validation (YoY stability, named-catcher check) is
the more rigorous version of that question — no need to separately rerun the
old artifact's analysis. Model #5's plan is still in the assistant's project
memory (`baseball-model5-called-strike-plan`) if the detail behind a design
choice is needed; the plan itself is now fully executed, not just scoped.
Still not started: live game-feed mode, `PostgresWarehouse`, model #2
(arsenal re-classification), model #6 (swing decision, needs #5's P(strike)
as RV(take) — now unblocked) and model #15 (ABS counterfactual, also now
unblocked), viz 7-8, 10-12, 14, 15-19, Retrosheet backfill.

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
  Stats API boxscore (`bb ingest officials`), one request per game, landed as
  raw JSON only — `bb build officials` is the separate step that turns that
  into the queryable `dim_official` lake table. The raw JSON sat unbuilt for a
  full session before model #5 needed it; if `UMPIRE_COLUMN` comes back all
  null, this is the step that was skipped.
- **A new lake table isn't queryable until it's in `bbetl.warehouse.LAKE_TABLES`.**
  `bb build officials` writes `dim_official`'s Parquet correctly, but writing
  the file and registering it with the warehouse are two different steps
  (same for every `dim_*`/`mart_*` table) — this one was missed for a full
  session, so `/framing/umpires` 503'd with "table not built" even though the
  file existed on disk and the build command had reported success. If a new
  table 503s despite `bb build <whatever>` succeeding, check `LAKE_TABLES`
  before checking anything else.
- **`dim_official` only covers 2023+.** Every take from an earlier game has a
  null umpire id. Any `group_by(UMPIRE_COLUMN)` (or any column that can be
  null for a structural reason, not just missing data) MUST filter the null
  out explicitly — polars groups null as its own bucket rather than dropping
  it, and here that bucket silently absorbed 50,000+ pre-2023 pitches per
  season into one fake "umpire". `framing_runs`/`umpire_zone_rate`
  (`models/called_strike.py`) do this now; it is not automatic anywhere else
  a null-capable column gets grouped.
- **`bbml` now depends on `bbetl`** (`pyproject.toml`), added when viz #20/#13's
  grid builders needed `bbetl.transforms.zones`'s smoothing machinery
  (`build_grid`, `GRID_N`, `MetricSpec`). It worked without the declared
  dependency before this because the workspace root installs every package
  into one shared venv regardless — don't rely on that again; declare the
  dependency explicitly if a package's code actually imports across the
  boundary, the way `bbapi` already declares `bbml`.
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
