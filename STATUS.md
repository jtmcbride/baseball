# Development bookmark

**Paused:** 2026-08-15 · **Milestone 1 complete and verified.** M2 not started.

Read this first in a new session, then `README.md` for how the thing works and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M2/M3 backlog.

---

## Where we are

M1 was "ingestion pipeline + pitch explorer UI". That is done end to end:
raw Savant CSV → Parquet lake → DuckDB marts → FastAPI → React charts.

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite. Complete. |
| `packages/bbml` | **Scaffold only** — `__init__.py` with the train/serve parity note. No code. |
| `apps/api` | 8 routes, JSON + Arrow IPC. Complete for M1. |
| `apps/web` | Filter bar, player search, 4 charts + arsenal table. **Never visually inspected** — see gap below. |

**Verification status:** 71 Python tests, 14 frontend tests, `tsc --noEmit`,
`ruff check`, and `bb check` (data quality) all pass. Arsenal/movement numbers
reconciled by hand against public Savant values.

---

## The one real gap

**Nobody has looked at the rendered UI.** The build compiles, types check, unit
tests pass, and every API payload the page consumes is verified correct — but a
browser screenshot was declined at the end of the session, so **layout
collisions, chart geometry, overflow, and dark mode are unchecked**. Treat the
frontend as "plausibly correct, visually unverified."

To check:

```bash
make dev            # API :8000 + UI :5173
open http://localhost:5173      # search "Skubal" — good dense arsenal
```

Both servers were stopped at pause; nothing is running.

---

## Current local data

Not the full backfill — a working slice.

- **355,305 pitches**, 2025-04-01 → 2025-08-05, 1,214 games, regular season only.
- `data/raw` 76 MB · `data/lake` 130 MB · `data/db` 2 MB.
- Marts: `mart_pitcher_arsenal` 3,235 rows; `mart_zone_profile` 4,265 grids
  (2,185 pitcher / 2,080 batter).

**Known wart:** `data/raw/statcast/season=2025/2025-08-05.csv.gz` exists but has
no manifest row — it was landed directly by a throughput-profiling script rather
than through `ingest_range`. Harmless (the build reads raw files, so it is in
`fact_pitch`), but `bb ingest statcast` would re-fetch that day. Fix by
re-ingesting the date or leaving it; nothing depends on it.

Full 2015→present backfill has **not** been run. Measured cost: 6.4s per game-day
sustained → ~3.5 hours for ~2,000 days. Resumable, so it can be started and
interrupted freely.

---

## Not committed

`git init` was run; **there are zero commits.** All 9 top-level paths are
untracked. `.gitignore` correctly excludes `data/`, `.venv/`, `node_modules/`.
Nothing was committed because it was never requested — worth doing before any
significant further work.

---

## Resume paths

**Continue to M2 (next-pitch model)** — the approved next milestone:

1. `packages/bbml/features/` — the feature builder. **One function, two callers**
   (batch over Parquet, live over an MLB game-feed state object). Write the golden
   parity test *before* the live path exists; it is the contract that keeps the
   two from drifting.
2. `packages/bbml/datasets/` — time-based splits (train ≤2023 / val 2024 /
   test 2025+) with leakage assertions as code: no full-season aggregates, no
   post-pitch info, group-by-pitcher split for unseen-pitcher generalization.
3. Baseline **first**: pitcher's season usage rate bucketed by count. Publish its
   log-loss, then assert in a test that the trained model beats it. A model that
   loses to this baseline is noise, and it is easy to ship one without noticing.
4. LightGBM multiclass — arsenal-masked pitch type + 26-class location grid.
5. `/predict/next-pitch` route + at-bat replay UI (viz #9 in the plan).

Note M2 needs more history than the current slice — run the backfill first, or at
minimum several full seasons.

**Or finish M1 properly:** visually verify the UI, then run the full backfill.

---

## Decisions already made — don't relitigate

- **Colour encodes pitch *family* (3 hues), shape encodes pitch type.** Not
  cosmetic: the movement plot is a scatter, so it falls under the all-pairs CVD
  gate, which the validated palette clears with three slots. Nine pitch types
  cannot take nine hues. Centroid labels + the arsenal table supply the required
  contrast relief.
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

## Gotchas that cost time to rediscover

- **Savant truncates silently at 25,000 rows** — HTTP 200, no marker, data just
  stops mid-day. Guarded, but never widen the date partition without re-checking.
- **Savant revises published data** after the fact. `bb ingest refresh` re-pulls a
  trailing window; append-only ingest goes stale invisibly.
- **Statcast's `umpire` column is empty in every season.** Umpires come from the
  Stats API boxscore (`bb ingest officials`), one request per game — needed for
  the framing/called-strike models, skipped for M1.
- **DuckDB persists a view's resolved schema.** Rebuilding the lake with a changed
  column set leaves stale views that fail confusingly. `build pitches` now
  re-registers automatically; keep it that way.
- **Bat tracking is 2024+, swing path 2025+.** Nullable across the whole lake.
  `bb check --coverage` reports per-season availability — read it rather than
  assuming.

## Quick sanity check after `git pull` / fresh session

```bash
uv sync --python 3.13 && cd apps/web && npm install && cd ../..
uv run pytest && uv run bb check
uv run bb status          # ingest manifest
```
