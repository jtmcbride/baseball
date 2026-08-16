# Baseball Analytics Platform

Interactive Statcast analytics: pitch-level visualization, smoothed hot/cold zone
maps, and (from M2) a next-pitch ML predictor.

Local-first, but storage and config sit behind interfaces so a move to hosted
Postgres is a config change rather than a migration.

## Quick start

```bash
make setup                                  # uv sync + npm install
cp .env.example .env                        # adjust if you want

uv run bb ingest statcast \
  --start 2025-04-01 --end 2025-06-30       # a slice, to try it out
uv run bb ingest dims                       # games/teams/players
uv run bb ingest crosswalk                  # MLBAM <-> FG/BBRef/Retrosheet
uv run bb build pitches && uv run bb build marts
uv run bb check                             # data quality gate

make dev                                    # API :8000 + UI :5173
```

Full backfill (2015→today, ~7.7M pitches) is `make backfill`. Measured sustained
rate is **6.4s per game-day**, so expect **~3.5 hours** for ~2,000 game days —
almost all of it Savant's server-side query time, not our rate limit. It is
resumable: interrupt it and re-run, and it picks up where it stopped.

## Architecture

```
packages/bbcore   config, Warehouse adapter (DuckDB now, Postgres later)
packages/bbetl    source clients, ingest jobs, transforms, quality checks
packages/bbml     features / training / serving          (M2)
apps/api          FastAPI — JSON for small payloads, Arrow IPC for pitch data
apps/web          React + TypeScript, custom SVG charts
sql/marts         analytical marts
```

Data moves through three layers, each rebuildable from the one above:

| Layer | Path | Why |
|---|---|---|
| **raw** | `data/raw/` | Immutable, as-fetched. Makes the 3-hour crawl a one-time cost — every later schema change is a local reprocess. |
| **lake** | `data/lake/` | Typed, deduped, enriched Parquet. The analytical source of truth. |
| **warehouse** | `data/db/baseball.duckdb` | Views over the lake plus materialized marts. |

## Things worth knowing

**Savant truncates silently.** `statcast_search/csv` caps at 25,000 rows and does
not error, set a header, or emit a marker — it returns HTTP 200 with the data cut
short mid-day. Verified: a 14-day request comes back with exactly 24,999 rows
ending partway through day 8. The ingester partitions by single game date
(~4,400 pitches, ~5.6x headroom) and trips a guard that re-fetches by team if any
partition lands near the cap.

**Savant also revises published data** — pitch classifications get corrected and
xwOBA recomputed after the fact. `bb ingest refresh` re-pulls a trailing window;
append-only ingest goes stale in place, invisibly.

**Statcast's `umpire` column is empty** in every season. Home-plate umpires come
from the Stats API boxscore instead (`bb ingest officials`), which costs one
request per game and is only needed for the umpire/framing models.

**Arm-side normalization is load-bearing.** `hb_arm_in` flips horizontal break so
positive is always arm-side. Without it, lefties and righties mirror each other
and every cross-handedness comparison silently inverts. A quality check asserts
sinkers show positive arm-side run for *both* hands.

**Hot/cold zones are kernel-smoothed, not binned.** Raw per-cell averages are
mostly noise at the corners. Every grid ships a reliability mask (effective
sample size per cell) and the UI fades cells below the floor rather than
presenting an interpolated estimate at full saturation.

**Bat-tracking columns are 2024+** (`bat_speed`, `swing_length`) **and swing-path
columns 2025+**. They are nullable across the whole 2015+ lake; `bb check
--coverage` reports per-season availability so this is discovered from the data
rather than assumed.

## Data sources & terms

| Source | Use | Notes |
|---|---|---|
| Baseball Savant | pitch-level Statcast | Public, unofficial for bulk use. Rate-limited politely via `BB_SAVANT_RPS`. |
| MLB Stats API | schedule, players, umpires, live feed | Official, free, no scraping. |
| FanGraphs | advanced stats, projections | Scraped via pybaseball. Prohibits scraping for redistribution. |
| Baseball Reference | bWAR, historical | Same. Aggressive rate limits. |
| Chadwick Bureau | ID crosswalk | The glue between all of the above. |

FanGraphs and Baseball Reference prohibit scraping for redistribution. This is
fine for local/personal use. If this is ever hosted publicly, serve **derived
aggregates** computed from their data rather than their raw tables, and keep the
crawler slow.

## Testing

```bash
uv run pytest                    # transforms, ingest machinery, zone smoothing, API contracts
uv run bb check --coverage       # data quality gate (non-zero exit on ERROR)
cd apps/web && npx tsc --noEmit
```

The quality suite exists because the failure mode here is not a crash — it is a
backfill that completes, looks healthy, and is quietly wrong.
