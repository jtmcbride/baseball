# Development bookmark

**Paused:** 2026-08-16 · **Milestone 1 complete. M2 (next-pitch predictor) built
and registered end-to-end** — feature builder, both model heads, API routes,
replay UI. Not yet run against the full backfill.

Read this first in a new session, then `README.md` for how the thing works and
`~/.claude/plans/i-m-building-an-interactive-zany-ember.md` for the full
architecture plan and the M3 backlog.

---

## Where we are

| Layer | State |
|---|---|
| `packages/bbcore` | Config + `Warehouse` adapter (DuckDB). Postgres impl deliberately absent — M3. |
| `packages/bbetl` | Savant / Stats API / Chadwick clients, transforms, marts, quality suite. Complete. |
| `packages/bbml` | Feature builder (batch+live, parity-tested), datasets/splits, `UsageRateBaseline`, `NextPitchModel` (pitch type), `LocationModel` (26-class grid), `PersonalizedBlend`, `registry.py` (versioned artifacts + optional MLflow), `bb-ml` CLI. |
| `apps/api` | `/predict/next-pitch` (what-if), `/games/{game_pk}/replay`, `/players/{id}/games` added. 11 routes total, JSON + Arrow IPC. |
| `apps/web` | Filter bar, player search, 4 charts, arsenal table, **at-bat replay strip** (viz #9). **Never visually inspected** — see gap below. |

**Verification status:** 90 backend Python tests (bbcore/bbetl/bbml) + 19 API
tests + 14 frontend tests, `tsc --noEmit`, `ruff check`, `bb check` (data
quality), `bb-ml status` all pass/registered. Both models trained on the
355k-pitch working slice and saved to `data/models/{next_pitch,location}/`.

---

## The one real gap

**Nobody has looked at the rendered UI**, including the new replay strip. Browser
screenshot tooling was declined once already this project (see memory) — ask
before installing/using it again. To check by hand:

```bash
make dev            # API :8000 + UI :5173
open http://localhost:5173      # search "Skubal" — good dense arsenal
```

Both servers were left running at pause (`localhost:8000`, `localhost:5173`) —
may already be up; check before starting new ones.

---

## Current local data

Still not the full backfill — the same working slice as M1.

- **355,305 pitches**, 2025-04-01 → 2025-08-05, 1,214 games, regular season only.
- `data/raw` 76 MB · `data/lake` 130 MB · `data/db` 2 MB.
- Marts: `mart_pitcher_arsenal` 3,235 rows; `mart_zone_profile` 4,265 grids.

Full 2015→present backfill has **not** been run. Measured cost: 6.4s per game-day
sustained → ~3.5 hours for ~2,000 days. Resumable — safe to start and interrupt.
**Both ML models should be retrained once it lands** — single-season numbers
below will move, especially the location model (13.5% top-1 on 26 classes, a
much harder problem than pitch type and the one most likely to improve with more
seasons of history per pitcher).

---

## Committed

`git log`: two commits — M1 (`6e0d555`) and the M2 model package
(`67e2e16`). The M2 API/frontend work from this session (predict router, replay
UI, registry) is staged for the next commit — check `git status` before assuming
it landed.

---

## Model numbers (single-season slice, train Apr1-Jun3 / val Jun4-17 / test Jun18-Aug5)

| model | log_loss | top1 | ece |
|---|---|---|---|
| baseline (per-pitcher usage) | 1.5942 | - | - |
| next-pitch (global LightGBM) | 1.2736 | 0.457 | 0.0193 |
| next-pitch + personalized blend | 1.2840 | 0.460 | 0.0061 |
| location (26-class grid) | 2.9572 | 0.135 | - |

Personalization is answered in full in conversation history: one global model,
`pitcher` deliberately not a feature, personalization via expanding-window
per-pitch-type usage priors. Per-pitcher models measured ~20% worse. See
`next_pitch.py` module docstring for the full writeup — don't re-derive it.

---

## Resume paths

**Finish M2 properly:**
1. Visually verify the replay UI (the gap above).
2. Run the full backfill, retrain both models (`bb-ml next-pitch`, `bb-ml
   location`), see if the location model's 13.5% top-1 improves with more data.
3. Consider a location arsenal-style prior (where a pitcher tends to miss) as a
   feature — the location model currently uses the same feature set as pitch
   type, which wasn't built with location-specific signal in mind.

**Or move to M3:** live game-feed mode; `PostgresWarehouse`; models 2/3/5 (arsenal
re-classification, Stuff+/Location+, called-strike probability); viz 6-8, 10-20;
Retrosheet backfill.

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

## Quick sanity check after `git pull` / fresh session

```bash
uv sync --python 3.13 && cd apps/web && npm install && cd ../..
uv run pytest && uv run bb check
uv run bb status          # ingest manifest
uv run bb-ml status       # registered model versions
```
