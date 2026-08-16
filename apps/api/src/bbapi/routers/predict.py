"""Next-pitch predictions: a live what-if endpoint and a historical replay.

Both read whichever model version `bb-ml`'s registry currently points LATEST
at — the API never trains or picks a version itself, so a new model becomes
live by re-running training, not by redeploying this service.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bbapi.deps import require_table, warehouse
from bbcore.config import get_settings
from bbml.features.build import build_features, location_class_expr
from bbml.features.context import IncrementalState, PendingPitch, load_batch_frame
from bbml.features.schema import LOC_GRID_N
from bbml.models.location import LocationModel
from bbml.models.next_pitch import NextPitchModel
from bbml.registry import latest_dir

router = APIRouter(tags=["predict"])


@lru_cache(maxsize=1)
def _next_pitch_model() -> NextPitchModel | None:
    d = latest_dir("next_pitch", settings=get_settings())
    return NextPitchModel.load(d) if d else None


@lru_cache(maxsize=1)
def _location_model() -> LocationModel | None:
    d = latest_dir("location", settings=get_settings())
    return LocationModel.load(d) if d else None


def _require_models() -> tuple[NextPitchModel, LocationModel | None]:
    model = _next_pitch_model()
    if model is None:
        raise HTTPException(
            503,
            "No next-pitch model is registered yet. Run `bb-ml next-pitch` to train one.",
        )
    return model, _location_model()


def _proba_by_pitch_type(model: NextPitchModel, proba_row) -> list[dict[str, Any]]:
    ranked = sorted(zip(model.classes, proba_row.tolist(), strict=True), key=lambda p: -p[1])
    return [{"pitch_type": pt, "probability": round(p, 4)} for pt, p in ranked if p >= 0.005]


def _proba_by_location(proba_row) -> list[dict[str, Any]]:
    return [
        {"class": i, "row": i // LOC_GRID_N, "col": i % LOC_GRID_N, "probability": round(p, 4)}
        for i, p in enumerate(proba_row.tolist())
        if p >= 0.01
    ]


class GameState(BaseModel):
    """The pre-pitch situation to predict into. Mirrors `PendingPitch`."""

    pitcher_id: int
    batter_id: int
    season: int
    balls: int = 0
    strikes: int = 0
    outs_when_up: int = 0
    inning: int = 1
    base_state: int = 0
    bat_score: int = 0
    fld_score: int = 0
    stand: str
    p_throws: str
    home_team: str
    inning_topbot: str = "Top"
    n_thruorder_pitcher: int | None = None
    pitcher_days_since_prev_game: int | None = None


@router.post("/predict/next-pitch")
def predict_next_pitch(state: GameState) -> dict[str, Any]:
    """What-if prediction: seed the pitcher's season history and predict one
    synthetic next pitch in the given situation."""
    require_table("fact_pitch")
    model, loc_model = _require_models()

    incremental = IncrementalState.seeded(state.pitcher_id, state.season)
    pending = PendingPitch(
        game_pk=0,
        at_bat_number=0,
        pitch_number=1,
        game_date=None,
        season=state.season,
        pitcher=state.pitcher_id,
        batter=state.batter_id,
        balls=state.balls,
        strikes=state.strikes,
        outs_when_up=state.outs_when_up,
        inning=state.inning,
        base_state=state.base_state,
        bat_score=state.bat_score,
        fld_score=state.fld_score,
        stand=state.stand,
        p_throws=state.p_throws,
        home_team=state.home_team,
        inning_topbot=state.inning_topbot,
        n_thruorder_pitcher=state.n_thruorder_pitcher,
        pitcher_days_since_prev_game=state.pitcher_days_since_prev_game,
    )
    feats = incremental.features_for(pending)
    if feats.height == 0:
        raise HTTPException(422, "Could not build features for this game state.")

    pitch_proba = model.predict_proba(feats)[0]
    result: dict[str, Any] = {
        "pitcher_id": state.pitcher_id,
        "pitch_type": _proba_by_pitch_type(model, pitch_proba),
        "sample_size": int(feats["prior_pitches_seen"][0] or 0),
    }
    if loc_model is not None:
        result["location"] = _proba_by_location(loc_model.predict_proba(feats)[0])
    return result


@router.get("/games/{game_pk}/replay")
def game_replay(game_pk: int, pitcher_id: int) -> list[dict[str, Any]]:
    """Every pitch a pitcher threw in one game, actual vs. predicted.

    Uses the batch feature path — same function the model trained on — so this
    is exactly what the model saw at training time, not a live re-derivation.
    Each pitch's prediction is made from state strictly before it, same as
    training: pitch N's prediction never sees pitch N's own outcome.
    """
    require_table("fact_pitch")
    model, loc_model = _require_models()

    season_row = warehouse().execute(
        "SELECT DISTINCT season FROM fact_pitch WHERE game_pk = $g", {"g": game_pk}
    ).to_pylist()
    if not season_row:
        raise HTTPException(404, f"No pitches found for game_pk {game_pk}")
    season = season_row[0]["season"]

    game = load_batch_frame(seasons=[season], pitcher_ids=[pitcher_id]).filter(
        pl.col("game_pk") == game_pk
    )
    if game.height == 0:
        raise HTTPException(404, f"Pitcher {pitcher_id} did not pitch in game {game_pk}")

    game = game.sort(["at_bat_number", "pitch_number"])
    feats = build_features(game, with_targets=False)
    actual_loc = game.select(location_class_expr().alias("actual_location"))

    pitch_proba = model.predict_proba(feats)
    loc_proba = loc_model.predict_proba(feats) if loc_model is not None else None

    out = []
    for i, row in enumerate(game.iter_rows(named=True)):
        entry = {
            "at_bat_number": row["at_bat_number"],
            "pitch_number": row["pitch_number"],
            "balls": row["balls"],
            "strikes": row["strikes"],
            "actual_pitch_type": row["pitch_type"],
            "actual_plate_x": row["plate_x"],
            "actual_plate_z_norm": row["plate_z_norm"],
            "actual_location_class": actual_loc["actual_location"][i],
            "predicted_pitch_type": _proba_by_pitch_type(model, pitch_proba[i])[:5],
        }
        if loc_proba is not None:
            entry["predicted_location"] = _proba_by_location(loc_proba[i])[:5]
        out.append(entry)
    return out
