"""FastAPI application: read-only access to the baseball warehouse."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bbapi.deps import settings, warehouse
from bbapi.routers import pitches, players, predict, zones
from bbcore.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Baseball Analytics API",
    version="0.1.0",
    description="Statcast pitch data, arsenals, and hot/cold zone grids.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().cors_origin_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)

meta = APIRouter(tags=["meta"])


@meta.get("/health")
def health() -> dict[str, Any]:
    """Reports which pipeline stages have actually run — the first thing to check
    when the UI is empty."""
    wh = warehouse()
    tables = {
        name: wh.table_exists(name)
        for name in (
            "fact_pitch",
            "dim_player",
            "dim_game",
            "dim_team",
            "dim_player_ids",
            "mart_pitcher_arsenal",
            "mart_zone_profile",
        )
    }
    pitch_count = wh.scalar("SELECT count(*) FROM fact_pitch") if tables["fact_pitch"] else 0
    return {"status": "ok", "tables": tables, "pitches": pitch_count}


@meta.get("/seasons")
def seasons() -> list[dict[str, Any]]:
    return (
        warehouse()
        .execute(
            """
        SELECT season, count(*) AS pitches, count(DISTINCT game_pk) AS games,
               min(game_date) AS first_game, max(game_date) AS last_game
        FROM fact_pitch GROUP BY season ORDER BY season DESC
        """
        )
        .to_pylist()
    )


app.include_router(meta)
app.include_router(players.router)
app.include_router(pitches.router)
app.include_router(zones.router)
app.include_router(predict.router)


def main() -> None:
    import uvicorn

    s = settings()
    uvicorn.run("bbapi.main:app", host=s.api_host, port=s.api_port, reload=True)


if __name__ == "__main__":
    main()
