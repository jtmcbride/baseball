"""Central settings. Everything path-, URL-, or limit-shaped lives here.

The point of routing all of it through one object is the cloud-ready seam: moving
from a local DuckDB file to a hosted Postgres should be an env change, not a code
change. Nothing downstream may hardcode a path or a rate limit.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]

WarehouseBackend = Literal["duckdb", "postgres"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BB_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage -------------------------------------------------------------
    data_root: Path = Field(default=REPO_ROOT / "data")
    warehouse_backend: WarehouseBackend = "duckdb"
    duckdb_path: Path | None = None
    postgres_dsn: str | None = None

    # --- Ingestion -----------------------------------------------------------
    season_start: int = 2015
    savant_rps: float = 0.25
    savant_max_retries: int = 5
    savant_timeout_s: float = 120.0
    statsapi_rps: float = 4.0
    fangraphs_rps: float = 0.2
    refresh_window_days: int = 14
    user_agent: str = "baseball-analytics/0.1 (personal research)"

    # --- API -----------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    # --- Derived paths -------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        """Immutable landed responses. Never written to twice for the same key."""
        return self.data_root / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lake_dir(self) -> Path:
        """Typed, hive-partitioned Parquet. The analytical source of truth."""
        return self.data_root / "lake"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_dir(self) -> Path:
        return self.data_root / "db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duckdb_file(self) -> Path:
        return self.duckdb_path or (self.db_dir / "baseball.duckdb")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def current_season(self) -> int:
        """Seasons roll over in the offseason; anything before March belongs to
        the prior season's data footprint."""
        today = dt.date.today()
        return today.year if today.month >= 3 else today.year - 1

    def seasons(self) -> list[int]:
        return list(range(self.season_start, self.current_season + 1))

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.lake_dir, self.db_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
