"""Mart builds: execute the SQL in sql/marts/ and the Python-side zone grids."""

from __future__ import annotations

from pathlib import Path

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbcore.storage import open_warehouse
from bbetl.warehouse import register_all

log = get_logger(__name__)

SQL_DIR = Path(__file__).resolve().parents[4] / "sql"

# Order matters where one mart reads another.
SQL_MARTS = ["mart_pitcher_arsenal"]


def build_sql_marts(
    names: list[str] | None = None, *, settings: Settings | None = None
) -> dict[str, int]:
    s = settings or get_settings()
    register_all(settings=s)
    targets = names or SQL_MARTS
    out: dict[str, int] = {}
    with open_warehouse(settings=s) as wh:
        for name in targets:
            path = SQL_DIR / "marts" / f"{name}.sql"
            if not path.exists():
                raise FileNotFoundError(f"Missing mart SQL: {path}")
            log.info("building %s", name)
            wh.execute_none(path.read_text())
            out[name] = wh.scalar(f"SELECT count(*) FROM {name}") or 0
            log.info("  %s: %d rows", name, out[name])
    return out


def build_zone_marts(*, settings: Settings | None = None, min_pitches: int = 250) -> dict[str, int]:
    from bbetl.transforms.zones import build_zone_profiles

    s = settings or get_settings()
    out: dict[str, int] = {}
    for role in ("batter", "pitcher"):
        df = build_zone_profiles(role=role, min_pitches=min_pitches, settings=s)  # type: ignore[arg-type]
        out[role] = df.height
    # Expose the new grids through the warehouse alongside everything else.
    register_all(settings=s)
    return out
