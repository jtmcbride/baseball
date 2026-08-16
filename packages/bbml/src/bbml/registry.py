"""Model registry: versioned artifacts on disk, with optional MLflow tracking.

The artifact on disk is the source of truth — `load_latest` never touches
MLflow, so serving works on a machine that doesn't have it installed. MLflow
(a local file-backed tracking store, no server) is purely a side channel for
comparing runs; if it isn't installed, `save_model` logs a warning and still
completes the part that matters.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Protocol

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger

log = get_logger(__name__)


class _Savable(Protocol):
    def save(self, directory: Path) -> Path: ...


def _mlruns_dir(settings: Settings) -> Path:
    return settings.models_dir / "mlruns"


def save_model(
    model: _Savable,
    name: str,
    *,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Save a fitted model as the new version of `name`, and point LATEST at it.

    Versions are timestamp-named directories rather than incrementing integers —
    sortable, collision-free across concurrent training runs, and self-describing
    when browsing `data/models/` by hand.
    """
    s = settings or get_settings()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = s.models_dir / name / run_id
    model.save(directory)
    (s.models_dir / name / "LATEST").write_text(run_id)
    log.info("saved %s/%s", name, run_id)

    try:
        _log_to_mlflow(name, run_id, directory, params or {}, metrics or {}, s)
    except ImportError:
        log.warning("mlflow not installed — skipping run tracking (artifact still saved)")
    except Exception:
        log.exception("mlflow logging failed — artifact was still saved")

    return directory


def _log_to_mlflow(
    name: str,
    run_id: str,
    directory: Path,
    params: dict[str, Any],
    metrics: dict[str, float],
    settings: Settings,
) -> None:
    import mlflow

    mlruns = _mlruns_dir(settings)
    mlruns.mkdir(parents=True, exist_ok=True)
    # A sqlite backend rather than the plain file store: MLflow's filesystem
    # tracking backend is in maintenance mode and refuses new runs outright.
    mlflow.set_tracking_uri(f"sqlite:///{mlruns / 'mlflow.db'}")
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(directory))


def latest_version(name: str, *, settings: Settings | None = None) -> str | None:
    s = settings or get_settings()
    pointer = s.models_dir / name / "LATEST"
    return pointer.read_text().strip() if pointer.exists() else None


def latest_dir(name: str, *, settings: Settings | None = None) -> Path | None:
    s = settings or get_settings()
    version = latest_version(name, settings=s)
    if version is None:
        return None
    directory = s.models_dir / name / version
    return directory if directory.exists() else None
