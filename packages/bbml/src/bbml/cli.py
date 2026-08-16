"""`bb-ml` — train and evaluate the next-pitch and location models."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from bbcore.config import get_settings
from bbcore.logging import setup_logging

app = typer.Typer(add_completion=False, help="Train and evaluate baseball ML models.")
console = Console()


@app.callback()
def _root() -> None:
    setup_logging()


def _load_split(seasons: list[int] | None):
    from bbml import datasets as ds
    from bbml.features.context import build_batch_features

    feats = ds.prepare(build_batch_features(seasons=seasons))
    split = ds.auto_split(feats)
    console.print(split.describe())
    return split


@app.command("next-pitch")
def train_next_pitch(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    rounds: Annotated[int, typer.Option(help="Max boosting rounds.")] = 1500,
) -> None:
    """Train the pitch-type model, evaluate against the baseline, and register it."""
    from bbml.features.schema import TARGET_PITCH_TYPE
    from bbml.models.baseline import UsageRateBaseline, evaluate, log_loss
    from bbml.models.next_pitch import NextPitchModel
    from bbml.models.personalize import PersonalizedBlend
    from bbml.registry import save_model

    split = _load_split(list(season) if season else None)

    baseline = UsageRateBaseline().fit(split.train)
    base_ll = log_loss(
        split.test[TARGET_PITCH_TYPE].to_list(),
        baseline.predict_proba(split.test),
        baseline.classes,
    )

    model = NextPitchModel().fit(split.train, split.val, num_boost_round=rounds)
    proba = model.predict_proba(split.test)
    ev = evaluate(split.test[TARGET_PITCH_TYPE].to_list(), proba, model.classes)

    blend = PersonalizedBlend.fit(split.train)
    blended = blend.apply(proba, split.test)
    ev_blended = evaluate(split.test[TARGET_PITCH_TYPE].to_list(), blended, model.classes)

    table = Table(title="next-pitch model")
    table.add_column("")
    table.add_column("log_loss", justify="right")
    table.add_column("top1", justify="right")
    table.add_column("ece", justify="right")
    table.add_row("baseline", f"{base_ll:.4f}", "-", "-")
    table.add_row("model", f"{ev.log_loss:.4f}", f"{ev.top1:.3f}", f"{ev.ece:.4f}")
    table.add_row(
        "model + blend",
        f"{ev_blended.log_loss:.4f}",
        f"{ev_blended.top1:.3f}",
        f"{ev_blended.ece:.4f}",
    )
    console.print(table)

    directory = save_model(
        model,
        "next_pitch",
        params={"rounds": rounds, "best_iteration": model.best_iteration},
        metrics={
            "baseline_log_loss": base_ll,
            "log_loss": ev.log_loss,
            "top1": ev.top1,
            "ece": ev.ece,
            "improvement_over_baseline": (base_ll - ev.log_loss) / base_ll,
        },
    )
    console.print(f"[green]Saved to {directory}[/green]")


@app.command("location")
def train_location(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    rounds: Annotated[int, typer.Option(help="Max boosting rounds.")] = 1500,
) -> None:
    """Train the location model and register it."""
    from bbml.features.schema import TARGET_LOCATION
    from bbml.models.baseline import log_loss, top_k_accuracy
    from bbml.models.location import LocationModel
    from bbml.registry import save_model

    split = _load_split(list(season) if season else None)

    model = LocationModel().fit(split.train, split.val, num_boost_round=rounds)
    proba = model.predict_proba(split.test)
    classes = list(range(model.n_classes))
    y = split.test[TARGET_LOCATION].to_list()
    ll = log_loss([str(v) if v is not None else None for v in y], proba, [str(c) for c in classes])
    top1 = top_k_accuracy(
        [str(v) if v is not None else None for v in y], proba, [str(c) for c in classes], k=1
    )

    table = Table(title="location model")
    table.add_column("log_loss", justify="right")
    table.add_column("top1", justify="right")
    table.add_row(f"{ll:.4f}", f"{top1:.3f}")
    console.print(table)

    directory = save_model(
        model,
        "location",
        params={"rounds": rounds, "best_iteration": model.best_iteration},
        metrics={"log_loss": ll, "top1": top1},
    )
    console.print(f"[green]Saved to {directory}[/green]")


@app.command("status")
def status() -> None:
    """Show which model versions are registered."""
    from bbml.registry import latest_version

    s = get_settings()
    table = Table(title="Registered models")
    table.add_column("name")
    table.add_column("latest version")
    for name in ("next_pitch", "location"):
        d = s.models_dir / name
        version = latest_version(name, settings=s) if d.exists() else None
        table.add_row(name, version or "[dim]none[/dim]")
    console.print(table)


if __name__ == "__main__":
    app()
