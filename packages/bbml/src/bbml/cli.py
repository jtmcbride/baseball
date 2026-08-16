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


@app.command("stuff")
def train_pitch_quality(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    rounds: Annotated[int, typer.Option(help="Max boosting rounds.")] = 2000,
) -> None:
    """Train Stuff+ / Location+ / Pitching+ and register all three heads.

    All three regress the same count-based run value target on different slices
    of the same frame, so they are trained together — a Stuff+ and a Location+
    built from different target definitions would not decompose into anything.
    """
    from bbml import datasets as ds
    from bbml.features.run_value import RunValue
    from bbml.features.stuff import ROLES, TARGET_RUN_VALUE, build_pitch_quality_frame
    from bbml.models.pitch_quality import (
        PitchQualityModel,
        aggregate_correlation,
        evaluate,
        predictive_validity,
        stability,
    )
    from bbml.registry import save_model

    frame = build_pitch_quality_frame(seasons=list(season) if season else None).sort(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )
    # check_features=False: these models are supposed to read the pitch itself.
    split = ds.auto_split(frame, check_features=False)

    # The run value table is fitted on TRAIN only. It is a league aggregate, but
    # one fitted across the whole frame would still carry test-season outcomes
    # into the labels the model is scored against.
    rv = RunValue.fit(split.train)
    train, val, test = (rv.attach(p) for p in (split.train, split.val, split.test))
    scored = rv.attach(frame)
    console.print(f"target sd {scored[TARGET_RUN_VALUE].std():.4f}")

    table = Table(title="pitch quality")
    for col in ("", "iters", "r2", "rank", "agg corr", "yoy grade", "yoy actual"):
        table.add_column(col, justify="right" if col else "left")

    for role in ROLES:
        model = PitchQualityModel(role=role).fit(train, val, num_boost_round=rounds)
        ev = evaluate(model, test)
        agg, n_groups = aggregate_correlation(model, test)
        yoy = stability(model, scored)
        pv = predictive_validity(model, scored)
        yoy_grade = float(yoy[f"{role}_yoy"].mean()) if yoy.height else float("nan")
        yoy_actual = float(yoy["own_rv_yoy"].mean()) if yoy.height else float("nan")
        table.add_row(
            role,
            str(model.best_iteration),
            f"{ev.r2:.4f}",
            f"{ev.rank_corr:.3f}",
            f"{agg:.3f}",
            f"{yoy_grade:.3f}",
            f"{yoy_actual:.3f}",
        )

        directory = save_model(
            model,
            f"{role}_plus",
            params={"rounds": rounds, "best_iteration": model.best_iteration},
            metrics={
                "r2": ev.r2,
                "rank_corr": ev.rank_corr,
                "aggregate_corr": agg,
                "aggregate_groups": float(n_groups),
                "stability": yoy_grade,
                "stability_of_actual": yoy_actual,
                "predictive_validity": (
                    float(pv[f"{role}_vs_next"].mean()) if pv.height else float("nan")
                ),
                "predictive_validity_of_actual": (
                    float(pv["own_rv_vs_next"].mean()) if pv.height else float("nan")
                ),
            },
        )
        # The target definition ships with the model: a grade is only meaningful
        # against the run value table it was fitted to.
        rv.save(directory / "run_value.json")

    console.print(table)
    console.print("[dim]r2 is per-pitch and expected to be ~0 — see pitch_quality.py.[/dim]")
    build_stuff_mart(season=list(season) if season else None)


@app.command("stuff-mart")
def build_stuff_mart(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
) -> None:
    """Score every pitch with the registered heads and rebuild `mart_pitcher_stuff`."""
    from bbml.marts import build_pitch_quality_mart

    mart = build_pitch_quality_mart(seasons=list(season) if season else None)
    console.print(f"[green]mart_pitcher_stuff: {mart.height} rows[/green]")


@app.command("swing")
def train_swing_path(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to 2023+.")] = None,
    rounds: Annotated[int, typer.Option(help="Max boosting rounds.")] = 2000,
    min_swings: Annotated[int, typer.Option(help="Batter-season qualifier.")] = 200,
) -> None:
    """Train the swing-path heads and report the plane-value leaderboard.

    Needs `vaa_deg` in the lake — rebuild with `bb build pitches` if this errors
    on a missing column.
    """
    from bbml import datasets as ds
    from bbml.features.swing import build_swing_frame
    from bbml.models.swing_path import (
        ROLES,
        SwingPathModel,
        evaluate,
        plane_value_by_batter,
    )
    from bbml.registry import save_model

    frame = build_swing_frame(seasons=list(season) if season else None).sort(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )
    # check_features=False: this model reads the pitch AND the swing, which is
    # the whole point — see the note in `datasets.validate`.
    split = ds.auto_split(frame, check_features=False)

    table = Table(title="swing path")
    for col in ("head", "n", "metric", "value", "league plane"):
        table.add_column(col, justify="right" if col != "head" else "left")

    for role in ROLES:
        model = SwingPathModel(role=role).fit(split.train, split.val, num_boost_round=rounds)
        ev = evaluate(model, split.test)
        headline = ("auc", ev["auc"]) if role == "whiff" else ("r2", ev["r2"])
        table.add_row(
            role,
            f"{int(ev['n']):,}",
            headline[0],
            f"{headline[1]:.4f}",
            f"{model.league_plane:.1f}°",
        )

        directory = save_model(
            model,
            f"swing_{role}",
            params={"rounds": rounds, "best_iteration": model.best_iteration},
            metrics={k: v for k, v in ev.items() if k != "n"},
        )
        console.print(f"[dim]saved {directory}[/dim]")

        if role == "whiff":
            board = plane_value_by_batter(model, split.test, min_swings=min_swings)
            best = Table(title="swing plane value — whiffs avoided per 100 swings")
            for col in ("batter", "season", "attack angle", "per 100"):
                best.add_column(col, justify="right" if col != "batter" else "left")
            for row in board.head(5).iter_rows(named=True):
                best.add_row(
                    str(row["batter"]),
                    str(row["season"]),
                    f"{row['attack_angle']:.1f}°",
                    f"{row['plane_value_per_100']:+.2f}",
                )
            console.print(best)

    console.print(table)


@app.command("status")
def status() -> None:
    """Show which model versions are registered."""
    from bbml.registry import latest_version

    s = get_settings()
    table = Table(title="Registered models")
    table.add_column("name")
    table.add_column("latest version")
    for name in (
        "next_pitch",
        "location",
        "stuff_plus",
        "location_plus",
        "pitching_plus",
        "swing_whiff",
        "swing_contact",
    ):
        d = s.models_dir / name
        version = latest_version(name, settings=s) if d.exists() else None
        table.add_row(name, version or "[dim]none[/dim]")
    console.print(table)


if __name__ == "__main__":
    app()
