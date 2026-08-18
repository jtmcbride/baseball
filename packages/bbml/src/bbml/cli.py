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
    build_swing_mart(season=list(season) if season else None, min_swings=min_swings)


@app.command("swing-mart")
def build_swing_mart(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    min_swings: Annotated[int, typer.Option(help="Batter-season qualifier.")] = 200,
) -> None:
    """Score every swing with the registered heads and rebuild `mart_batter_swing`."""
    from bbml.marts import build_batter_swing_mart

    mart = build_batter_swing_mart(seasons=list(season) if season else None, min_swings=min_swings)
    console.print(f"[green]mart_batter_swing: {mart.height} rows[/green]")


@app.command("spray-mart")
def build_spray_mart(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    min_batted_balls: Annotated[int, typer.Option(help="Batter-season qualifier.")] = 100,
) -> None:
    """Rebuild `mart_batter_spray` (viz #8): smoothed xwOBA-on-contact surface
    over field position, batter x season. No model to score first — it reads
    `x_ft`/`y_ft` (`bbetl.transforms.statcast.enrich`) straight off `fact_pitch`."""
    from bbml.marts import build_batter_spray_mart

    mart = build_batter_spray_mart(
        seasons=list(season) if season else None, min_batted_balls=min_batted_balls
    )
    console.print(f"[green]mart_batter_spray: {mart.height} rows[/green]")


@app.command("called-strike")
def train_called_strike(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    rounds: Annotated[int, typer.Option(help="Max boosting rounds.")] = 1000,
    min_pitches: Annotated[int, typer.Option(help="Catcher/umpire qualifier.")] = 500,
) -> None:
    """Train the called-strike model and report catcher framing runs.

    Needs `dim_official` for umpire ids — run `bb ingest officials` then
    `bb build officials` first if the umpire column comes back all null.
    """
    import polars as pl

    from bbml import datasets as ds
    from bbml.features.called_strike import CATCHER_COLUMN, UMPIRE_COLUMN, build_called_strike_frame
    from bbml.features.run_value import RunValue
    from bbml.features.stuff import load_pitch_frame
    from bbml.models.called_strike import CalledStrikeModel, evaluate, framing_runs
    from bbml.registry import save_model

    frame = build_called_strike_frame(seasons=list(season) if season else None).sort(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )
    # check_features=False: this model reads the pitch itself, same reasoning
    # as the pitch-quality and swing-path models.
    split = ds.auto_split(frame, check_features=False)

    # RunValue needs the WHOLE plate appearance — including the swings this
    # taken-pitches-only frame drops — to know how each one actually ended, so
    # it is fit on a separate load rather than on `split.train`.
    train_seasons = sorted(split.train["season"].unique().to_list())
    rv = RunValue.fit(load_pitch_frame(seasons=train_seasons))

    model = CalledStrikeModel().fit(split.train, split.val, num_boost_round=rounds)
    ev = evaluate(model, split.test)
    console.print(
        f"log_loss={ev.log_loss:.4f}  auc={ev.auc:.4f}  ece={ev.ece:.4f}  n={ev.n:,}"
    )

    directory = save_model(
        model,
        "called_strike",
        params={"rounds": rounds, "best_iteration": model.best_iteration},
        metrics={"log_loss": ev.log_loss, "auc": ev.auc, "ece": ev.ece},
    )
    rv.save(directory / "run_value.json")
    console.print(f"[dim]saved {directory}[/dim]")

    framing = framing_runs(split.test, model, rv, group_col=CATCHER_COLUMN, min_pitches=min_pitches)
    table = Table(title="catcher framing runs (test season)")
    for col in ("catcher", "framing runs", "n"):
        table.add_column(col, justify="right" if col != "catcher" else "left")
    for row in framing.head(5).iter_rows(named=True):
        table.add_row(str(row[CATCHER_COLUMN]), f"{row['framing_runs']:+.2f}", f"{row['n']:,}")
    console.print(table)

    n_umpires = split.test.filter(pl.col(UMPIRE_COLUMN).is_not_null()).height
    if n_umpires == 0:
        console.print(
            "[yellow]No umpire ids in the test split — dim_official missing or "
            "not joined. Framing runs above are catcher-only.[/yellow]"
        )

    build_called_strike_marts(season=list(season) if season else None)


@app.command("called-strike-mart")
def build_called_strike_marts(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
) -> None:
    """Score every take with the registered model and rebuild the catcher
    framing and umpire zone marts, plus their spatial grids (viz #20/#13)."""
    from bbml.marts import (
        build_catcher_framing_grid,
        build_catcher_framing_mart,
        build_umpire_zone_grid,
        build_umpire_zone_mart,
    )

    catcher = build_catcher_framing_mart(seasons=list(season) if season else None)
    umpire = build_umpire_zone_mart(seasons=list(season) if season else None)
    console.print(f"[green]mart_catcher_framing: {catcher.height} rows[/green]")
    console.print(f"[green]mart_umpire_zone: {umpire.height} rows[/green]")

    catcher_grid = build_catcher_framing_grid(seasons=list(season) if season else None)
    umpire_grid = build_umpire_zone_grid(seasons=list(season) if season else None)
    console.print(f"[green]mart_zone_profile (catcher): {catcher_grid.height} grids[/green]")
    console.print(f"[green]mart_zone_profile (umpire): {umpire_grid.height} grids[/green]")


@app.command("arsenal")
def arsenal_clusters(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable. Defaults to all.")] = None,
    min_pitches: Annotated[int, typer.Option(help="Pitcher-season qualifier.")] = 200,
) -> None:
    """Re-derive each pitcher-season's arsenal from physical shape (GMM, BIC-
    selected k) and rebuild `mart_pitcher_arsenal_clusters` (M3 model #2).

    No training step, no registered artifact — a small model is fit per
    pitcher-season, same shape as the zone-profile grids. Prints the
    pitcher-seasons where this disagrees most with Savant's own `pitch_type`
    labels (largest |arsenal_size_diff|) so a run is spot-checkable without
    opening the Parquet.
    """
    from bbml.marts import build_arsenal_cluster_mart

    mart = build_arsenal_cluster_mart(seasons=list(season) if season else None, min_pitches=min_pitches)
    console.print(f"[green]mart_pitcher_arsenal_clusters: {mart.height} rows[/green]")
    if mart.height == 0:
        return

    import polars as pl

    per_season = mart.unique(subset=["mlbam_id", "season"]).with_columns(
        pl.col("arsenal_size_diff").abs().alias("_abs_diff")
    )
    table = Table(title="Largest disagreements with Savant's pitch_type (|arsenal_size_diff|)")
    for col in ("pitcher", "season", "cluster_k", "savant_pitch_types", "arsenal_size_diff", "season_purity"):
        table.add_column(col, justify="right" if col != "pitcher" else "left")
    for row in per_season.sort("_abs_diff", descending=True).head(10).iter_rows(named=True):
        table.add_row(
            str(row["mlbam_id"]),
            str(row["season"]),
            str(row["cluster_k"]),
            str(row["savant_pitch_types"]),
            f"{row['arsenal_size_diff']:+d}",
            f"{row['season_purity']:.2f}",
        )
    console.print(table)


@app.command("arsenal-embed")
def arsenal_embed(
    encoding: Annotated[str, typer.Option(help="slot or histogram — see arsenal_embed.py.")] = "slot",
    reducer: Annotated[str, typer.Option(help="tsne, umap, or pca.")] = "tsne",
    neighbors: Annotated[int, typer.Option(help="Nearest neighbors stored per pitcher-season.")] = 10,
) -> None:
    """Embed every pitcher-season from `mart_pitcher_arsenal_clusters` into 2D
    and cluster it into archetypes (M3 model #11, backs viz #12: "who does
    this pitcher resemble?"). Rebuilds `mart_arsenal_embedding` and
    `mart_arsenal_neighbors`. Run `bb-ml arsenal` first if the cluster mart
    is missing.
    """
    import polars as pl

    from bbml.marts import build_arsenal_embedding_marts
    from bbml.models.arsenal_embed import NAMED_SPOT_CHECKS

    embedding, nbrs, validation = build_arsenal_embedding_marts(
        encoding=encoding, reducer=reducer, n_neighbors=neighbors
    )
    console.print(f"[green]mart_arsenal_embedding: {embedding.height} rows[/green]")
    console.print(f"[green]mart_arsenal_neighbors: {nbrs.height} rows[/green]")

    table = Table(title=f"arsenal embedding validation ({encoding} / {reducer})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in ("trustworthiness", "yoy_neighbor_rank", "n_pitcher_seasons", "k", "silhouette"):
        table.add_row(key, f"{validation[key]:.4f}" if isinstance(validation[key], float) else str(validation[key]))
    console.print(table)

    arch = Table(title="archetypes")
    for col in ("archetype_id", "label", "n"):
        arch.add_column(col, justify="right" if col != "label" else "left")
    counts = embedding.group_by("archetype_id").agg(pl.col("archetype_label").first(), pl.len())
    for row in counts.sort("archetype_id").iter_rows(named=True):
        arch.add_row(str(row["archetype_id"]), row["archetype_label"], str(row["len"]))
    console.print(arch)

    console.print("[dim]Named spot-checks (nearest neighbors by feature-space distance):[/dim]")
    for pid, desc in NAMED_SPOT_CHECKS.items():
        neighbors_for = nbrs.filter(pl.col("mlbam_id") == pid).sort(
            ["season", "rank"], descending=[True, False]
        )
        console.print(f"  {pid} — {desc}")
        for row in neighbors_for.head(5).iter_rows(named=True):
            console.print(
                f"    #{row['rank']}: {row['neighbor_id']} ({row['neighbor_season']}) d={row['distance']:.3f}"
            )


@app.command("arsenal-bakeoff")
def arsenal_bakeoff() -> None:
    """Re-run the encoding x reducer bake-off from `arsenal_embed.py`'s
    docstring against the current lake and print the table. Slow (fits
    t-SNE/UMAP six times); for reproducing the measured numbers, not routine
    use."""
    import polars as pl

    from bbcore.config import get_settings
    from bbml.marts import MART_ARSENAL_CLUSTERS
    from bbml.models.arsenal_embed import bake_off

    s = get_settings()
    cluster_path = s.lake_dir / MART_ARSENAL_CLUSTERS
    if not any(cluster_path.glob("*.parquet")):
        console.print(f"[red]{MART_ARSENAL_CLUSTERS} not found. Run `bb-ml arsenal` first.[/red]")
        raise typer.Exit(1)
    df = pl.read_parquet(cluster_path / "*.parquet")
    result = bake_off(df)

    table = Table(title="arsenal embedding bake-off")
    for col in result.columns:
        table.add_column(col, justify="right" if col not in ("encoding", "reducer") else "left")
    for row in result.iter_rows():
        table.add_row(*(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row))
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
        "called_strike",
    ):
        d = s.models_dir / name
        version = latest_version(name, settings=s) if d.exists() else None
        table.add_row(name, version or "[dim]none[/dim]")
    console.print(table)


if __name__ == "__main__":
    app()
