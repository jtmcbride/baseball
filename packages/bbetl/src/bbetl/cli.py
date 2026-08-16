"""`bb` — the ingestion and transform CLI.

Every job is idempotent: re-running skips completed partitions, an interrupted
backfill resumes, and `--force` re-fetches.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from bbcore.config import get_settings
from bbcore.logging import setup_logging

app = typer.Typer(add_completion=False, help="Baseball analytics data pipeline.")
ingest_app = typer.Typer(help="Fetch source data into the raw layer.")
app.add_typer(ingest_app, name="ingest")

console = Console()


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


@app.callback()
def _root() -> None:
    setup_logging()


# --- ingest ------------------------------------------------------------------


@ingest_app.command("statcast")
def ingest_statcast(
    start: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD. Defaults to BB_SEASON_START.")
    ] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD. Defaults to today.")] = None,
    force: Annotated[bool, typer.Option(help="Re-fetch partitions already marked done.")] = False,
) -> None:
    """Backfill pitch-level Statcast data, one partition per game date."""
    from bbetl.sources.savant import ingest_range

    s = get_settings()
    s.ensure_dirs()
    start_d = _parse_date(start) if start else dt.date(s.season_start, 2, 1)
    end_d = _parse_date(end) if end else dt.date.today()

    stats = ingest_range(start_d, end_d, settings=s, force=force)
    _print_stats("statcast", stats)


@ingest_app.command("refresh")
def ingest_refresh(
    days: Annotated[
        int | None, typer.Option(help="Trailing window. Defaults to BB_REFRESH_WINDOW_DAYS.")
    ] = None,
) -> None:
    """Re-pull recent days. Savant revises classifications and xwOBA after publication."""
    from bbetl.sources.savant import refresh_recent

    s = get_settings()
    s.ensure_dirs()
    stats = refresh_recent(settings=s, days=days)
    _print_stats("statcast refresh", stats)


@ingest_app.command("dims")
def ingest_dims(
    start: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD. Defaults to BB_SEASON_START.")
    ] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD. Defaults to today.")] = None,
) -> None:
    """Build dim_game, dim_team, and dim_player from the MLB Stats API.

    Player ids are taken from the pitches already ingested, so run this after
    `bb ingest statcast` and `bb build pitches`.
    """
    import polars as pl

    from bbetl.sources.statsapi import StatsAPIClient, ingest_games, ingest_people, ingest_teams

    s = get_settings()
    s.ensure_dirs()
    start_d = _parse_date(start) if start else dt.date(s.season_start, 1, 1)
    end_d = _parse_date(end) if end else dt.date.today()

    client = StatsAPIClient(settings=s)
    try:
        n_games = ingest_games(start_d, end_d, settings=s, client=client)
        n_teams = ingest_teams(list(range(start_d.year, end_d.year + 1)), settings=s, client=client)

        pitch_glob = s.lake_dir / "fact_pitch" / "season=*" / "*.parquet"
        ids: list[int] = []
        if any(s.lake_dir.glob("fact_pitch/season=*/*.parquet")):
            df = pl.scan_parquet(pitch_glob).select("batter", "pitcher").collect()
            ids = sorted(
                set(df["batter"].drop_nulls().to_list()) | set(df["pitcher"].drop_nulls().to_list())
            )
        else:
            console.print("[yellow]No fact_pitch yet — skipping dim_player.[/yellow]")
        n_people = ingest_people(ids, settings=s, client=client) if ids else 0
    finally:
        client.close()

    _print_stats("dims", {"games": n_games, "teams": n_teams, "players": n_people})


@ingest_app.command("crosswalk")
def ingest_crosswalk() -> None:
    """Build the Chadwick MLBAM<->FanGraphs<->BBRef<->Retrosheet ID crosswalk."""
    from bbetl.sources.chadwick import build_crosswalk

    s = get_settings()
    s.ensure_dirs()
    df = build_crosswalk(settings=s)
    console.print(f"Crosswalk built: [green]{df.height:,}[/green] players")


@ingest_app.command("officials")
def ingest_officials_cmd(
    season: Annotated[list[int] | None, typer.Option(help="Repeatable.")] = None,
    force: Annotated[bool, typer.Option()] = False,
) -> None:
    """Fetch home-plate umpires (one request per game).

    Statcast's `umpire` column is empty in every season, so this is the only
    source. Needed for the called-strike / framing models; skip it for M1.
    """
    import polars as pl

    from bbetl.sources.statsapi import ingest_officials

    s = get_settings()
    path = s.lake_dir / "dim_game" / "part_0.parquet"
    if not path.exists():
        console.print("[red]dim_game missing — run `bb ingest dims` first.[/red]")
        raise typer.Exit(1)
    games = pl.read_parquet(path)
    if season:
        games = games.filter(pl.col("season").is_in(list(season)))
    n = ingest_officials(games["game_pk"].to_list(), settings=s, force=force)
    console.print(f"Fetched officials for [green]{n:,}[/green] games")


# --- build -------------------------------------------------------------------

build_app = typer.Typer(help="Transform raw data into the Parquet lake and marts.")
app.add_typer(build_app, name="build")


@build_app.command("pitches")
def build_pitches(
    season: Annotated[
        list[int] | None, typer.Option(help="Repeatable. Defaults to all ingested.")
    ] = None,
) -> None:
    """Rebuild `fact_pitch` from landed raw files. Never re-crawls."""
    from bbetl.transforms.statcast import build_all

    s = get_settings()
    s.ensure_dirs()
    result = build_all(list(season) if season else None, settings=s)

    table = Table(title="fact_pitch")
    table.add_column("season")
    table.add_column("pitches", justify="right")
    for yr, n in sorted(result.items()):
        table.add_row(str(yr), f"{n:,}")
    table.add_row("[bold]total[/bold]", f"[bold]{sum(result.values()):,}[/bold]")
    console.print(table)


@build_app.command("marts")
def build_marts(
    min_pitches: Annotated[int, typer.Option(help="Qualifier for zone grids.")] = 250,
    skip_zones: Annotated[bool, typer.Option(help="SQL marts only.")] = False,
) -> None:
    """Build analytical marts from fact_pitch."""
    from bbetl.marts import build_sql_marts, build_zone_marts

    s = get_settings()
    s.ensure_dirs()
    sql_out = build_sql_marts(settings=s)
    zone_out = {} if skip_zones else build_zone_marts(settings=s, min_pitches=min_pitches)

    table = Table(title="marts")
    table.add_column("mart")
    table.add_column("rows", justify="right")
    for k, v in sql_out.items():
        table.add_row(k, f"{v:,}")
    for k, v in zone_out.items():
        table.add_row(f"mart_zone_profile ({k})", f"{v:,}")
    console.print(table)


@build_app.command("all")
def build_everything() -> None:
    """fact_pitch -> register -> marts, in order."""
    build_pitches(season=None)
    build_register()
    build_marts(min_pitches=250, skip_zones=False)


@build_app.command("register")
def build_register() -> None:
    """(Re)create warehouse views over the lake."""
    from bbetl.warehouse import register_all

    names = register_all()
    console.print(f"Registered {len(names)} table(s): {', '.join(names)}")


# --- status ------------------------------------------------------------------


@app.command("status")
def status(
    source: Annotated[str | None, typer.Option(help="Limit to one source.")] = None,
) -> None:
    """Show ingest progress from the manifest."""
    from bbetl.manifest import Manifest

    mf = Manifest()
    rows = mf.summary(source)
    if not rows:
        console.print("[yellow]No ingest runs recorded yet.[/yellow]")
        raise typer.Exit()

    table = Table(title="Ingest manifest")
    table.add_column("source")
    table.add_column("status")
    table.add_column("partitions", justify="right")
    table.add_column("rows", justify="right")
    for src, st, n, total in rows:
        style = {"ok": "green", "failed": "red", "empty": "dim"}.get(st, "")
        table.add_row(src, f"[{style}]{st}[/{style}]" if style else st, f"{n:,}", f"{total:,}")
    console.print(table)

    if source:
        fails = mf.failures(source)
        if fails:
            console.print(f"\n[red]{len(fails)} failed partitions:[/red]")
            for key, err in fails[:20]:
                console.print(f"  {key}: {err}")


@app.command("check")
def check(
    coverage: Annotated[bool, typer.Option(help="Also print per-season column coverage.")] = False,
) -> None:
    """Run data quality checks. Exits non-zero on any ERROR."""
    from bbetl.quality import run_checks, season_coverage

    report = run_checks()

    table = Table(title="Data quality")
    table.add_column("check")
    table.add_column("severity")
    table.add_column("result")
    for r in report.results:
        mark = (
            "[green]PASS[/green]"
            if r.passed
            else ("[red]FAIL[/red]" if r.severity == "error" else "[yellow]WARN[/yellow]")
        )
        table.add_row(r.name, r.severity, mark)
    console.print(table)

    for r in report.failed:
        style = "red" if r.severity == "error" else "yellow"
        console.print(f"\n[{style}]{r.name}[/{style}]: {r.detail}")
        for row in r.rows[:5]:
            console.print(f"    {row}")

    if coverage:
        console.print("\n[bold]Per-season column coverage (% non-null)[/bold]")
        console.print(season_coverage())

    if not report.ok:
        console.print(f"\n[red]{len(report.errors)} error-level check(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All error-level checks passed.[/green]")


@app.command("config")
def show_config() -> None:
    """Print resolved settings — useful for confirming .env actually loaded."""
    s = get_settings()
    table = Table(title="Resolved settings")
    table.add_column("key")
    table.add_column("value")
    for k, v in s.model_dump().items():
        table.add_row(k, str(v))
    for extra in ("raw_dir", "lake_dir", "db_dir", "duckdb_file", "current_season"):
        table.add_row(f"[dim]{extra}[/dim]", str(getattr(s, extra)))
    console.print(table)


def _print_stats(label: str, stats: dict[str, int]) -> None:
    table = Table(title=f"{label} result")
    for k in stats:
        table.add_column(k, justify="right")
    table.add_row(*[f"{v:,}" for v in stats.values()])
    console.print(table)


if __name__ == "__main__":
    app()
