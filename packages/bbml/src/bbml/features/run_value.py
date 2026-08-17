"""Count-based, context-neutral run value — the target every pitch-quality model regresses on.

WHY NOT JUST USE `delta_run_exp`
--------------------------------
Savant ships `delta_run_exp`, the change in base-out run expectancy caused by the
pitch. It is the right *idea* and the wrong *target*, for two separate reasons,
both measured on our own lake (2.4M pitches):

  * **Context.** It is a base-out delta, so the same strikeout is worth -0.21 with
    the bases empty and much more with the bases loaded. A stuff model has no
    access to the base-out state — that isn't a property of the pitch — so all of
    that spread is noise it can only average over. Worse, it isn't quite
    *independent* noise: pitchers work differently from the stretch.
  * **Ball-in-play luck.** Grouping `delta_run_exp` by description: every non-BIP
    outcome has SD 0.03-0.09, while `hit_into_play` alone has SD 0.487 on 20% of
    the pitches. Nearly all of the target's variance is a single bucket, and most
    of *that* is where the ball happened to land, not how good the pitch was.

WHAT THIS BUILDS INSTEAD
------------------------
The standard count-based construction, derived entirely from our own data — no
imported linear weights, no external run-expectancy table:

  1. **Terminal value** of a plate appearance = the context-averaged run value of
     its outcome, `w(event) = mean(delta_run_exp | event)`. Averaging over
     contexts is exactly what strips the base-out dependence back out.
  2. **De-noised contact.** For balls in play we do not use the observed outcome
     at all. `delta_run_exp` is regressed on Savant's `estimated_woba_using_
     speedangle` (a launch-speed/angle expectation, which knows nothing about
     where fielders stood) and the fitted value is used instead. A 108mph lineout
     and a 108mph double now carry the same value, which is the point.
  3. **Count run expectancy** `RE(balls, strikes)` = mean terminal value over the
     plate appearances that pass through that count. Twelve numbers.
  4. **Pitch run value** = `RE(count after) - RE(count before)`, or
     `terminal value - RE(count before)` on the pitch that ends the PA.

Step 4 makes `E[rv | count] == 0` at every count by construction, so the count a
pitch was thrown in carries no run value of its own and a model cannot score
points by learning which counts are favourable. That is the property that makes
this usable as a *pitch quality* target rather than a *situation* target.

TWO CONSEQUENCES WORTH KNOWING BEFORE THEY SURPRISE YOU
-------------------------------------------------------
* **A two-strike foul is worth exactly zero.** The count does not change, so the
  telescoping difference is 0. This is correct inside a count-based framework and
  is what every published Stuff+-style model does; it is not a bug to be fixed by
  special-casing fouls.
* **The scale is per-pitch and small.** Typical values are ±0.05, with a strikeout
  around +0.2 for the pitcher. Aggregate to RV/100 before showing a human.

Sign convention: `rv_pitcher` is flipped from Savant's, so **higher is better for
the pitcher** — matching `mart_pitcher_arsenal.rv_per_100`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from bbcore.logging import get_logger

log = get_logger(__name__)

# Columns `RunValue.fit` and `.attach` need on the input frame.
REQUIRED_COLUMNS: list[str] = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "events",
    "description",
    "delta_run_exp",
    "estimated_woba_using_speedangle",
]

# Plate appearances that end this way tell us nothing about the pitch that
# happened to be in flight, so they are dropped rather than given a weight.
# `intent_walk` also has a null `delta_run_exp` throughout the lake.
EXCLUDED_EVENTS: frozenset[str] = frozenset(
    {"intent_walk", "catcher_interf", "truncated_pa", "game_advisory"}
)

BIP_DESCRIPTION = "hit_into_play"

# Counts are capped rather than trusted: a handful of rows in the raw feed carry
# 4 balls or 3 strikes mid-PA.
MAX_BALLS, MAX_STRIKES = 3, 2


def _count_key(balls: pl.Expr, strikes: pl.Expr) -> pl.Expr:
    return balls.clip(0, MAX_BALLS).cast(pl.Utf8) + "-" + strikes.clip(0, MAX_STRIKES).cast(pl.Utf8)


@dataclass
class RunValue:
    """A fitted count-based run value table. Fit once, then `attach` to any frame."""

    count_re: dict[str, float] = field(default_factory=dict)
    event_value: dict[str, float] = field(default_factory=dict)
    bip_intercept: float = 0.0
    bip_slope: float = 0.0
    n_fit: int = 0

    # --- fitting -------------------------------------------------------------

    @classmethod
    def fit(cls, df: pl.DataFrame) -> RunValue:
        pa = _plate_appearances(df)

        bip = pa.filter(
            (pl.col("description") == BIP_DESCRIPTION)
            & pl.col("estimated_woba_using_speedangle").is_not_null()
            & pl.col("delta_run_exp").is_not_null()
        )
        if bip.height < 100:
            raise ValueError(
                f"Only {bip.height} balls in play with an xwOBA estimate — not enough to "
                "fit the contact de-noiser. Check that the lake actually has "
                "estimated_woba_using_speedangle populated."
            )
        slope, intercept = np.polyfit(
            bip["estimated_woba_using_speedangle"].to_numpy(),
            bip["delta_run_exp"].to_numpy(),
            deg=1,
        )

        events = (
            pa.filter(pl.col("delta_run_exp").is_not_null())
            .group_by("events")
            .agg(pl.col("delta_run_exp").mean().alias("w"))
        )
        event_value = dict(zip(events["events"].to_list(), events["w"].to_list(), strict=True))

        model = cls(
            event_value=event_value,
            bip_intercept=float(intercept),
            bip_slope=float(slope),
            n_fit=pa.height,
        )

        # RE(count) is averaged over PAs, not over pitches: a PA that fouls off
        # six 1-2 pitches passes through 1-2 once. Weighting by pitches would let
        # foul-prone plate appearances bend the table toward themselves.
        terminal = pa.with_columns(model._terminal_value_expr())
        reached = (
            df.join(
                terminal.select("game_pk", "at_bat_number", "terminal_value"),
                on=["game_pk", "at_bat_number"],
                how="inner",
            )
            .with_columns(_count_key(pl.col("balls"), pl.col("strikes")).alias("count_key"))
            .unique(subset=["game_pk", "at_bat_number", "count_key"])
        )
        re_tbl = reached.group_by("count_key").agg(
            pl.col("terminal_value").mean().alias("re"), pl.len().alias("n")
        )
        model.count_re = dict(
            zip(re_tbl["count_key"].to_list(), re_tbl["re"].to_list(), strict=True)
        )
        log.info(
            "run value fitted on %d plate appearances; RE(0-0)=%.4f, contact map "
            "rv = %.4f + %.4f * xwoba",
            pa.height,
            model.count_re.get("0-0", float("nan")),
            model.bip_intercept,
            model.bip_slope,
        )
        return model

    def _terminal_value_expr(self) -> pl.Expr:
        """Value of the plate appearance's outcome, contact de-noised."""
        return (
            pl.when(
                (pl.col("description") == BIP_DESCRIPTION)
                & pl.col("estimated_woba_using_speedangle").is_not_null()
            )
            .then(self.bip_intercept + self.bip_slope * pl.col("estimated_woba_using_speedangle"))
            .otherwise(pl.col("events").replace_strict(self.event_value, default=None))
            .alias("terminal_value")
        )

    # --- application ---------------------------------------------------------

    def attach(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add `rv_pitcher` (higher = better for the pitcher) to every pitch.

        Rows whose plate appearance was excluded, or whose outcome carries no
        weight, come back null rather than zero — zero is a real run value here
        and would quietly become the most common label in the training set.
        """
        pa = _plate_appearances(df).with_columns(self._terminal_value_expr())

        out = df.join(
            pa.select("game_pk", "at_bat_number", "terminal_value"),
            on=["game_pk", "at_bat_number"],
            how="left",
        ).sort(["game_pk", "at_bat_number", "pitch_number"])

        re_of = lambda e: e.replace_strict(self.count_re, default=None)  # noqa: E731
        out = out.with_columns(
            _count_key(pl.col("balls"), pl.col("strikes")).alias("_count_key"),
            _count_key(
                pl.col("balls").shift(-1).over(["game_pk", "at_bat_number"]),
                pl.col("strikes").shift(-1).over(["game_pk", "at_bat_number"]),
            ).alias("_next_count_key"),
            pl.col("pitch_number")
            .eq(pl.col("pitch_number").max().over(["game_pk", "at_bat_number"]))
            .alias("_is_last"),
        )

        rv_batting = pl.when(pl.col("_is_last")).then(pl.col("terminal_value")).otherwise(
            re_of(pl.col("_next_count_key"))
        ) - re_of(pl.col("_count_key"))
        return out.with_columns((-rv_batting).alias("rv_pitcher")).drop(
            "_count_key", "_next_count_key", "_is_last", "terminal_value"
        )

    def marginal_strike_value(self, balls: pl.Series, strikes: pl.Series) -> np.ndarray:
        """Run-expectancy swing (batting side) between a take being called a
        ball vs a strike at this count — `strike_value(count)` in the framing
        runs formula (`models/called_strike.py`): a positive number is how
        costly a strike call is to the batter here, relative to a ball call.

        Built once per `(balls, strikes)` pair rather than looked up directly
        in `count_re`, because terminal counts have no entry there: a ball at
        3-2 is a walk and a strike at *-2 is a strikeout, neither a reachable
        in-progress count. Those route to `event_value` instead — the same
        branch `attach`'s `_terminal_value_expr` takes for the last pitch of a
        plate appearance.
        """
        table: dict[str, float] = {}
        for b in range(MAX_BALLS + 1):
            for s in range(MAX_STRIKES + 1):
                ball_call = (
                    self.event_value.get("walk", float("nan"))
                    if b + 1 > MAX_BALLS
                    else self.count_re.get(f"{b + 1}-{s}", float("nan"))
                )
                strike_call = (
                    self.event_value.get("strikeout", float("nan"))
                    if s + 1 > MAX_STRIKES
                    else self.count_re.get(f"{b}-{s + 1}", float("nan"))
                )
                table[f"{b}-{s}"] = ball_call - strike_call
        key = _count_key(balls, strikes)
        return key.replace_strict(table, default=None).to_numpy().astype(float)

    # --- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "count_re": self.count_re,
            "event_value": self.event_value,
            "bip_intercept": self.bip_intercept,
            "bip_slope": self.bip_slope,
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunValue:
        return cls(**d)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> RunValue:
        return cls.from_dict(json.loads(path.read_text()))


def _plate_appearances(df: pl.DataFrame) -> pl.DataFrame:
    """One row per plate appearance: the pitch that ended it."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Run value needs columns {missing}")
    return df.filter(
        pl.col("events").is_not_null()
        & (pl.col("events") != "")
        & ~pl.col("events").is_in(EXCLUDED_EVENTS)
    ).unique(subset=["game_pk", "at_bat_number"], keep="last")
