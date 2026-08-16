"""Data quality checks that fail the pipeline.

The failure mode this exists to catch is not a crash — it is a backfill that
completes, looks healthy, and is quietly wrong. Silent Savant truncation, a
handedness flip that inverts every movement plot, an ID join that drops players:
none of these raise. They just produce plausible numbers that are not true.

Severity split:
  ERROR — structurally broken. Fails the run.
  WARN  — expected in some seasons (bat tracking is null before 2024) or merely
          worth eyeballing. Reported, does not fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from bbcore.config import Settings, get_settings
from bbcore.logging import get_logger
from bbcore.storage import Warehouse, open_warehouse

log = get_logger(__name__)

Severity = Literal["error", "warn"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    detail: str
    rows: list[dict] = field(default_factory=list)


@dataclass
class QualityReport:
    results: list[CheckResult]

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.failed if r.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _check(
    wh: Warehouse, name: str, sql: str, *, severity: Severity, expect_empty: bool, detail: str
) -> CheckResult:
    """A check passes when its query returns no rows (or rows, if inverted)."""
    df = pl.from_arrow(wh.execute(sql))
    assert isinstance(df, pl.DataFrame)
    passed = (df.height == 0) if expect_empty else (df.height > 0)
    return CheckResult(
        name=name,
        passed=passed,
        severity=severity,
        detail=detail if not passed else "ok",
        rows=df.head(10).to_dicts() if not passed else [],
    )


def run_checks(*, settings: Settings | None = None) -> QualityReport:
    s = settings or get_settings()
    results: list[CheckResult] = []

    with open_warehouse(settings=s) as wh:
        # --- structural ------------------------------------------------------
        results.append(
            _check(
                wh,
                "fact_pitch.primary_key_unique",
                """
                SELECT game_pk, at_bat_number, pitch_number, count(*) n
                FROM fact_pitch GROUP BY 1,2,3 HAVING count(*) > 1
                """,
                severity="error",
                expect_empty=True,
                detail="Duplicate pitches — the (game_pk, at_bat_number, pitch_number) key is violated.",
            )
        )

        results.append(
            _check(
                wh,
                "fact_pitch.not_empty",
                "SELECT 1 FROM fact_pitch LIMIT 1",
                severity="error",
                expect_empty=False,
                detail="fact_pitch is empty.",
            )
        )

        # --- Savant truncation canary ---------------------------------------
        # A day at or above the cap means the response was truncated and the
        # subdivision guard did not fire. Anything over ~6k is also suspicious:
        # a full MLB slate runs ~4,400 pitches.
        results.append(
            _check(
                wh,
                "fact_pitch.no_truncated_days",
                """
                SELECT game_date, count(*) n FROM fact_pitch
                GROUP BY 1 HAVING count(*) >= 24900
                """,
                severity="error",
                expect_empty=True,
                detail="A game date sits at the Savant 25k row cap — data was silently truncated.",
            )
        )

        # A game with an implausible pitch count usually means a partial fetch.
        results.append(
            _check(
                wh,
                "fact_pitch.plausible_game_pitch_counts",
                """
                SELECT game_pk, count(*) n FROM fact_pitch
                WHERE game_type = 'R'
                GROUP BY 1 HAVING count(*) < 100 OR count(*) > 800
                """,
                severity="warn",
                expect_empty=True,
                detail="Games with <100 or >800 pitches — suspended, partial, or partially fetched.",
            )
        )

        # --- physical ranges -------------------------------------------------
        # Bounds are physical, not stylistic. The floor has to accommodate
        # position players mopping up in blowouts — a catcher lobbing 35mph in the
        # 8th is real data, and an earlier 40mph floor flagged it as corruption.
        # Below ~25mph is tracking error; above ~108 exceeds the fastest pitch
        # ever recorded.
        results.append(
            _check(
                wh,
                "fact_pitch.velocity_physically_possible",
                """
                SELECT game_pk, at_bat_number, pitch_number, release_speed
                FROM fact_pitch
                WHERE is_tracked_pitch
                  AND release_speed IS NOT NULL
                  AND (release_speed < 25 OR release_speed > 108)
                """,
                severity="error",
                expect_empty=True,
                detail="Release speeds outside 25-108 mph — physically impossible, so tracking error.",
            )
        )

        # The blind spot this suite had until 2026-08-16: velocity was range
        # checked and location never was, so on the full lake 10 of the 11
        # physically impossible records got through — including a pitch recorded
        # as crossing the plate 35 feet wide and 57 feet underground. Those feed
        # the location model's target and Location+ directly.
        #
        # Both checks are scoped to `is_tracked_pitch` on purpose: the transform
        # quarantines impossible records behind that flag, and this is what
        # verifies the quarantine actually held.
        results.append(
            _check(
                wh,
                "fact_pitch.location_physically_possible",
                """
                SELECT game_pk, at_bat_number, pitch_number, plate_x, plate_z,
                       release_pos_x, release_pos_z
                FROM fact_pitch
                WHERE is_tracked_pitch
                  AND (abs(plate_x) > 15 OR plate_z < -10 OR plate_z > 20
                       OR abs(release_pos_x) > 8
                       OR release_pos_z <= 0 OR release_pos_z > 10)
                """,
                severity="error",
                expect_empty=True,
                detail=(
                    "Plate or release coordinates off the field entirely — tracking error. "
                    "Bounds are loose on purpose: an intentional ball really does cross 11ft "
                    "wide, so this is impossibility, not implausibility."
                ),
            )
        )

        # Visibility for the quarantine itself. It is silent by construction —
        # the row survives, it just stops being tracked — so a feed change that
        # started quarantining thousands would otherwise look like nothing.
        results.append(
            _check(
                wh,
                "fact_pitch.quarantine_rate",
                """
                SELECT season, count(*) AS quarantined
                FROM fact_pitch
                WHERE NOT is_tracked_pitch
                  AND description NOT IN ('automatic_ball', 'automatic_strike')
                GROUP BY 1 HAVING count(*) > 100
                """,
                severity="warn",
                expect_empty=True,
                detail="Seasons quarantining >100 pitches as impossible tracking — check the feed.",
            )
        )

        # Separately: an actual pitcher throwing under 55mph is worth a look, but
        # it is a curiosity (a true eephus exists), not a data defect.
        if wh.table_exists("dim_player"):
            results.append(
                _check(
                    wh,
                    "fact_pitch.true_pitcher_velocity",
                    """
                    SELECT f.game_pk, p.full_name, min(f.release_speed) slowest
                    FROM fact_pitch f JOIN dim_player p ON p.mlbam_id = f.pitcher
                    WHERE p.primary_position = 'P' AND f.release_speed < 55
                    GROUP BY 1, 2
                    """,
                    severity="warn",
                    expect_empty=True,
                    detail="Rostered pitchers throwing under 55mph — usually a genuine eephus.",
                )
            )

        results.append(
            _check(
                wh,
                "fact_pitch.strike_zone_sane",
                """
                SELECT game_pk, at_bat_number, pitch_number, sz_top, sz_bot
                FROM fact_pitch
                WHERE sz_top IS NOT NULL AND sz_bot IS NOT NULL
                  AND (sz_top <= sz_bot OR sz_top > 5 OR sz_bot < 0.5)
                """,
                severity="error",
                expect_empty=True,
                detail="Inverted or implausible strike zone bounds — plate_z_norm would be garbage.",
            )
        )

        # --- derived-column correctness --------------------------------------
        # The handedness flip is the highest-consequence transform in the
        # pipeline and fails silently: if it inverts, every movement plot and
        # every cross-handedness comparison is wrong but still looks plausible.
        # Both hands must show POSITIVE mean arm-side run on a sinker.
        results.append(
            _check(
                wh,
                "fact_pitch.arm_side_normalization",
                """
                SELECT p_throws, avg(hb_arm_in) mean_arm_break
                FROM fact_pitch
                WHERE pitch_type = 'SI' AND is_tracked_pitch
                GROUP BY 1 HAVING avg(hb_arm_in) <= 0
                """,
                severity="error",
                expect_empty=True,
                detail="Sinkers must show positive arm-side break for BOTH hands; the flip is inverted.",
            )
        )

        results.append(
            _check(
                wh,
                "fact_pitch.zone_rate_plausible",
                """
                SELECT round(100.0*sum(is_in_zone::INT)/count(*), 1) zone_pct
                FROM fact_pitch WHERE is_tracked_pitch
                HAVING zone_pct < 35 OR zone_pct > 60
                """,
                severity="error",
                expect_empty=True,
                detail="League zone rate outside 35-60% — plate_x/plate_z or the zone test is wrong.",
            )
        )

        results.append(
            _check(
                wh,
                "fact_pitch.csw_plausible",
                """
                SELECT round(100.0*sum(is_csw::INT)/count(*), 1) csw
                FROM fact_pitch WHERE is_tracked_pitch
                HAVING csw < 22 OR csw > 32
                """,
                severity="warn",
                expect_empty=True,
                detail="League CSW% outside 22-32% — check the description vocabulary.",
            )
        )

        # --- referential integrity -------------------------------------------
        if wh.table_exists("dim_player"):
            results.append(
                _check(
                    wh,
                    "fact_pitch.pitchers_mapped",
                    """
                    SELECT DISTINCT f.pitcher
                    FROM fact_pitch f LEFT JOIN dim_player p ON p.mlbam_id = f.pitcher
                    WHERE p.mlbam_id IS NULL AND f.pitcher IS NOT NULL
                    """,
                    severity="error",
                    expect_empty=True,
                    detail="Pitchers in fact_pitch missing from dim_player — rows would be dropped by joins.",
                )
            )
            results.append(
                _check(
                    wh,
                    "fact_pitch.batters_mapped",
                    """
                    SELECT DISTINCT f.batter
                    FROM fact_pitch f LEFT JOIN dim_player p ON p.mlbam_id = f.batter
                    WHERE p.mlbam_id IS NULL AND f.batter IS NOT NULL
                    """,
                    severity="error",
                    expect_empty=True,
                    detail="Batters in fact_pitch missing from dim_player.",
                )
            )

        if wh.table_exists("dim_game"):
            results.append(
                _check(
                    wh,
                    "fact_pitch.games_mapped",
                    """
                    SELECT DISTINCT f.game_pk
                    FROM fact_pitch f LEFT JOIN dim_game g USING (game_pk)
                    WHERE g.game_pk IS NULL
                    """,
                    severity="warn",
                    expect_empty=True,
                    detail="Games in fact_pitch missing from dim_game — widen the `bb ingest dims` range.",
                )
            )

    return QualityReport(results)


def season_coverage(*, settings: Settings | None = None) -> pl.DataFrame:
    """Per-season null rates for the columns whose availability changes over time.

    Availability is discovered from the data rather than hardcoded, so a column
    that starts or stops being published shows up here instead of silently
    becoming null in a model's feature vector.
    """
    s = settings or get_settings()
    cols = [
        "release_spin_rate",
        "spin_axis",
        "arm_angle",
        "launch_speed",
        "estimated_woba_using_speedangle",
        "bat_speed",
        "swing_length",
        "attack_angle",
        "swing_path_tilt",
        "delta_run_exp",
    ]
    exprs = ",\n".join(f"round(100.0 * count({c}) / count(*), 1) AS {c}_pct" for c in cols)
    with open_warehouse(settings=s) as wh:
        df = pl.from_arrow(
            wh.execute(
                f"SELECT season, count(*) AS pitches,\n{exprs}\n"
                "FROM fact_pitch GROUP BY season ORDER BY season"
            )
        )
    assert isinstance(df, pl.DataFrame)
    return df
