"""The train/serve parity contract.

This is the most important test in the ML package. It replays a real game one
pitch at a time through the live code path and asserts the resulting features are
identical to the batch-computed features for the same pitches.

If it fails, the live model is being fed something different from what it was
trained on — the failure mode that is nearly impossible to diagnose from
production metrics, because nothing errors and accuracy just quietly sags.
"""

from __future__ import annotations

import polars as pl
import pytest

from bbcore.config import get_settings
from bbml.features import FEATURE_NAMES, build_features
from bbml.features.context import IncrementalState, PendingPitch

# Fields describing the pre-pitch state; the live path knows all of these.
PENDING_FIELDS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "base_state",
    "bat_score",
    "fld_score",
    "stand",
    "p_throws",
    "home_team",
    "inning_topbot",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
]


def _lake_available() -> bool:
    return any((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet"))


@pytest.fixture(scope="module")
def one_pitcher_game() -> pl.DataFrame:
    """A single pitcher's pitches from one real game, chronologically ordered."""
    if not _lake_available():
        pytest.skip("no lake built — run `bb build pitches` first")
    from bbml.features.context import load_batch_frame

    df = load_batch_frame()
    if df.height == 0:
        pytest.skip("lake is empty")

    # Pick the (game, pitcher) pair with the most pitches: a starter's full outing
    # exercises multi-inning sequence and fatigue features.
    counts = df.group_by(["game_pk", "pitcher"]).len().sort("len", descending=True)
    game_pk, pitcher = counts["game_pk"][0], counts["pitcher"][0]
    return df.filter((pl.col("game_pk") == game_pk) & (pl.col("pitcher") == pitcher)).sort(
        ["at_bat_number", "pitch_number"]
    )


class TestTrainServeParity:
    def test_live_path_reproduces_batch_features_exactly(self, one_pitcher_game):
        """Replay the outing pitch by pitch and compare against batch."""
        game = one_pitcher_game
        assert game.height > 40, "need a substantial outing to be a meaningful test"

        batch = build_features(game, with_targets=False).sort(["at_bat_number", "pitch_number"])

        state = IncrementalState()
        live_rows: list[pl.DataFrame] = []

        for row in game.iter_rows(named=True):
            pending = PendingPitch(**{k: row[k] for k in PENDING_FIELDS})
            live_rows.append(state.features_for(pending))
            # Only now does the pitch become known history.
            state.record(row)

        live = pl.concat(live_rows, how="diagonal_relaxed")

        assert live.height == batch.height
        mismatches: list[str] = []
        for col in FEATURE_NAMES:
            b = batch[col].to_list()
            live_col = live[col].to_list()
            for i, (bv, lv) in enumerate(zip(b, live_col, strict=True)):
                if bv is None and lv is None:
                    continue
                if isinstance(bv, float) and isinstance(lv, float):
                    if abs(bv - lv) < 1e-9:
                        continue
                elif bv == lv:
                    continue
                mismatches.append(f"{col}[pitch {i}]: batch={bv!r} live={lv!r}")

        assert not mismatches, "train/serve drift:\n" + "\n".join(mismatches[:15])

    def test_live_path_cannot_see_the_current_pitch(self, one_pitcher_game):
        """The synthetic row carries no pitch data, so any feature that read the
        current pitch would surface as a null here rather than as free accuracy."""
        game = one_pitcher_game
        state = IncrementalState()
        row = game.row(0, named=True)
        pending = PendingPitch(**{k: row[k] for k in PENDING_FIELDS})
        feats = state.features_for(pending)

        # On the very first pitch of a PA with no history, every backward-looking
        # feature must be null — there is genuinely nothing to look back at.
        for col in ("prev_pitch_type_1", "prev_plate_x_1", "prev_velo_1"):
            assert feats[col][0] is None, f"{col} should be null on the first pitch"

    def test_priors_warm_up_as_history_accumulates(self, one_pitcher_game):
        """Expanding-window priors must be null at the start and populated later —
        the signature of a strictly-backward window."""
        game = one_pitcher_game
        batch = build_features(game, with_targets=False).sort(["at_bat_number", "pitch_number"])
        assert batch["prior_pitches_seen"][0] == 0
        assert batch["prior_usage_ff"][0] is None, "no history means no rate, not zero"
        assert batch["prior_pitches_seen"][-1] == batch.height - 1
        assert batch["prior_usage_ff"][-1] is not None
