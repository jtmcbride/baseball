"""Model tests, including the regression gate: the model must beat the baseline.

These run against whatever the local lake holds, so they are integration tests
rather than fixtures. They skip cleanly on a machine with no data.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bbcore.config import get_settings
from bbml import datasets as ds
from bbml.features.context import build_batch_features
from bbml.features.schema import FEATURE_NAMES, PITCH_TYPES
from bbml.models import NextPitchModel, UsageRateBaseline
from bbml.models.baseline import count_bucket, expected_calibration_error, log_loss
from bbml.models.personalize import PersonalizedBlend


def _lake_available() -> bool:
    return any((get_settings().lake_dir / "fact_pitch").glob("season=*/*.parquet"))


@pytest.fixture(scope="module")
def split() -> ds.Split:
    if not _lake_available():
        pytest.skip("no lake built")
    feats = ds.prepare(build_batch_features())
    if feats.height < 20_000:
        pytest.skip("not enough data for a meaningful model test")
    return ds.auto_split(feats)


@pytest.fixture(scope="module")
def trained(split: ds.Split):
    model = NextPitchModel().fit(split.train, split.val, num_boost_round=400)
    y = split.test["target_pitch_type"].to_list()
    proba = model.predict_proba(split.test)
    return model, y, proba


class TestMetrics:
    def test_log_loss_rewards_confident_correctness(self):
        classes = ["A", "B"]
        confident_right = np.array([[0.99, 0.01]])
        confident_wrong = np.array([[0.01, 0.99]])
        assert log_loss(["A"], confident_right, classes) < 0.05
        assert log_loss(["A"], confident_wrong, classes) > 4.0

    def test_ece_is_zero_for_a_perfectly_calibrated_predictor(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.5, 1.0, 4000)
        proba = np.stack([p, 1 - p], axis=1)
        y = ["A" if rng.random() < pi else "B" for pi in p]
        assert expected_calibration_error(y, proba, ["A", "B"]) < 0.05

    def test_ece_catches_overconfidence(self):
        # Always claims 99% and is right only half the time.
        n = 2000
        proba = np.tile([0.99, 0.01], (n, 1))
        y = ["A" if i % 2 == 0 else "B" for i in range(n)]
        assert expected_calibration_error(y, proba, ["A", "B"]) > 0.4

    def test_null_targets_are_excluded(self):
        proba = np.array([[0.9, 0.1], [0.9, 0.1]])
        assert log_loss(["A", None], proba, ["A", "B"]) == pytest.approx(
            log_loss(["A"], proba[:1], ["A", "B"])
        )


class TestBaseline:
    def test_count_buckets_group_behaviourally(self):
        assert count_bucket(0, 2) == count_bucket(1, 2) == "putaway"
        assert count_bucket(3, 0) == count_bucket(3, 1) == "must_strike"
        assert count_bucket(0, 0) == "first"

    def test_probabilities_are_normalized(self, split):
        bl = UsageRateBaseline().fit(split.train)
        proba = bl.predict_proba(split.test.head(500))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_unknown_pitcher_falls_back_to_league(self, split):
        bl = UsageRateBaseline().fit(split.train)
        row = split.test.head(1).with_columns(pl.lit(-999).alias("pitcher"))
        proba = bl.predict_proba(row)
        assert np.isclose(proba.sum(), 1.0)
        assert not np.isnan(proba).any()


class TestNextPitchModel:
    def test_beats_the_baseline(self, split, trained):
        """The regression gate.

        A next-pitch model that cannot beat a per-pitcher count-bucket lookup has
        learned nothing, and that is easy to ship without noticing because 45%
        top-1 accuracy sounds good in isolation. Log-loss against the baseline is
        the honest comparison, so it is asserted rather than eyeballed.
        """
        _, y, proba = trained
        baseline = UsageRateBaseline().fit(split.train)
        base_ll = log_loss(y, baseline.predict_proba(split.test), PITCH_TYPES)
        model_ll = log_loss(y, proba, PITCH_TYPES)
        improvement = (base_ll - model_ll) / base_ll
        assert model_ll < base_ll, (
            f"model log-loss {model_ll:.4f} does not beat baseline {base_ll:.4f}"
        )
        # A trivial win is also a failure — it means the features add nothing.
        assert improvement > 0.05, f"only {improvement:.1%} better than baseline"

    def test_predictions_are_valid_distributions(self, trained):
        _, _, proba = trained
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()
        assert proba.shape[1] == len(PITCH_TYPES)

    def test_is_reasonably_calibrated(self, trained):
        """The UI shows probabilities, so they have to mean what they say."""
        _, y, proba = trained
        assert expected_calibration_error(y, proba, PITCH_TYPES) < 0.05

    def test_round_trips_through_disk(self, split, trained, tmp_path):
        model, _, proba = trained
        model.save(tmp_path / "m")
        reloaded = NextPitchModel.load(tmp_path / "m")
        assert np.allclose(reloaded.predict_proba(split.test.head(200)), proba[:200])

    def test_refuses_to_load_against_a_changed_schema(self, trained, tmp_path, monkeypatch):
        """Silently scoring with mismatched features is worse than failing."""
        import json

        model, _, _ = trained
        model.save(tmp_path / "m")
        meta = json.loads((tmp_path / "m" / "meta.json").read_text())
        meta["features"] = meta["features"][:-1]
        (tmp_path / "m" / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="feature list"):
            NextPitchModel.load(tmp_path / "m")

    def test_uses_no_leaking_features(self):
        from bbml.features.schema import assert_no_leakage

        assert_no_leakage(FEATURE_NAMES)
        assert "pitch_type" not in FEATURE_NAMES
        assert "release_speed" not in FEATURE_NAMES
        # `pitcher` is excluded on purpose: personalization goes through priors so
        # the model generalizes to arms it has never seen.
        assert "pitcher" not in FEATURE_NAMES


class TestPersonalization:
    def test_shrinkage_weight_grows_with_history(self, split):
        blend = PersonalizedBlend.fit(split.train)
        rows = split.test.head(3).with_columns(pl.Series("prior_pitches_seen", [0, 1500, 100_000]))
        w = blend.weights(rows)
        assert w[0] == 0.0, "no history means no personal weight"
        assert w[1] == pytest.approx(blend.max_weight / 2, abs=1e-6)
        assert w[2] < blend.max_weight and w[2] > 0.39

    def test_blend_stays_a_distribution(self, split, trained):
        _, _, proba = trained
        blend = PersonalizedBlend.fit(split.train)
        out = blend.apply(proba, split.test)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_blend_improves_calibration(self, split, trained):
        """The reason the blend exists — it trades a little log-loss for ECE."""
        _, y, proba = trained
        blend = PersonalizedBlend.fit(split.train)
        blended = blend.apply(proba, split.test)
        assert expected_calibration_error(y, blended, PITCH_TYPES) <= expected_calibration_error(
            y, proba, PITCH_TYPES
        )
