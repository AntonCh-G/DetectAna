"""Tests for named threshold tracks on ``MahalanobisScorer``.

Calibrating used to overwrite one mutable ``self.threshold``, so a run that
calibrated a bead threshold and then a centroid threshold saved a pickle holding
only the second one. Scoring used the returned floats, so the tables were right,
but anything that loaded the pickle and called ``is_ood()`` silently used the
wrong threshold. These tests pin the fix.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from detectana.scorer import MahalanobisScorer


@pytest.fixture
def fitted_scorer():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 5))
    return MahalanobisScorer().fit(X), X


def test_calibrating_a_second_track_leaves_the_first_alone(fitted_scorer):
    scorer, X = fitted_scorer
    bead = scorer.calibrate(X, percentile=99.0, track="bead")
    centroid = scorer.calibrate(X, percentile=85.0, track="centroid")

    assert bead > centroid
    assert scorer.get_threshold("bead") == bead
    assert scorer.get_threshold("centroid") == centroid


def test_is_ood_uses_the_track_it_is_asked_for(fitted_scorer):
    scorer, X = fitted_scorer
    scorer.calibrate(X, percentile=99.0, track="bead")
    scorer.calibrate(X, percentile=50.0, track="centroid")
    scores = scorer.score(X)

    strict = scorer.is_ood(scores, track="bead")
    loose = scorer.is_ood(scores, track="centroid")
    assert strict.sum() < loose.sum()
    np.testing.assert_array_equal(strict, scores > scorer.get_threshold("bead"))


def test_an_uncalibrated_track_is_an_error_not_a_default(fitted_scorer):
    scorer, X = fitted_scorer
    scorer.calibrate(X, percentile=99.0, track="bead")
    with pytest.raises(RuntimeError, match="centroid"):
        scorer.is_ood(scorer.score(X), track="centroid")


def test_saving_keeps_every_track(fitted_scorer, tmp_path):
    scorer, X = fitted_scorer
    bead = scorer.calibrate(X, percentile=99.0, track="bead")
    centroid = scorer.calibrate(X, percentile=85.0, track="centroid")

    path = tmp_path / "scorer.pkl"
    scorer.save(path)
    loaded = MahalanobisScorer.load(path)

    assert loaded.get_threshold("bead") == bead
    assert loaded.get_threshold("centroid") == centroid
    np.testing.assert_array_equal(loaded.score(X), scorer.score(X))


def test_the_default_track_still_reads_as_a_plain_attribute(fitted_scorer, tmp_path):
    scorer, X = fitted_scorer
    threshold = scorer.calibrate(X, percentile=99.0)
    assert scorer.threshold == threshold

    scorer.threshold = 1.5
    assert scorer.get_threshold() == 1.5
    np.testing.assert_array_equal(scorer.is_ood(scorer.score(X)), scorer.score(X) > 1.5)

    path = tmp_path / "scorer.pkl"
    scorer.save(path)
    assert MahalanobisScorer.load(path).threshold == 1.5


def test_an_uncalibrated_scorer_has_no_threshold(fitted_scorer):
    scorer, X = fitted_scorer
    assert scorer.threshold is None
    with pytest.raises(RuntimeError, match="not calibrated"):
        scorer.is_ood(scorer.score(X))


def test_a_pickle_written_before_named_tracks_still_loads(fitted_scorer, tmp_path):
    """Old ``scorer.pkl`` files carry a single ``threshold`` key."""
    scorer, X = fitted_scorer
    legacy_state = {
        "mean": scorer._mean,
        "inv_cov": scorer._inv_cov,
        "threshold": 3.25,
    }
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as fh:
        pickle.dump(legacy_state, fh)

    loaded = MahalanobisScorer.load(path)
    assert loaded.threshold == 3.25
    assert loaded.get_threshold() == 3.25
    np.testing.assert_array_equal(loaded.score(X), scorer.score(X))


def test_a_scorer_instance_pickled_before_named_tracks_still_unpickles(fitted_scorer):
    """EmbeddingPipeline pickles whole scorer objects, so instance state matters."""
    scorer, X = fitted_scorer
    legacy = MahalanobisScorer.__new__(MahalanobisScorer)
    legacy.__dict__.update(
        {"_mean": scorer._mean, "_inv_cov": scorer._inv_cov, "threshold": 2.5}
    )

    revived = pickle.loads(pickle.dumps(legacy))
    assert revived.threshold == 2.5
    np.testing.assert_array_equal(revived.score(X), scorer.score(X))
