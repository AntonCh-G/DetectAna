"""The embedding track scores T² per atom, like the geometric track (issue 03)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from detectana.embedding_scorer import EmbeddingPipeline
from detectana.scorer import MahalanobisScorer


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(11)
    return rng.normal(size=(300, 4, 5)), rng.normal(size=(100, 4, 5))


def test_per_atom_score_is_t2(embeddings):
    train, valid = embeddings
    pipe = EmbeddingPipeline().fit(train)
    per_atom = pipe.score_per_atom(valid)
    for i in range(train.shape[1]):
        expected = MahalanobisScorer().fit(train[:, i, :]).score_t2(valid[:, i, :])
        np.testing.assert_allclose(per_atom[:, i], expected, rtol=1e-12)
    np.testing.assert_allclose(pipe.score(valid), per_atom.max(axis=1))


def test_threshold_is_a_t2_percentile(embeddings):
    train, valid = embeddings
    pipe = EmbeddingPipeline().fit(train)
    thr = pipe.calibrate(valid, percentile=90.0)
    assert thr == pytest.approx(np.percentile(pipe.score(valid), 90.0))


def test_saved_pipeline_records_the_statistic(embeddings, tmp_path):
    train, valid = embeddings
    pipe = EmbeddingPipeline().fit(train)
    pipe.calibrate(valid)
    pipe.save(tmp_path / "emb.pkl")
    with open(tmp_path / "emb.pkl", "rb") as fh:
        assert pickle.load(fh)["statistic"] == "T2"
    loaded = EmbeddingPipeline.load(tmp_path / "emb.pkl")
    assert loaded.threshold == pipe.threshold
    np.testing.assert_array_equal(loaded.score(valid), pipe.score(valid))


def test_distance_based_pickle_is_refused(embeddings, tmp_path):
    """A pickle written before T² holds a distance threshold; mixing it with T² scores is wrong."""
    train, valid = embeddings
    pipe = EmbeddingPipeline().fit(train)
    pipe.calibrate(valid)
    pipe.save(tmp_path / "emb.pkl")
    with open(tmp_path / "emb.pkl", "rb") as fh:
        state = pickle.load(fh)
    del state["statistic"]
    with open(tmp_path / "emb.pkl", "wb") as fh:
        pickle.dump(state, fh)
    with pytest.raises(ValueError, match="distance"):
        EmbeddingPipeline.load(tmp_path / "emb.pkl")
