"""Unit tests for previously untested detectana modules.

Covers:
- topology.check_chemistry / check_chemistry_batch
- scorer.MahalanobisScorer save/load (state fields)
- descriptors.DescriptorPipeline save/load (state fields)
- aggregator.bead_score_summary / add_embedding_scores
- onset.detect_onset (exhaustive scenarios)
- embedding_scorer.EmbeddingPipeline (shapes, OOD detection, save/load)

All tests use synthetic data or the shared smoke-data fixtures from conftest.py.
No full PIMD trajectory required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. Chemistry checks
# ---------------------------------------------------------------------------

def test_check_chemistry_normal_frame(topo, train_positions):
    """Normal training frame: no broken bonds, no close contacts."""
    from detectana.topology import check_chemistry

    flags = check_chemistry(train_positions[0], topo)
    assert not flags.has_broken_bond, f"Unexpected broken bonds: {flags.broken_bonds}"
    assert not flags.has_close_contact, f"Unexpected close contacts: {flags.close_contacts}"
    assert flags.ring_planarity_rmsd >= 0.0


def test_check_chemistry_broken_bond(topo, train_positions):
    """Stretching a bond beyond cutoff flags it as broken."""
    from detectana.topology import check_chemistry

    pos = train_positions[0].copy()
    i, j = topo.bonds[0]
    direction = pos[j] - pos[i]
    direction /= np.linalg.norm(direction) + 1e-12
    pos[j] += direction * 3.0  # move atom j 3 Å further — well past 2.0 Å cutoff

    flags = check_chemistry(pos, topo, bond_break_cutoff=2.0)
    assert flags.has_broken_bond
    broken_pairs = {(a, b) for a, b, _ in flags.broken_bonds}
    assert (i, j) in broken_pairs or (j, i) in broken_pairs


def test_check_chemistry_close_contact(topo, train_positions):
    """Placing two non-bonded atoms 0.5 Å apart flags a close contact."""
    from detectana.topology import check_chemistry

    pos = train_positions[0].copy()
    bonded = set(map(frozenset, topo.bonds))

    # Find first non-bonded pair
    target_i = target_j = None
    for ii in range(topo.n_atoms):
        for jj in range(ii + 1, topo.n_atoms):
            if frozenset({ii, jj}) not in bonded:
                target_i, target_j = ii, jj
                break
        if target_i is not None:
            break

    assert target_i is not None, "Could not find a non-bonded pair in aspirin topology"

    direction = pos[target_j] - pos[target_i]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    pos[target_j] = pos[target_i] + direction * 0.5  # 0.5 Å — well below 1.2 Å cutoff

    flags = check_chemistry(pos, topo, close_contact_cutoff=1.2)
    assert flags.has_close_contact
    assert len(flags.close_contacts) >= 1


def test_check_chemistry_batch_length(topo, train_positions):
    """Batch check returns one ChemistryFlags per input frame."""
    from detectana.topology import check_chemistry_batch

    n = 5
    flags_list = check_chemistry_batch(train_positions[:n], topo)
    assert len(flags_list) == n


def test_chemistry_flags_to_dict_keys(topo, train_positions):
    """to_dict() contains the expected keys."""
    from detectana.topology import check_chemistry

    flags = check_chemistry(train_positions[0], topo)
    d = flags.to_dict()
    expected = {"broken_bond", "n_broken_bonds", "close_contact", "n_close_contacts", "ring_planarity_rmsd_Å"}
    assert expected.issubset(d.keys())


# ---------------------------------------------------------------------------
# 2. MahalanobisScorer save/load
# ---------------------------------------------------------------------------

def test_mahalanobis_scorer_save_load(tmp_path, fitted_pipeline):
    """Round-trip save/load preserves threshold and scores."""
    from detectana.scorer import MahalanobisScorer

    pipe, X_train = fitted_pipeline
    X_pca = pipe.transform(X_train)

    scorer = MahalanobisScorer()
    scorer.fit(X_pca)
    scorer.calibrate(X_pca, percentile=99.0)
    scores_before = scorer.score(X_pca)

    path = tmp_path / "scorer.pkl"
    scorer.save(path)

    scorer2 = MahalanobisScorer.load(path)
    assert scorer2.threshold == scorer.threshold
    np.testing.assert_allclose(scorer2.score(X_pca), scores_before)


# ---------------------------------------------------------------------------
# 3. DescriptorPipeline save/load
# ---------------------------------------------------------------------------

def test_descriptor_pipeline_save_load(tmp_path, fitted_pipeline):
    """Round-trip save/load preserves n_components, explained_variance_ratio, and transform output."""
    pipe, X_train = fitted_pipeline
    X_pca_before = pipe.transform(X_train)

    path = tmp_path / "pipeline.pkl"
    pipe.save(path)

    from detectana.descriptors import DescriptorPipeline
    pipe2 = DescriptorPipeline.load(path)

    assert pipe2.n_components == pipe.n_components
    np.testing.assert_allclose(pipe2.explained_variance_ratio, pipe.explained_variance_ratio)
    np.testing.assert_allclose(pipe2.transform(X_train), X_pca_before)


# ---------------------------------------------------------------------------
# 4. Aggregator helpers
# ---------------------------------------------------------------------------

def test_bead_score_summary_keys():
    """bead_score_summary returns all expected scalar keys."""
    from detectana.aggregator import bead_score_summary

    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 5, (4, 100))
    summary = bead_score_summary(scores, threshold=2.5)
    assert set(summary.keys()) == {"mean", "median", "p95", "p99", "max", "frac_ood"}


def test_bead_score_summary_frac_ood_extremes():
    """frac_ood is 1.0 when all scores exceed threshold and 0.0 when none do."""
    from detectana.aggregator import bead_score_summary

    all_above = np.ones((4, 50)) * 10.0
    all_below = np.ones((4, 50)) * 0.1

    assert bead_score_summary(all_above, threshold=5.0)["frac_ood"] == pytest.approx(1.0)
    assert bead_score_summary(all_below, threshold=5.0)["frac_ood"] == pytest.approx(0.0)


def test_add_embedding_scores_columns():
    """add_embedding_scores appends the five emb_* columns."""
    from detectana.aggregator import add_embedding_scores, aggregate_bead_scores

    n = 10
    steps = np.arange(n, dtype=np.int64) * 50
    bead_scores = np.ones((4, n))
    agg = aggregate_bead_scores(bead_scores, np.ones(n), steps, threshold=0.5)

    emb_bead = np.ones((4, n)) * 2.0
    emb_cent = np.ones(n) * 2.0
    result = add_embedding_scores(agg, emb_bead, emb_cent, steps, emb_threshold=1.0)

    expected_cols = {"emb_bead_max", "emb_bead_p95", "emb_bead_frac_ood", "emb_centroid_score", "emb_centroid_ood"}
    assert expected_cols.issubset(result.columns)


def test_add_embedding_scores_nan_for_unscored():
    """Frames without embedding coverage get NaN in emb_* columns."""
    from detectana.aggregator import add_embedding_scores, aggregate_bead_scores

    n = 5
    steps = np.array([0, 50, 100, 150, 200], dtype=np.int64)
    bead_scores = np.ones((4, n))
    agg = aggregate_bead_scores(bead_scores, np.ones(n), steps, threshold=0.5)

    emb_steps = np.array([0, 100, 200], dtype=np.int64)
    emb_bead = np.ones((4, 3)) * 2.0
    emb_cent = np.ones(3) * 2.0
    result = add_embedding_scores(agg, emb_bead, emb_cent, emb_steps, emb_threshold=1.0)

    unscored = result[result["step"].isin([50, 150])]
    assert unscored["emb_bead_max"].isna().all()
    assert unscored["emb_centroid_score"].isna().all()


# ---------------------------------------------------------------------------
# 5. Onset detection (exhaustive)
# ---------------------------------------------------------------------------

_WINDOW = 10
_STEP = 1
_FRAC = 0.5
_N = 20
_STRIDE = 50


def _make_onset_df(bead_frac: np.ndarray, centroid_ood: np.ndarray) -> pd.DataFrame:
    """Minimal DataFrame accepted by detect_onset."""
    assert len(bead_frac) == len(centroid_ood) == _N
    return pd.DataFrame({
        "step": np.arange(_N, dtype=np.int64) * _STRIDE,
        "bead_frac_ood": bead_frac.astype(np.float64),
        "centroid_ood": centroid_ood.astype(bool),
    })


def _detect(bead_frac, centroid_ood):
    from detectana.onset import detect_onset
    df = _make_onset_df(bead_frac, centroid_ood)
    return detect_onset(df, threshold=1.0, window_frames=_WINDOW, step_frames=_STEP, fraction_threshold=_FRAC)


def test_detect_onset_no_anomaly():
    """All-normal trajectory → all onset fields None."""
    result = _detect(np.zeros(_N), np.zeros(_N, dtype=bool))
    assert result.first_bead_anomaly_step is None
    assert result.persistent_bead_anomaly_step is None
    assert result.centroid_anomaly_step is None
    assert result.collective_anomaly_step is None


def test_detect_onset_transient_spike():
    """Single OOD frame doesn't trigger persistent onset (mean 1/10 < 0.5)."""
    bead_frac = np.zeros(_N)
    bead_frac[5] = 1.0  # lone spike

    result = _detect(bead_frac, np.zeros(_N, dtype=bool))

    assert result.first_bead_anomaly_frame == 5
    assert result.first_bead_anomaly_step == 5 * _STRIDE
    assert result.persistent_bead_anomaly_step is None


def test_detect_onset_persistent_bead_no_centroid():
    """Dense bead OOD block triggers persistent onset; centroid stays clean."""
    bead_frac = np.zeros(_N)
    bead_frac[:10] = 1.0  # first window fully OOD

    result = _detect(bead_frac, np.zeros(_N, dtype=bool))

    assert result.persistent_bead_anomaly_frame == 0
    assert result.persistent_bead_anomaly_step == 0
    assert result.centroid_anomaly_step is None
    assert result.collective_anomaly_step is None


def test_detect_onset_full_onset():
    """Both bead and centroid fully OOD → all four onset types triggered."""
    result = _detect(np.ones(_N), np.ones(_N, dtype=bool))

    assert result.first_bead_anomaly_frame == 0
    assert result.persistent_bead_anomaly_frame == 0
    assert result.centroid_anomaly_frame == 0
    assert result.collective_anomaly_frame == 0


def test_detect_onset_at_frame_zero():
    """Anomaly begins at frame 0: first_bead_anomaly_frame is 0."""
    bead_frac = np.zeros(_N)
    bead_frac[0] = 1.0

    result = _detect(bead_frac, np.zeros(_N, dtype=bool))

    assert result.first_bead_anomaly_frame == 0
    assert result.first_bead_anomaly_step == 0
    assert result.persistent_bead_anomaly_step is None  # 1/10 = 0.1 < 0.5


def test_detect_onset_at_last_frame():
    """Anomaly fills only the last possible window: persistent onset at frame 10.

    With window=10 and step=1, the last window start is frame 10 (frames 10-19).
    Setting bead_frac[15:]=1.0 puts exactly 5 OOD frames in window 10 (mean=0.5),
    while all earlier windows contain at most 4 OOD frames (mean<0.5).
    """
    bead_frac = np.zeros(_N)
    bead_frac[15:] = 1.0  # 5 trailing OOD frames; window [10:20] mean=0.5, all earlier <0.5

    result = _detect(bead_frac, np.zeros(_N, dtype=bool))

    assert result.persistent_bead_anomaly_frame == 10
    assert result.persistent_bead_anomaly_step == 10 * _STRIDE


def test_detect_onset_boundary_at_and_below():
    """Exactly fraction_threshold triggers onset; one fewer does not (>= comparison)."""
    # Exactly 5/10 = 0.5 = fraction_threshold → should trigger
    bead_at = np.zeros(_N)
    bead_at[:5] = 1.0
    result_at = _detect(bead_at, np.zeros(_N, dtype=bool))
    assert result_at.persistent_bead_anomaly_frame == 0

    # 4/10 = 0.4 < 0.5 → should NOT trigger
    bead_below = np.zeros(_N)
    bead_below[:4] = 1.0
    result_below = _detect(bead_below, np.zeros(_N, dtype=bool))
    assert result_below.persistent_bead_anomaly_step is None


# ---------------------------------------------------------------------------
# 6. Embedding scorer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_emb_pipeline():
    from detectana.embedding_scorer import EmbeddingPipeline

    rng = np.random.default_rng(42)
    n_train, n_atoms, n_feat = 100, 5, 8
    train_emb = rng.normal(0.0, 1.0, (n_train, n_atoms, n_feat))
    val_emb = rng.normal(0.0, 1.0, (20, n_atoms, n_feat))

    pipe = EmbeddingPipeline()
    pipe.fit(train_emb)
    pipe.calibrate(val_emb, percentile=99.0)
    return pipe, n_atoms, n_feat


def test_embedding_score_per_atom_shape(fitted_emb_pipeline):
    """score_per_atom returns (n_frames, n_atoms)."""
    pipe, n_atoms, n_feat = fitted_emb_pipeline
    rng = np.random.default_rng(1)
    emb = rng.normal(0.0, 1.0, (15, n_atoms, n_feat))
    result = pipe.score_per_atom(emb)
    assert result.shape == (15, n_atoms)


def test_embedding_score_shape(fitted_emb_pipeline):
    """score (max over atoms) returns (n_frames,)."""
    pipe, n_atoms, n_feat = fitted_emb_pipeline
    rng = np.random.default_rng(2)
    emb = rng.normal(0.0, 1.0, (15, n_atoms, n_feat))
    result = pipe.score(emb)
    assert result.shape == (15,)
    assert np.all(np.isfinite(result))


def test_embedding_ood_detection(fitted_emb_pipeline):
    """Embeddings far from training distribution are all flagged as OOD."""
    pipe, n_atoms, n_feat = fitted_emb_pipeline
    rng = np.random.default_rng(3)
    # Mean-shifted by 10σ — guaranteed to exceed threshold
    ood_emb = rng.normal(10.0, 1.0, (20, n_atoms, n_feat))
    scores = pipe.score(ood_emb)
    assert pipe.is_ood(scores).all(), "Expected all OOD frames to be flagged"


def test_embedding_pipeline_save_load(tmp_path, fitted_emb_pipeline):
    """Round-trip save/load preserves threshold, n_atoms, n_features, and scores."""
    pipe, n_atoms, n_feat = fitted_emb_pipeline
    rng = np.random.default_rng(4)
    val_emb = rng.normal(0.0, 1.0, (10, n_atoms, n_feat))
    scores_before = pipe.score(val_emb)

    path = tmp_path / "emb_pipeline.pkl"
    pipe.save(path)

    from detectana.embedding_scorer import EmbeddingPipeline
    pipe2 = EmbeddingPipeline.load(path)

    assert pipe2.threshold == pipe.threshold
    assert pipe2.n_atoms == n_atoms
    assert pipe2.n_features == n_feat
    np.testing.assert_allclose(pipe2.score(val_emb), scores_before)
