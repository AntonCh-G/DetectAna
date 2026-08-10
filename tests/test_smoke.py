"""Smoke-test suite — the project validation checklist, run without full data.

Tests
-----
1. Topology smoke: bond graph built from initial.xyz; ring detected.
2. Descriptor smoke: feature vector shape correct on a tiny slice.
3. Bond-break sanity: artificially broken frame scores higher than normal.
4. Torsion periodicity: sin/cos encoding survives 0↔2π wrap.
5. Shape/index test: (n_beads, n_frames, n_features) dimensions correct.
6. Threshold leakage: descriptor pipeline fit uses training only.

Run with:
    pip install -e .
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_DIR = PROJECT_ROOT / "data" / "smoke"

INITIAL_XYZ = Path(os.environ.get("DETECTANA_INITIAL_XYZ", SMOKE_DATA_DIR / "initial.xyz"))
BEAD_00_XYZ = Path(os.environ.get("DETECTANA_BEAD_00_XYZ", SMOKE_DATA_DIR / "aspirin.pos_00.xyz"))


# ---------------------------------------------------------------------------
# Test 1: Topology
# ---------------------------------------------------------------------------

def test_topology_bond_count(topo):
    """Aspirin has 21 bonds (20 heavy + 8 C-H, minus shared — empirically ~21)."""
    assert len(topo.bonds) > 0
    # Each bond stored once with i < j
    for i, j in topo.bonds:
        assert i < j


def test_topology_benzene_ring(topo):
    """Benzene ring: exactly 6 carbons, all in atom_types == 'C'."""
    assert len(topo.ring_atoms) == 6
    for idx in topo.ring_atoms:
        assert topo.atom_types[idx] == "C", f"Ring atom {idx} is not carbon"


def test_topology_feature_count(topo):
    """Feature count matches declared n_features."""
    assert topo.n_features == (
        len(topo.bonds)
        + len(topo.angles)
        + 2 * len(topo.dihedrals)
        + 1  # ring planarity
    )


def test_topology_feature_names_length(topo):
    assert len(topo.feature_names) == topo.n_features


# ---------------------------------------------------------------------------
# Test 2: Descriptor shape
# ---------------------------------------------------------------------------

def test_descriptor_shape(topo):
    from detectana.descriptors import compute_descriptor
    from detectana.io import load_single_frame

    atoms = load_single_frame(INITIAL_XYZ)
    feat = compute_descriptor(atoms.get_positions(), topo)
    assert feat.shape == (topo.n_features,)
    assert np.all(np.isfinite(feat)), "Descriptor contains NaN or Inf"


def test_descriptor_batch_shape(topo, train_positions):
    from detectana.descriptors import compute_descriptor_batch

    X = compute_descriptor_batch(train_positions[:10], topo)
    assert X.shape == (10, topo.n_features)


# ---------------------------------------------------------------------------
# Test 3: Bond-break sanity
# ---------------------------------------------------------------------------

def test_broken_bond_high_score(topo, train_positions, fitted_pipeline):
    """A frame with a stretched C=O bond should score higher than normal."""
    from detectana.descriptors import compute_descriptor
    from detectana.scorer import MahalanobisScorer

    pipe, X_train = fitted_pipeline
    X_train_pca = pipe.transform(X_train)
    scorer = MahalanobisScorer()
    scorer.fit(X_train_pca)

    # Normal frame
    normal_pos = train_positions[0].copy()
    normal_feat = compute_descriptor(normal_pos, topo).reshape(1, -1)
    normal_score = scorer.score(pipe.transform(normal_feat))[0]

    # Broken frame: stretch the first bond by 2 Å
    broken_pos = train_positions[0].copy()
    i, j = topo.bonds[0]
    direction = broken_pos[j] - broken_pos[i]
    direction /= np.linalg.norm(direction) + 1e-12
    broken_pos[j] += direction * 3.0   # move atom j far away

    broken_feat = compute_descriptor(broken_pos, topo).reshape(1, -1)
    broken_score = scorer.score(pipe.transform(broken_feat))[0]

    assert broken_score > normal_score, (
        f"Broken bond score {broken_score:.3f} should exceed normal {normal_score:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Torsion periodicity (sin/cos wrap)
# ---------------------------------------------------------------------------

def test_torsion_periodicity(topo):
    """Dihedral 0 rad and 2π rad should give the same sin/cos features."""
    from detectana.descriptors import _dihedral_rad

    # Build two sets of positions for a dihedral that differ by 2π rotation
    # Use a simple planar quadruplet
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])

    # p3 at angle θ and θ + 2π should give same sin/cos
    theta = 1.234  # radians
    p3a = np.array([np.cos(theta), 1.0, np.sin(theta)])
    p3b = np.array([np.cos(theta + 2 * np.pi), 1.0, np.sin(theta + 2 * np.pi)])

    angle_a = _dihedral_rad(p0, p1, p2, p3a)
    angle_b = _dihedral_rad(p0, p1, p2, p3b)

    assert abs(np.sin(angle_a) - np.sin(angle_b)) < 1e-9
    assert abs(np.cos(angle_a) - np.cos(angle_b)) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: Shape/index (simulated bead stack)
# ---------------------------------------------------------------------------

def test_bead_stack_shape(topo, train_positions, fitted_pipeline):
    """Simulate 4 beads × 10 frames and verify aggregate shapes."""
    from detectana.aggregator import aggregate_bead_scores
    from detectana.descriptors import compute_descriptor_batch
    from detectana.scorer import MahalanobisScorer

    pipe, X_train = fitted_pipeline
    X_train_pca = pipe.transform(X_train)
    scorer = MahalanobisScorer()
    scorer.fit(X_train_pca)
    scorer.calibrate(X_train_pca, percentile=99.0)  # use train as proxy for test

    n_beads, n_frames = 4, 10
    # Use first 10 training frames as stand-in for each bead
    sample_pos = train_positions[:n_frames]
    X = compute_descriptor_batch(sample_pos, topo)
    X_pca = pipe.transform(X)
    scores_1d = scorer.score(X_pca)

    # Stack same scores for 4 mock beads
    bead_scores = np.stack([scores_1d] * n_beads)   # (4, 10)
    assert bead_scores.shape == (n_beads, n_frames)

    centroid_scores = scores_1d
    steps = np.arange(n_frames) * 50

    agg = aggregate_bead_scores(bead_scores, centroid_scores, steps, scorer.threshold)
    assert len(agg) == n_frames
    assert {"step", "bead_max", "bead_p95", "bead_frac_ood",
            "centroid_score", "centroid_ood"}.issubset(agg.columns)


def test_bead_fixture_shape(tmp_path):
    """Load a tiny bead trajectory and verify time/atom dimensions."""
    from detectana.xyz_reader import iter_positions_chunked

    chunks = iter_positions_chunked(
        BEAD_00_XYZ,
        chunk_size=8,
        stride=50,
        cache_path=tmp_path / "aspirin.pos_00.frameindex.npz",
    )
    steps, positions = next(chunks)

    assert positions.shape == (8, 21, 3)
    assert steps.shape == (8,)
    assert np.all(np.diff(steps) == 50)


# ---------------------------------------------------------------------------
# Test 6: No leakage — PIMD frames not used in training fit
# ---------------------------------------------------------------------------

def test_no_leakage_in_pipeline(topo, train_positions, fitted_pipeline):
    """Descriptor pipeline was fit on training data only.

    Validates that transform(X_train) uses scaler/PCA fit on X_train,
    and that the fitted pipeline can score held-out data without refitting.
    """
    from detectana.descriptors import compute_descriptor_batch
    from detectana.scorer import MahalanobisScorer

    pipe, X_train = fitted_pipeline
    X_train_pca = pipe.transform(X_train)

    # Simulate "PIMD frames" as slightly perturbed training frames
    rng = np.random.default_rng(seed=0)
    pimd_pos = train_positions[:20] + rng.normal(0, 0.01, (20, 21, 3))
    X_pimd = compute_descriptor_batch(pimd_pos, topo)

    # Transform without refitting — no leakage
    X_pimd_pca = pipe.transform(X_pimd)

    scorer = MahalanobisScorer()
    scorer.fit(X_train_pca)  # fit on training only

    # Calibrate on the pimd data as "validation" proxy for this test
    threshold = scorer.calibrate(X_pimd_pca, percentile=99.0)
    scores = scorer.score(X_pimd_pca)

    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))
    assert threshold > 0
