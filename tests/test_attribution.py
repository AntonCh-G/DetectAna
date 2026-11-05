"""Reconstruction-based contribution (RBC; ADR 0007), the maths alone."""

from __future__ import annotations

import numpy as np
import pytest

from detectana.attribution import atom_strength, quadratic_score, reconstruction_contributions


def test_adr_0007_example_rbc_values_overlap():
    """M = [[1,1],[1,1]], z = [1,1]: the score is 4 and each single RBC is 4.
    Attribution strengths, not shares: they do not sum to the score."""
    M = np.array([[1.0, 1.0], [1.0, 1.0]])
    z = np.array([1.0, 1.0])
    assert quadratic_score(z, M) == pytest.approx(4.0)
    rbc = reconstruction_contributions(z, M, [[0], [1]])
    np.testing.assert_allclose(rbc, [4.0, 4.0])


def _random_psd(rng, m, rank=None):
    A = rng.normal(size=(m, rank or m))
    return A @ A.T


def test_rbc_is_the_largest_score_drop_from_reconstructing_the_group():
    """Definition check: RBC_g = score − min over the group's values of the score."""
    rng = np.random.default_rng(0)
    m = 6
    M = _random_psd(rng, m)
    z = rng.normal(size=m)
    groups = [[0], [1, 2], [3, 4, 5]]
    rbc = reconstruction_contributions(z, M, groups)
    score = quadratic_score(z, M)
    for g, cols in enumerate(groups):
        # Minimise (z − ξf)ᵀ M (z − ξf) over f by least squares on M^{1/2}.
        L = np.linalg.cholesky(M)
        f, *_ = np.linalg.lstsq(L.T[:, cols], L.T @ z, rcond=None)
        z_rec = z.copy()
        z_rec[cols] -= f
        assert rbc[g] == pytest.approx(score - quadratic_score(z_rec, M), rel=1e-9)


def test_reconstructing_every_number_recovers_the_whole_score():
    rng = np.random.default_rng(1)
    M = _random_psd(rng, 5, rank=3)  # singular, like T² in standardised space
    z = rng.normal(size=5)
    rbc = reconstruction_contributions(z, M, [list(range(5))])
    assert rbc[0] == pytest.approx(quadratic_score(z, M), rel=1e-9)


def test_a_group_the_score_cannot_see_contributes_nothing():
    M = np.diag([2.0, 0.0])
    z = np.array([1.0, 5.0])
    np.testing.assert_allclose(reconstruction_contributions(z, M, [[0], [1]]), [2.0, 0.0], atol=1e-12)


def test_sin_and_cos_reconstructed_together_differ_from_either_alone():
    """A dihedral's pair is one group: its RBC is at least either single RBC."""
    rng = np.random.default_rng(2)
    M = _random_psd(rng, 4)
    z = rng.normal(size=4)
    single = reconstruction_contributions(z, M, [[0], [1]])
    pair = reconstruction_contributions(z, M, [[0, 1]])[0]
    assert pair >= single.max() - 1e-12
    assert not np.isclose(pair, single).any()


def test_batch_of_frames_gives_one_row_per_frame():
    rng = np.random.default_rng(3)
    M = _random_psd(rng, 4)
    Z = rng.normal(size=(7, 4))
    groups = [[0], [1, 2], [3]]
    batch = reconstruction_contributions(Z, M, groups)
    assert batch.shape == (7, 3)
    for n in range(7):
        np.testing.assert_allclose(batch[n], reconstruction_contributions(Z[n], M, groups))
    np.testing.assert_allclose(quadratic_score(Z, M), [quadratic_score(z, M) for z in Z])


def test_atom_strength_is_the_largest_group_value_touching_the_atom():
    strength = np.array([[40.0, 8.0, 2.0]])
    group_atoms = [(3, 13), (2, 3, 13), (0, 1)]
    out = atom_strength(strength, group_atoms, n_atoms=15)
    assert out.shape == (1, 15)
    assert out[0, 3] == out[0, 13] == 40.0
    assert out[0, 2] == 8.0
    assert out[0, 0] == out[0, 1] == 2.0
    assert out[0, 14] == 0.0


def test_empty_group_is_refused():
    with pytest.raises(ValueError, match="empty"):
        reconstruction_contributions(np.ones(2), np.eye(2), [[]])
