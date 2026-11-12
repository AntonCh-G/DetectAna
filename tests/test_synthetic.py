"""Synthetic demo frames: rattle (the shipped smoke set) and harmonic normal modes."""

from __future__ import annotations

import numpy as np
import pytest

from detectana.evaluation import dihedral_angles_deg
from detectana.io import load_reference_frames
from detectana.synthetic import (
    ESTER,
    ESTER_SIGMA_DEG,
    KT_300K_EV,
    SEED_TRAIN,
    harmonic_model,
    reference_frames,
)

N_TRAIN_SMOKE = 64


@pytest.fixture(scope="module")
def model(topo):
    return harmonic_model(topo.reference_positions, topo)


def test_rattle_frames_reproduce_the_shipped_smoke_set(topo, initial_xyz):
    """Moving the generator into the library must not change data/smoke."""
    shipped, _, _ = load_reference_frames(initial_xyz.parent / "reference_train.xyz")
    regenerated = reference_frames(topo.reference_positions, topo, N_TRAIN_SMOKE, SEED_TRAIN, motion="rattle")
    np.testing.assert_allclose(regenerated, shipped, atol=1e-9)


def test_hessian_has_exactly_the_six_rigid_motions_as_zero_modes(model, topo):
    eig = np.linalg.eigvalsh(model.hessian)
    assert eig.min() > -1e-8
    n_zero = int(np.sum(eig < 1e-6 * eig.max()))
    assert n_zero == 6
    assert model.mode_amplitudes.shape == (3 * topo.n_atoms - 6,)


def test_mean_harmonic_energy_is_half_kt_per_mode(model, topo):
    """Equipartition: ⟨½ xᵀHx⟩ = (3N − 6)·kT/2 for exact thermal sampling."""
    rng = np.random.default_rng(0)
    x = model.sample_displacements(4000, rng).reshape(4000, -1)
    energy = 0.5 * np.einsum("ni,ij,nj->n", x, model.hessian, x)
    dof = 3 * topo.n_atoms - 6
    assert energy.mean() == pytest.approx(dof * KT_300K_EV / 2, rel=0.03)


def test_harmonic_frames_are_deterministic_and_keep_the_free_torsions(topo):
    a = reference_frames(topo.reference_positions, topo, 300, 5, motion="harmonic")
    b = reference_frames(topo.reference_positions, topo, 300, 5, motion="harmonic")
    np.testing.assert_array_equal(a, b)
    ester = np.deg2rad(dihedral_angles_deg(a, ESTER))
    spread = np.rad2deg(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * ester))))))  # circular std
    assert spread == pytest.approx(ESTER_SIGMA_DEG, rel=0.25)


def test_harmonic_bonds_fluctuate_by_thermal_amounts(topo):
    frames = reference_frames(topo.reference_positions, topo, 500, 6, motion="harmonic")
    i, j = topo.bonds[0]
    d = np.linalg.norm(frames[:, i] - frames[:, j], axis=1)
    assert 0.005 < d.std() < 0.06  # Å; generic force constants at 300 K


def test_unknown_motion_is_refused(topo):
    with pytest.raises(ValueError, match="motion"):
        reference_frames(topo.reference_positions, topo, 2, 0, motion="md")
