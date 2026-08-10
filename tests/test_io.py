"""Tests for the trajectory readers in ``detectana.io``.

The HDF5 loader and the full-trajectory loaders had no coverage, which matters
because they are where a wrong atom count or a mismatched centroid has to be
caught: internal coordinates are index tuples into one fixed atom ordering, so a
frame that slips through here produces descriptors that are wrong in a way
nothing downstream can notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from detectana.io import (
    ASPIRIN_SPEC,
    MoleculeSpec,
    PIMDTrajectory,
    iter_bead_positions,
    load_embeddings_h5,
    load_pimd_trajectory_hdf5,
    load_reference_frames,
    load_single_frame,
    load_trajectory_frames,
    load_trajectory_positions,
)

SMOKE_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke"
CENTROID_XYZ = SMOKE_DIR / "aspirin.xc.xyz"
BEAD_XYZ = SMOKE_DIR / "aspirin.pos_00.xyz"
TRAIN_XYZ = SMOKE_DIR / "reference_train.xyz"


def _write_hdf5(path, n_frames=4, n_beads=3, n_atoms=21, forces=False, centroid_shape=None):
    """Write a minimal trajectory file in the layout the loader documents."""
    import h5py

    rng = np.random.default_rng(0)
    bead = rng.normal(size=(n_frames, n_beads, n_atoms, 3))
    centroid = bead.mean(axis=1) if centroid_shape is None else np.zeros(centroid_shape)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("bead_positions", data=bead)
        fh.create_dataset("positions", data=centroid)
        fh.create_dataset("potential", data=np.arange(n_frames, dtype=np.float64))
        if forces:
            fh.create_dataset("bead_forces", data=np.zeros_like(bead))
            fh.create_dataset("forces", data=np.zeros((n_frames, n_atoms, 3)))
    return path


# ---------------------------------------------------------------------------
# MoleculeSpec
# ---------------------------------------------------------------------------

def test_spec_from_xyz_reads_the_first_frame(initial_xyz):
    spec = MoleculeSpec.from_xyz(initial_xyz)
    assert spec == ASPIRIN_SPEC


def test_spec_accepts_a_matching_atom_count():
    ASPIRIN_SPEC.check_n_atoms(21, source="test")  # must not raise


def test_spec_rejects_a_wrong_atom_count():
    with pytest.raises(ValueError, match="expected 21 atoms, got 9"):
        ASPIRIN_SPEC.check_n_atoms(9, source="ethanol.xyz")


def test_spec_rejects_reordered_atoms(initial_xyz):
    """Same atoms, different order: the descriptors would silently be wrong."""
    from ase.io import read

    atoms = read(str(initial_xyz), index=0)
    reordered = atoms[[*range(1, len(atoms)), 0]]
    with pytest.raises(ValueError, match="atom-type mismatch"):
        ASPIRIN_SPEC.check_atoms(reordered, frame_idx=3, source="reordered.xyz")


def test_spec_error_names_the_frame_and_the_file(ethanol_xyz):
    from ase.io import read

    atoms = read(str(ethanol_xyz), index=0)
    with pytest.raises(ValueError, match=r"\[ethanol\] Frame 7"):
        ASPIRIN_SPEC.check_atoms(atoms, frame_idx=7, source="ethanol")


# ---------------------------------------------------------------------------
# Single frame and reference sets
# ---------------------------------------------------------------------------

def test_load_single_frame_without_a_spec_defines_the_molecule(initial_xyz):
    atoms = load_single_frame(initial_xyz)
    assert len(atoms) == 21


def test_load_single_frame_checks_against_a_spec(ethanol_xyz):
    load_single_frame(ethanol_xyz, spec=MoleculeSpec.from_xyz(ethanol_xyz))
    with pytest.raises(ValueError):
        load_single_frame(ethanol_xyz, spec=ASPIRIN_SPEC)


def test_load_reference_frames_returns_positions_forces_energies():
    positions, forces, energies = load_reference_frames(TRAIN_XYZ, spec=ASPIRIN_SPEC)
    n_frames = positions.shape[0]
    assert positions.shape == (n_frames, 21, 3)
    assert forces.shape == positions.shape
    assert energies.shape == (n_frames,)
    assert np.isfinite(positions).all()


def test_load_reference_frames_tolerates_a_positions_only_file(tmp_path):
    """Forces fall back to zeros rather than failing the load."""
    from ase.io import read, write

    frames = [read(str(TRAIN_XYZ), index=0)]
    plain = tmp_path / "positions_only.xyz"
    write(str(plain), frames, format="xyz")

    positions, forces, energies = load_reference_frames(plain)
    assert positions.shape == (1, 21, 3)
    np.testing.assert_array_equal(forces, np.zeros_like(positions))
    np.testing.assert_array_equal(energies, np.zeros(1))


# ---------------------------------------------------------------------------
# Chunked bead reader
# ---------------------------------------------------------------------------

def test_iter_bead_positions_chunks_cover_every_frame():
    chunks = list(iter_bead_positions(BEAD_XYZ, chunk_size=10, spec=ASPIRIN_SPEC))
    assert len(chunks) > 1, "chunk_size=10 should split the smoke trajectory"

    steps = np.concatenate([s for s, _ in chunks])
    positions = np.concatenate([p for _, p in chunks])
    assert positions.shape == (len(steps), 21, 3)
    assert (np.diff(steps) > 0).all(), "steps must increase across chunks"

    single = list(iter_bead_positions(BEAD_XYZ, chunk_size=100_000, spec=ASPIRIN_SPEC))
    np.testing.assert_array_equal(np.concatenate([p for _, p in single]), positions)


def test_iter_bead_positions_rejects_a_different_molecule(ethanol_xyz):
    ethanol_spec = MoleculeSpec.from_xyz(ethanol_xyz)
    with pytest.raises(ValueError):
        list(iter_bead_positions(BEAD_XYZ, spec=ethanol_spec))


# ---------------------------------------------------------------------------
# Full-trajectory loaders
# ---------------------------------------------------------------------------

def test_load_trajectory_frames_reads_an_extxyz_reference_file():
    atoms_list, steps = load_trajectory_frames(TRAIN_XYZ, pimd=False, spec=ASPIRIN_SPEC)
    assert len(atoms_list) == len(steps)
    assert all(len(a) == 21 for a in atoms_list)
    # No Step field in extxyz comments → steps fall back to the frame index.
    np.testing.assert_array_equal(steps, np.arange(len(atoms_list)))


def test_load_trajectory_frames_reads_an_ipi_centroid_file():
    atoms_list, steps = load_trajectory_frames(CENTROID_XYZ, pimd=True, spec=ASPIRIN_SPEC)
    assert len(atoms_list) == len(steps)
    assert all(len(a) == 21 for a in atoms_list)
    assert (np.diff(steps) > 0).all()


def test_load_trajectory_frames_rejects_a_different_molecule(ethanol_xyz):
    with pytest.raises(ValueError):
        load_trajectory_frames(
            CENTROID_XYZ, pimd=True, spec=MoleculeSpec.from_xyz(ethanol_xyz)
        )


def test_load_trajectory_positions_agrees_with_the_frame_loader():
    """The fast binary reader and the ASE reader must return the same geometry."""
    atoms_list, _ = load_trajectory_frames(CENTROID_XYZ, pimd=True)
    fast_positions, _ = load_trajectory_positions(
        CENTROID_XYZ, pimd=True, spec=ASPIRIN_SPEC
    )
    ase_positions = np.array([a.get_positions() for a in atoms_list])
    np.testing.assert_allclose(fast_positions, ase_positions, atol=1e-8)


def test_the_two_pimd_loaders_number_frames_differently():
    """Recording a real discrepancy rather than asserting the intended behaviour.

    ``load_trajectory_positions(pimd=True)`` parses ``Step:`` out of the iPI
    comment line via the byte-offset index and returns simulation steps.
    ``load_trajectory_frames(pimd=True)`` looks for ``Step:`` in ``atoms.info``,
    but ASE's plain-xyz reader discards the comment, so it always falls back to
    the frame index. Nothing in the pipeline calls the ASE path, so this is a
    latent trap rather than a live bug — the fix belongs with whoever owns that
    loader's contract.
    """
    _, ase_steps = load_trajectory_frames(CENTROID_XYZ, pimd=True)
    _, fast_steps = load_trajectory_positions(CENTROID_XYZ, pimd=True)

    np.testing.assert_array_equal(ase_steps, np.arange(len(ase_steps)))
    assert fast_steps[0] > 0, "the fast reader should recover real iPI step numbers"
    assert not np.array_equal(ase_steps, fast_steps)


def test_load_trajectory_positions_non_pimd_branch():
    positions, steps = load_trajectory_positions(TRAIN_XYZ, pimd=False, spec=ASPIRIN_SPEC)
    assert positions.shape == (len(steps), 21, 3)
    assert steps.dtype == np.int64


def test_load_trajectory_positions_rejects_mixed_atom_counts(tmp_path):
    """One file, two molecule sizes: the index catches it before any descriptor."""
    frame_21 = BEAD_XYZ.read_text().splitlines(keepends=True)[:23]
    mixed = tmp_path / "mixed.xyz"
    mixed.write_text("".join(frame_21) + "2\n# Step: 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")

    with pytest.raises(ValueError, match="mixed atom counts"):
        load_trajectory_positions(mixed, pimd=True)


# ---------------------------------------------------------------------------
# HDF5 trajectory loader
# ---------------------------------------------------------------------------

def test_load_pimd_trajectory_hdf5_returns_the_documented_shapes(tmp_path):
    path = _write_hdf5(tmp_path / "nvt.hdf5", n_frames=5, n_beads=4)
    traj = load_pimd_trajectory_hdf5(path, spec=ASPIRIN_SPEC)

    assert isinstance(traj, PIMDTrajectory)
    assert traj.bead_positions.shape == (5, 4, 21, 3)
    assert traj.centroid_positions.shape == (5, 21, 3)
    assert traj.potential.shape == (5,)
    assert traj.bead_positions.dtype == np.float64
    # The format carries no step numbers, so the frame index is the step.
    np.testing.assert_array_equal(traj.steps, np.arange(5))
    assert traj.bead_forces is None and traj.centroid_forces is None


def test_load_pimd_trajectory_hdf5_reads_forces_when_present(tmp_path):
    path = _write_hdf5(tmp_path / "with_forces.hdf5", forces=True)
    traj = load_pimd_trajectory_hdf5(path)
    assert traj.bead_forces is not None
    assert traj.bead_forces.shape == traj.bead_positions.shape
    assert traj.centroid_forces.shape == traj.centroid_positions.shape


def test_load_pimd_trajectory_hdf5_checks_the_atom_count(tmp_path):
    path = _write_hdf5(tmp_path / "ethanol.hdf5", n_atoms=9)
    with pytest.raises(ValueError, match="expected 21 atoms, got 9"):
        load_pimd_trajectory_hdf5(path, spec=ASPIRIN_SPEC)


def test_load_pimd_trajectory_hdf5_rejects_an_inconsistent_centroid(tmp_path):
    """A centroid that does not match the beads is a corrupt file, not a warning."""
    path = _write_hdf5(
        tmp_path / "bad_centroid.hdf5", n_frames=4, centroid_shape=(3, 21, 3)
    )
    with pytest.raises(ValueError, match="centroid shape"):
        load_pimd_trajectory_hdf5(path)


def test_load_pimd_trajectory_hdf5_without_a_spec_skips_the_check(tmp_path):
    path = _write_hdf5(tmp_path / "unchecked.hdf5", n_atoms=9)
    traj = load_pimd_trajectory_hdf5(path)
    assert traj.bead_positions.shape[2] == 9


# ---------------------------------------------------------------------------
# Embedding reader
# ---------------------------------------------------------------------------

def test_load_embeddings_h5_round_trip(tmp_path):
    import h5py

    rng = np.random.default_rng(1)
    features = rng.normal(size=(6, 21, 8)).astype(np.float32)
    steps = np.arange(0, 600, 100, dtype=np.int64)
    path = tmp_path / "emb.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("inv_features", data=features)
        fh.create_dataset("steps", data=steps)

    loaded, loaded_steps = load_embeddings_h5(path)
    assert loaded.dtype == np.float64
    assert loaded_steps.dtype == np.int64
    np.testing.assert_allclose(loaded, features.astype(np.float64))
    np.testing.assert_array_equal(loaded_steps, steps)
