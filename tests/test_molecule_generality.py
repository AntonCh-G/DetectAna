"""Tests that nothing in the pipeline is tied to aspirin.

Covers:
- MoleculeSpec built from a file, and the mismatch failures it produces.
- Topology and descriptors for a second molecule with a different atom count.
- A ring-less molecule: planarity feature and flag dropped, everything else works.
- Explicitly configured and explicitly disabled rings.

The aspirin smoke data is still the reference case; ethanol (9 atoms, no ring)
is built with ASE so the tests need no extra fixture files.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. MoleculeSpec
# ---------------------------------------------------------------------------

def test_spec_from_xyz_reads_the_file(ethanol_xyz):
    from detectana.io import MoleculeSpec

    spec = MoleculeSpec.from_xyz(ethanol_xyz)
    assert spec.n_atoms == 9
    assert spec.atom_types == ("C", "C", "O", "H", "H", "H", "H", "H", "H")


def test_spec_rejects_wrong_molecule(ethanol_xyz, initial_xyz):
    """Loading aspirin under an ethanol spec is a hard failure, not a warning."""
    from detectana.io import MoleculeSpec, load_single_frame

    spec = MoleculeSpec.from_xyz(ethanol_xyz)
    with pytest.raises(ValueError, match="expected 9 atoms"):
        load_single_frame(initial_xyz, spec=spec)


def test_spec_rejects_reordered_atoms(ethanol_xyz):
    """Same atom count, different element order → still a failure."""
    from ase.io import read, write

    from detectana.io import MoleculeSpec, load_single_frame

    spec = MoleculeSpec.from_xyz(ethanol_xyz)

    atoms = read(str(ethanol_xyz), index=0)
    atoms = atoms[[2, 1, 0] + list(range(3, len(atoms)))]  # swap C and O
    reordered = ethanol_xyz.parent / "reordered.xyz"
    write(str(reordered), atoms, format="xyz")

    with pytest.raises(ValueError, match="atom-type mismatch"):
        load_single_frame(reordered, spec=spec)


def test_spec_none_skips_validation(ethanol_xyz):
    """The frame that defines the molecule is loaded without a spec."""
    from detectana.io import load_single_frame

    atoms = load_single_frame(ethanol_xyz)
    assert len(atoms) == 9


def test_aspirin_spec_still_matches_the_smoke_data(initial_xyz):
    """The historical hard-coded aspirin check is preserved as ASPIRIN_SPEC."""
    from detectana.io import ASPIRIN_SPEC, load_single_frame

    atoms = load_single_frame(initial_xyz, spec=ASPIRIN_SPEC)
    assert len(atoms) == ASPIRIN_SPEC.n_atoms


# ---------------------------------------------------------------------------
# 2. Topology and descriptors for a ring-less molecule
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ethanol_topo(ethanol_xyz):
    from detectana.topology import build_topology

    return build_topology(ethanol_xyz)


def test_ringless_topology_builds(ethanol_topo):
    assert ethanol_topo.n_atoms == 9
    assert len(ethanol_topo.bonds) == 8       # 9 atoms, single connected molecule
    assert len(ethanol_topo.angles) > 0
    assert len(ethanol_topo.dihedrals) > 0


def test_ringless_topology_has_no_ring(ethanol_topo):
    assert ethanol_topo.ring_atoms == []
    assert not ethanol_topo.has_ring
    assert ethanol_topo.ring_idx is None


def test_ringless_feature_count_excludes_planarity(ethanol_topo):
    expected = (
        len(ethanol_topo.bonds)
        + len(ethanol_topo.angles)
        + 2 * len(ethanol_topo.dihedrals)
    )
    assert ethanol_topo.n_features == expected
    assert len(ethanol_topo.feature_names) == expected
    assert not any("planarity" in name for name in ethanol_topo.feature_names)


def test_ringless_descriptor_shapes_agree(ethanol_topo, ethanol_xyz):
    from detectana.descriptors import compute_descriptor, compute_descriptor_batch
    from detectana.io import load_single_frame

    positions = load_single_frame(ethanol_xyz).get_positions()

    single = compute_descriptor(positions, ethanol_topo)
    assert single.shape == (ethanol_topo.n_features,)
    assert np.all(np.isfinite(single))

    batch = compute_descriptor_batch(positions[np.newaxis], ethanol_topo)
    assert batch.shape == (1, ethanol_topo.n_features)
    np.testing.assert_allclose(batch[0], single, atol=1e-12)


def test_ringless_chemistry_flags_have_no_planarity(ethanol_topo, ethanol_xyz):
    from detectana.io import load_single_frame
    from detectana.topology import check_chemistry

    positions = load_single_frame(ethanol_xyz).get_positions()
    flags = check_chemistry(positions, ethanol_topo)

    assert flags.ring_planarity_rmsd is None
    assert flags.to_dict()["ring_planarity_rmsd_Å"] is None
    assert not flags.has_broken_bond


# ---------------------------------------------------------------------------
# 3. Vectorised and per-frame paths agree for the ringed case too
# ---------------------------------------------------------------------------

def test_aspirin_descriptor_paths_agree(topo, train_positions):
    """The vectorised batch path matches the per-frame loop, ring included."""
    from detectana.descriptors import compute_descriptor, compute_descriptor_batch

    batch = compute_descriptor_batch(train_positions[:3], topo)
    for i in range(3):
        np.testing.assert_allclose(
            batch[i], compute_descriptor(train_positions[i], topo), atol=1e-10
        )


def test_aspirin_still_carries_the_planarity_feature(topo):
    """Generalising must not silently change the aspirin feature vector."""
    assert topo.has_ring
    assert len(topo.ring_atoms) == 6
    assert topo.feature_names[-1] == topo.planarity_name
    assert topo.n_features == (
        len(topo.bonds) + len(topo.angles) + 2 * len(topo.dihedrals) + 1
    )


# ---------------------------------------------------------------------------
# 4. Explicit ring selection
# ---------------------------------------------------------------------------

def test_explicit_ring_atoms_are_used(initial_xyz):
    from detectana.topology import build_topology

    auto = build_topology(initial_xyz)
    explicit = build_topology(initial_xyz, ring_atoms=auto.ring_atoms)
    assert explicit.ring_atoms == auto.ring_atoms
    assert explicit.n_features == auto.n_features


def test_empty_ring_atoms_disables_planarity(initial_xyz):
    from detectana.topology import build_topology

    with_ring = build_topology(initial_xyz)
    without = build_topology(initial_xyz, ring_atoms=[])
    assert not without.has_ring
    assert without.n_features == with_ring.n_features - 1


def test_ring_atoms_must_form_a_closed_ring(initial_xyz):
    from detectana.topology import build_topology

    with pytest.raises(ValueError, match="not a closed ring"):
        build_topology(initial_xyz, ring_atoms=[0, 1, 2])


def test_ring_atoms_out_of_range_is_rejected(initial_xyz):
    from detectana.topology import build_topology

    with pytest.raises(ValueError, match="out of range"):
        build_topology(initial_xyz, ring_atoms=[0, 1, 2, 3, 4, 99])
