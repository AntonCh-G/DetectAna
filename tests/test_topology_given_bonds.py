"""build_topology with a bond graph the caller found (plan step 4, FFAST)."""

from __future__ import annotations

import numpy as np
import pytest

from detectana.topology import build_topology


def test_given_bonds_replace_the_radii_search_and_drive_angles_and_dihedrals(initial_xyz):
    found = build_topology(initial_xyz)
    given = build_topology(initial_xyz, bonds=[(j, i) for i, j in reversed(found.bonds)])
    assert given.bonds == found.bonds
    assert given.angles == found.angles and given.dihedrals == found.dihedrals
    assert given.ring_atoms == found.ring_atoms and given.bond_names == found.bond_names


def test_a_squeezed_frame_keeps_the_given_bonds(initial_xyz):
    # A thermal frame can push two non-bonded atoms into the radii search's range.
    from detectana.io import load_single_frame

    atoms = load_single_frame(initial_xyz)
    calm = build_topology(atoms)
    i, j = next((i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
                if (i, j) not in calm.bonds and np.linalg.norm(atoms.positions[i] - atoms.positions[j]) < 2.6)
    squeezed = atoms.copy()
    mid = (squeezed.positions[i] + squeezed.positions[j]) / 2
    for k in (i, j):
        u = squeezed.positions[k] - mid
        squeezed.positions[k] = mid + 0.8 * u / np.linalg.norm(u)
    assert (i, j) in build_topology(squeezed).bonds
    assert (i, j) not in build_topology(squeezed, bonds=calm.bonds).bonds


@pytest.mark.parametrize("bad", [[(0, 0)], [(0, 99)], [(-1, 2)]])
def test_bonds_that_are_not_atom_pairs_are_refused(bad, initial_xyz):
    with pytest.raises(ValueError, match="not pairs"):
        build_topology(initial_xyz, bonds=bad)
