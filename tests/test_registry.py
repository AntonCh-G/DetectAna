"""Descriptor registry (ADR 0006): named numbers with atoms and group IDs."""

from __future__ import annotations

import numpy as np
import pytest

from detectana.descriptors import compute_descriptor_batch
from detectana.registry import (
    BUILTIN_DESCRIPTORS,
    DescriptorValues,
    compute,
    descriptor,
    get_descriptor,
)
from detectana.topology import build_topology


def test_builtins_are_registered():
    assert BUILTIN_DESCRIPTORS == ("bonds", "angles", "dihedrals", "ring")
    for name in BUILTIN_DESCRIPTORS:
        assert get_descriptor(name).name == name


def test_builtins_side_by_side_equal_the_joined_descriptor(topo, train_positions):
    """Same numbers, same order, bit for bit: the joined track is today's descriptor."""
    blocks = [compute(get_descriptor(n), train_positions, topo) for n in BUILTIN_DESCRIPTORS]
    joined = np.concatenate([b.values for b in blocks], axis=1)
    assert np.array_equal(joined, compute_descriptor_batch(train_positions, topo))
    assert [name for b in blocks for name in b.names] == topo.feature_names


def test_bond_numbers_name_their_atoms(topo, train_positions):
    out = compute(get_descriptor("bonds"), train_positions[:3], topo)
    assert out.values.shape == (3, len(topo.bonds))
    assert out.atoms == tuple(tuple(b) for b in topo.bonds)
    assert len(set(out.groups)) == len(topo.bonds)


def test_a_dihedrals_sin_and_cos_share_one_group(topo, train_positions):
    out = compute(get_descriptor("dihedrals"), train_positions[:2], topo)
    nd = len(topo.dihedrals)
    assert out.values.shape == (2, 2 * nd)
    assert out.groups[:nd] == out.groups[nd:]
    assert len(set(out.groups)) == nd
    assert out.atoms[:nd] == out.atoms[nd:] == tuple(tuple(d) for d in topo.dihedrals)


def test_ring_group_names_the_ring_atoms(topo, train_positions):
    out = compute(get_descriptor("ring"), train_positions[:2], topo)
    assert out.values.shape == (2, 1)
    assert out.atoms == (tuple(topo.ring_atoms),)


def test_ring_is_empty_for_a_ringless_molecule(ethanol_xyz):
    from ase.io import read

    ethanol = build_topology(ethanol_xyz)
    pos = read(str(ethanol_xyz)).get_positions()[None]
    out = compute(get_descriptor("ring"), pos, ethanol)
    assert out.values.shape == (1, 0)
    assert out.names == out.atoms == out.groups == ()


def test_unknown_descriptor_is_refused():
    with pytest.raises(KeyError, match="no_such"):
        get_descriptor("no_such")


def test_a_second_function_under_a_taken_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        @descriptor("bonds")
        def other_bonds(positions, topology):  # pragma: no cover - never called
            raise AssertionError


def _registered(fn_name, fn):
    """Register ``fn`` under a unique test name and return its registry entry."""
    descriptor(fn_name)(fn)
    return get_descriptor(fn_name)


@pytest.mark.parametrize(
    ("label", "make", "match"),
    [
        ("wrong_rows", lambda n: DescriptorValues(np.zeros((n + 1, 1)), ("a",), ((0,),), ("g",)), "frames"),
        ("wrong_names", lambda n: DescriptorValues(np.zeros((n, 2)), ("a",), ((0,), (1,)), ("g", "h")), "names"),
        ("nan", lambda n: DescriptorValues(np.full((n, 1), np.nan), ("a",), ((0,),), ("g",)), "finite"),
        ("bad_atom", lambda n: DescriptorValues(np.zeros((n, 1)), ("a",), ((99,),), ("g",)), "atom"),
        ("dup_name", lambda n: DescriptorValues(np.zeros((n, 2)), ("a", "a"), ((0,), (1,)), ("g", "h")), "unique"),
        ("no_atoms", lambda n: DescriptorValues(np.zeros((n, 1)), ("a",), ((),), ("g",)), "atom"),
    ],
)
def test_malformed_descriptor_output_is_refused(topo, train_positions, label, make, match):
    reg = _registered(f"_test_bad_{label}", lambda positions, topology: make(len(positions)))
    with pytest.raises(ValueError, match=match):
        compute(reg, train_positions[:2], topo)


def test_code_hash_is_stable_and_differs_between_descriptors():
    hashes = {n: get_descriptor(n).code_hash() for n in BUILTIN_DESCRIPTORS}
    assert len(set(hashes.values())) == len(hashes)
    assert hashes["bonds"] == get_descriptor("bonds").code_hash()
    assert all(len(h) == 64 for h in hashes.values())
