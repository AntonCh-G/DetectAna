"""Bundle helpers a live caller needs: chemistry checks and the role's α."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import detectana
from detectana.bundle import fit
from detectana.topology import check_chemistry_batch


@pytest.fixture(scope="module")
def bundle(train_positions, topo):
    train, valid = train_positions[::3].copy(), np.delete(train_positions, np.s_[::3], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(train, valid, topo, percentiles={"bead": 90.0, "centroid": 99.0})


def test_check_chemistry_equals_the_topology_function(bundle, train_positions, topo):
    frames = train_positions[:4].copy()
    i, j = topo.bonds[0]
    frames[2, j] = frames[2, i] + 3.0 * (frames[2, j] - frames[2, i]) / np.linalg.norm(frames[2, j] - frames[2, i])
    got = bundle.check_chemistry(frames)
    want = check_chemistry_batch(frames, topo)
    assert [g.broken_bonds for g in got] == [w.broken_bonds for w in want]
    assert [g.close_contacts for g in got] == [w.close_contacts for w in want]
    assert got[2].has_broken_bond and not got[0].has_broken_bond
    assert isinstance(got[0], detectana.ChemistryFlags)


def test_check_chemistry_takes_one_frame(bundle, train_positions):
    assert len(bundle.check_chemistry(train_positions[0])) == 1


def test_false_flag_rate_is_the_roles_alpha(bundle):
    assert bundle.false_flag_rate("bead") == pytest.approx(0.10)
    assert bundle.false_flag_rate("centroid") == pytest.approx(0.01)
    with pytest.raises(KeyError):
        bundle.false_flag_rate("nope")
