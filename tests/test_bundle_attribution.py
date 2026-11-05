"""Bundle attribution: RBC per descriptor group, for T² and SPE (ADR 0007)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from detectana.attribution import Attribution
from detectana.bundle import JOINED_TRACKS, BundleError, fit


@pytest.fixture(scope="module")
def frames(train_positions):
    return train_positions[::3].copy(), np.delete(train_positions, np.s_[::3], axis=0)


@pytest.fixture(scope="module")
def tracked(frames, topo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(*frames, topo, percentiles={"bead": 90.0})


@pytest.fixture(scope="module")
def joined(frames, topo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(*frames, topo, tracks=JOINED_TRACKS, percentiles={"bead": 90.0})


def _stretch_ch(frame, topo, delta=0.3):
    """Move the H of the first C–H bond away from its C by ``delta`` Å."""
    sym = topo.atom_types
    i, j = next((i, j) for i, j in topo.bonds if {sym[i], sym[j]} == {"C", "H"})
    h, c = (i, j) if sym[i] == "H" else (j, i)
    out = frame.copy()
    u = out[h] - out[c]
    out[h] += delta * u / np.linalg.norm(u)
    return out, (c, h)


@pytest.mark.parametrize("track", ["bonds", "dihedrals", "ring"])
def test_t2_attribution_values_are_the_score(tracked, frames, track):
    _, valid = frames
    att = tracked.attribute(valid[:6], track, score="T2")
    np.testing.assert_allclose(att.values, tracked.score(valid[:6])[track]["T2"], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("track", ["bonds", "angles", "dihedrals"])
def test_spe_attribution_values_are_the_score(tracked, frames, track):
    _, valid = frames
    att = tracked.attribute(valid[:6], track, score="SPE")
    np.testing.assert_allclose(att.values, tracked.score(valid[:6])[track]["SPE"], rtol=1e-8, atol=1e-10)


def test_attribution_has_one_strength_per_group(tracked, frames, topo):
    att = tracked.attribute(frames[1][:3], "dihedrals")
    assert isinstance(att, Attribution)
    nd = len(topo.dihedrals)
    assert att.strength.shape == (3, nd)
    assert set(att.group_sizes) == {2}  # sin and cos together
    assert att.group_atoms == tuple(tuple(d) for d in topo.dihedrals)
    assert all(g.startswith("dihedrals:dih_") for g in att.group_ids)
    assert att.atom_strength.shape == (3, topo.n_atoms)
    assert np.all(att.strength >= 0.0)


def test_atom_strength_is_the_largest_touching_group(tracked, frames):
    att = tracked.attribute(frames[1][:2], "angles")
    for atom in range(att.atom_strength.shape[1]):
        touching = [g for g, atoms in enumerate(att.group_atoms) if atom in atoms]
        expected = att.strength[:, touching].max(axis=1) if touching else 0.0
        np.testing.assert_allclose(att.atom_strength[:, atom], expected)


def test_top_lists_the_strongest_groups_first(tracked, frames):
    att = tracked.attribute(frames[1][:2], "bonds")
    top = att.top(frame=1, k=4)
    assert len(top) == 4
    strengths = [t.strength for t in top]
    assert strengths == sorted(strengths, reverse=True)
    assert strengths[0] == att.strength[1].max()


@pytest.fixture(scope="module")
def large_fit(topo):
    """1500 rattled copies of the reference geometry: enough frames per PCA
    direction for the covariance, unlike the 64-frame smoke set."""
    rng = np.random.default_rng(11)
    ref = topo.reference_positions
    sigma = np.where(np.array(topo.atom_types) == "H", 0.02, 0.01)[None, :, None]
    frames = ref[None] + rng.normal(size=(2000, topo.n_atoms, 3)) * sigma
    return fit(frames[:1500], frames[1500:], topo, percentiles={"bead": 99.0}), frames[1500:]


def test_stretched_ch_bond_is_named_by_the_score_that_flags_it(large_fit, topo):
    """The ADR 0006 case: one C–H stretched by 0.3 Å. The bonds track flags it,
    and the score with the largest exceedance names that bond as its strongest
    group. (Here that is T²: the stretch lies mostly in kept PCA directions, so
    SPE's residual reflects the other bonds too; its RBC is not asserted.)"""
    bundle, valid = large_fit
    frame, (c, h) = _stretch_ch(valid[0], topo)
    ex = bundle.exceedance(bundle.score(frame), role="bead")
    assert ex.marked[0]
    track, score = ex.tests[0]
    assert track == "bonds"
    att = bundle.attribute(frame, track, score=score)
    top, runner_up = att.top(k=2)
    assert set(top.atoms) == {c, h}, (score, top)
    assert top.strength > 5 * runner_up.strength


def test_joined_track_groups_span_every_descriptor(joined, frames, topo):
    att = joined.attribute(frames[1][:1], "geometric")
    prefixes = {g.split(":")[0] for g in att.group_ids}
    assert prefixes == {"bonds", "angles", "dihedrals", "ring"}
    n_groups = len(topo.bonds) + len(topo.angles) + len(topo.dihedrals) + 1
    assert att.strength.shape == (1, n_groups)


def test_attribution_of_an_inactive_score_is_refused(tracked, frames):
    with pytest.raises(BundleError, match="SPE.*inactive"):
        tracked.attribute(frames[1][:1], "ring", score="SPE")


def test_attribution_carries_the_thresholds(tracked, frames):
    att = tracked.attribute(frames[1][:1], "bonds", score="SPE")
    assert att.thresholds == tracked.thresholds("bonds")["SPE"]
