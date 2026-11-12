"""Tracks, SPE, the false-alarm budget and exceedance (ADR 0006, 0007)."""

from __future__ import annotations

import sys
import textwrap
import uuid
import warnings

import numpy as np
import pytest
from ase.io import read

from detectana.bundle import DEFAULT_TRACKS, BundleError, fit, load
from detectana.descriptors import compute_descriptor_batch
from detectana.topology import build_topology

PERCENTILES = {"bead": 90.0, "centroid": 80.0}


@pytest.fixture(scope="module")
def split(train_positions):
    """Train/valid from the 96 smoke frames, interleaved."""
    return train_positions[::3].copy(), np.delete(train_positions, np.s_[::3], axis=0)


@pytest.fixture(scope="module")
def tracked(split, topo):
    train, valid = split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit(train, valid, topo, percentiles=PERCENTILES, window_frames=5, step_frames=1)


def test_default_is_four_tracks_one_descriptor_each(tracked):
    assert DEFAULT_TRACKS == {"bonds": ("bonds",), "angles": ("angles",),
                              "dihedrals": ("dihedrals",), "ring": ("ring",)}
    assert tracked.tracks == ("bonds", "angles", "dihedrals", "ring")
    for t in tracked.tracks:
        assert tracked.descriptors(t) == (t,)


def test_ring_track_has_t2_only_and_the_rest_have_both(tracked):
    tests = set(tracked.tests)
    assert ("ring", "T2") in tests and ("ring", "SPE") not in tests
    for t in ("bonds", "angles", "dihedrals"):
        assert {(t, "T2"), (t, "SPE")} <= tests
    assert tracked.manifest["budget"]["n_tests"] == len(tests) == 7


def test_each_test_gets_alpha_over_k(tracked, split, topo):
    """Bonferroni: each test's threshold is the 100(1 − α/K) percentile of its
    own validation values."""
    _, valid = split
    scores = tracked.score(valid)
    k = len(tracked.tests)
    for role, p in PERCENTILES.items():
        p_test = 100.0 * (1.0 - (1.0 - p / 100.0) / k)
        for track, score in tracked.tests:
            assert tracked.thresholds(track)[score][role] == pytest.approx(
                np.percentile(scores[track][score], p_test))
        assert tracked.manifest["budget"]["roles"][role]["percentile_per_test"] == pytest.approx(p_test)


def test_score_returns_every_active_score(tracked, split):
    _, valid = split
    scores = tracked.score(valid[:5])
    assert set(scores) == set(tracked.tracks)
    assert set(scores["ring"]) == {"T2"}
    assert set(scores["bonds"]) == {"T2", "SPE"}
    assert all(v.shape == (5,) for s in scores.values() for v in s.values())


def test_spe_is_the_squared_pca_residual(split, topo):
    """SPE equals the sklearn reconstruction error in standardised units."""
    from detectana.descriptors import DescriptorPipeline

    train, valid = split
    b = fit(train, valid, topo, tracks={"joined": ["bonds", "angles", "dihedrals", "ring"]},
            percentiles={"bead": 50.0})
    X_train = compute_descriptor_batch(train, topo)
    X = compute_descriptor_batch(valid, topo)
    pipe = DescriptorPipeline(0.95, 42).fit(X_train)
    Xs = pipe._scaler.transform(X)
    resid = Xs - pipe._pca.inverse_transform(pipe._pca.transform(Xs))
    np.testing.assert_allclose(b.score(valid)["joined"]["SPE"], np.sum(resid**2, axis=1), rtol=1e-8)


def test_exceedance_is_the_largest_ratio_and_names_its_test(tracked, split):
    _, valid = split
    scores = tracked.score(valid)
    ex = tracked.exceedance(scores, role="bead")
    ratios = {t: scores[t[0]][t[1]] / tracked.thresholds(t[0])[t[1]]["bead"] for t in tracked.tests}
    stacked = np.stack(list(ratios.values()))
    np.testing.assert_allclose(ex.values, stacked.max(axis=0))
    for n, test in enumerate(ex.tests):
        assert ratios[test][n] == ex.values[n]
    # Marked = some test above its threshold.
    np.testing.assert_array_equal(ex.marked, (stacked > 1.0).any(axis=0))


def test_exceedance_refuses_an_unknown_role(tracked, split):
    with pytest.raises(KeyError, match="median"):
        tracked.exceedance(tracked.score(split[1][:2]), role="median")


def test_round_trip_keeps_every_track_and_score(tracked, split, tmp_path):
    _, valid = split
    tracked.save(tmp_path / "b")
    loaded = load(tmp_path / "b")
    a, b = tracked.score(valid), loaded.score(valid)
    for track in tracked.tracks:
        assert a[track].keys() == b[track].keys()
        for score in a[track]:
            assert np.array_equal(a[track][score], b[track][score])
    assert loaded.manifest == tracked.manifest


def test_too_few_validation_frames_per_test_warns(split, topo):
    train, valid = split
    with pytest.warns(UserWarning, match="expected exceedances"):
        fit(train, valid, topo, percentiles={"bead": 99.0})


def test_ringless_molecule_drops_the_ring_track(ethanol_xyz):
    ethanol = build_topology(ethanol_xyz)
    base = read(str(ethanol_xyz)).get_positions()
    rng = np.random.default_rng(0)
    frames = base[None] + rng.normal(scale=0.03, size=(150, len(base), 3))
    b = fit(frames[:100], frames[100:], ethanol, percentiles={"bead": 80.0})
    assert "ring" not in b.tracks
    assert b.manifest["dropped_tracks"] == {"ring": "its descriptors give no numbers for this molecule"}


@pytest.mark.parametrize("tracks, match", [
    ({}, "no tracks"),
    ({"bad/name": ["bonds"]}, "track name"),
    ({"bonds": []}, "non-empty"),
    ({"bonds": "bonds"}, "non-empty"),
])
def test_bad_track_specs_are_refused(split, topo, tracks, match):
    with pytest.raises(ValueError, match=match):
        fit(*split, topo, tracks=tracks)


def test_unknown_score_is_refused(split, topo):
    with pytest.raises(ValueError, match="scores"):
        fit(*split, topo, scores=("T2", "Q"))


# ---------------------------------------------------------------------------
# SPE with no residual variance (two numbers tied exactly)
# ---------------------------------------------------------------------------

TIED = '''
import numpy as np
from detectana.registry import DescriptorValues, descriptor


@descriptor("{name}")
def tied(positions, topology):
    d = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1)
    return DescriptorValues(np.stack([d, 2.0 * d + 1.0], axis=1), ("d", "d2"), ((0, 1), (0, 1)), ("d", "d"))
'''


def test_spe_with_no_residual_variance_is_inactive(split, topo, tmp_path, monkeypatch):
    tag = uuid.uuid4().hex[:8]
    module = f"tied_{tag}"
    (tmp_path / f"{module}.py").write_text(textwrap.dedent(TIED.format(name=f"tied_{tag}")))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        b = fit(*split, topo, tracks={"tied": [f"{module}:tied"]}, percentiles={"bead": 80.0})
    finally:
        sys.modules.pop(module, None)
    spe = b.manifest["tracks"]["tied"]["scores"]["SPE"]
    assert spe == {"active": False, "reason": "no residual variance in validation"}
    assert b.tests == (("tied", "T2"),)


def test_bundle_error_on_tampered_budget(tracked, tmp_path):
    import json

    tracked.save(tmp_path / "b")
    m = json.loads((tmp_path / "b" / "manifest.json").read_text())
    m["budget"]["tests"].pop()
    (tmp_path / "b" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="budget"):
        load(tmp_path / "b")
