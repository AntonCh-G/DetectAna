"""Reference bundle: fit once, save, load, score — the library interface of ADR 0005."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import detectana
from detectana.bundle import BUNDLE_FORMAT_VERSION, BundleError, ReferenceBundle, fit, load
from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.io import load_reference_frames
from detectana.scorer import MahalanobisScorer
from detectana.topology import build_topology

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke"
INITIAL_XYZ = SMOKE / "initial.xyz"
TRAIN_XYZ = SMOKE / "reference_train.xyz"
VALID_XYZ = SMOKE / "reference_valid.xyz"
BEAD_00_XYZ = SMOKE / "aspirin.pos_00.xyz"


@pytest.fixture(scope="module")
def reference():
    train, _, _ = load_reference_frames(TRAIN_XYZ)
    valid, _, _ = load_reference_frames(VALID_XYZ)
    return train, valid


@pytest.fixture(scope="module")
def aspirin_topo():
    return build_topology(INITIAL_XYZ)


@pytest.fixture(scope="module")
def bundle(reference, aspirin_topo):
    train, valid = reference
    return fit(
        train, valid, aspirin_topo,
        pca_variance=0.95, random_seed=42,
        percentiles={"bead": 95.0, "centroid": 90.0},
        window_frames=20, step_frames=5, fraction_threshold=0.2,
    )


@pytest.fixture(scope="module")
def trajectory():
    pos, _, _ = load_reference_frames(BEAD_00_XYZ)
    return pos


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip_gives_identical_scores(bundle, trajectory, tmp_path):
    path = tmp_path / "aspirin.bundle"
    bundle.save(path)
    loaded = load(path)

    before = bundle.score(trajectory)
    after = loaded.score(trajectory)
    assert before.keys() == after.keys() == {"geometric"}
    assert np.array_equal(before["geometric"], after["geometric"])

    assert loaded.thresholds("geometric") == bundle.thresholds("geometric")
    assert loaded.elements == bundle.elements
    assert loaded.feature_names("geometric") == bundle.feature_names("geometric")
    assert loaded.onset_rule == bundle.onset_rule
    assert loaded.manifest == bundle.manifest


def test_saved_bundle_has_a_readable_manifest_and_no_pickle(bundle, tmp_path):
    path = tmp_path / "aspirin.bundle"
    bundle.save(path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["format"] == "detectana-reference-bundle"
    assert manifest["format_version"] == BUNDLE_FORMAT_VERSION
    assert manifest["statistic"] == "T2"
    assert manifest["detectana_version"] == detectana.__version__
    assert manifest["units"] == {"length": "angstrom", "angle": "radian"}
    track = manifest["tracks"]["geometric"]
    assert track["descriptor"]["name"] == "internal_coordinates"
    assert len(track["descriptor"]["code_sha256"]) == 64
    # Fitted arrays load without pickle.
    with np.load(path / "state.npz", allow_pickle=False) as state:
        assert set(state.files) == set(track["state_keys"])


def test_top_level_names(bundle, tmp_path):
    assert detectana.fit is fit
    assert detectana.load is load
    bundle.save(tmp_path / "b")
    assert isinstance(detectana.load(tmp_path / "b"), ReferenceBundle)


# ---------------------------------------------------------------------------
# Metadata API
# ---------------------------------------------------------------------------

def test_metadata_describes_the_molecule_and_rule(bundle, aspirin_topo, reference):
    assert bundle.format_version == BUNDLE_FORMAT_VERSION
    assert bundle.detectana_version == detectana.__version__
    assert bundle.elements == tuple(aspirin_topo.atom_types)
    assert bundle.n_atoms == aspirin_topo.n_atoms == 21
    assert bundle.length_unit == "angstrom"
    assert bundle.tracks == ("geometric",)
    assert bundle.feature_names("geometric") == tuple(aspirin_topo.feature_names)
    assert bundle.topology["bonds"] == [list(b) for b in aspirin_topo.bonds]
    assert bundle.topology["ring_atoms"] == list(aspirin_topo.ring_atoms)
    assert bundle.onset_rule == detectana.onset.WindowRule(20, 5, 0.2)
    assert bundle.history_frames == 20
    thr = bundle.thresholds("geometric")
    assert set(thr) == {"bead", "centroid"}
    assert thr["centroid"] <= thr["bead"]
    assert bundle.manifest["fit_data"] == {
        "n_train": len(reference[0]), "n_valid": len(reference[1]),
    }


def test_metadata_is_a_copy(bundle):
    bundle.topology["bonds"].clear()
    bundle.manifest["tracks"].clear()
    assert bundle.topology["bonds"]
    assert bundle.tracks == ("geometric",)


def test_unknown_track_is_refused(bundle):
    with pytest.raises(KeyError, match="bonds"):
        bundle.thresholds("bonds")


# ---------------------------------------------------------------------------
# Statistic
# ---------------------------------------------------------------------------

def test_score_is_t2_of_the_existing_scorer(bundle, reference, aspirin_topo, trajectory):
    """Same fit through DescriptorPipeline + MahalanobisScorer gives the same T²."""
    train, _ = reference
    pipe = DescriptorPipeline(pca_variance=0.95, random_seed=42).fit(
        compute_descriptor_batch(train, aspirin_topo)
    )
    scorer = MahalanobisScorer().fit(pipe.transform(compute_descriptor_batch(train, aspirin_topo)))
    expected = scorer.score_t2(pipe.transform(compute_descriptor_batch(trajectory, aspirin_topo)))
    np.testing.assert_allclose(bundle.score(trajectory)["geometric"], expected, rtol=1e-6)


def test_thresholds_are_t2_percentiles_of_validation(bundle, reference):
    _, valid = reference
    t2 = bundle.score(valid)["geometric"]
    thr = bundle.thresholds("geometric")
    assert thr["bead"] == pytest.approx(np.percentile(t2, 95.0))
    assert thr["centroid"] == pytest.approx(np.percentile(t2, 90.0))


def test_single_frame_scores_as_a_batch_of_one(bundle, trajectory):
    one = bundle.score(trajectory[3])["geometric"]
    many = bundle.score(trajectory)["geometric"]
    assert one.shape == (1,)
    np.testing.assert_allclose(one[0], many[3], rtol=1e-12)


# ---------------------------------------------------------------------------
# One-number input (issue 02)
# ---------------------------------------------------------------------------

def test_one_feature_bundle_fits_saves_loads_and_scores(tmp_path):
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

    def frames(n, spread):
        out = np.repeat(base[None], n, axis=0)
        out[:, 1, 0] += rng.normal(scale=spread, size=n)
        return out

    topo = build_topology(Atoms("H2", positions=base))
    b = fit(frames(200, 0.02), frames(100, 0.02), topo,
            window_frames=5, step_frames=1, fraction_threshold=0.4)
    assert b.feature_names("geometric") == ("bond_H0-H1_Å",)
    assert b.manifest["tracks"]["geometric"]["n_components"] == 1

    stretched = frames(10, 0.0)
    stretched[:, 1, 0] = 1.2
    t2 = b.score(stretched)["geometric"]
    assert np.all(t2 > b.thresholds("geometric")["bead"])

    b.save(tmp_path / "h2")
    assert np.array_equal(load(tmp_path / "h2").score(stretched)["geometric"], t2)


# ---------------------------------------------------------------------------
# Fit inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["train", "valid"])
def test_fit_refuses_empty_input(reference, aspirin_topo, which):
    train, valid = reference
    empty = train[:0]
    args = (empty, valid) if which == "train" else (train, empty)
    with pytest.raises(ValueError, match=which):
        fit(*args, aspirin_topo)


def test_fit_refuses_frames_of_the_wrong_molecule(reference, aspirin_topo):
    train, valid = reference
    with pytest.raises(ValueError, match="atoms"):
        fit(train[:, :20], valid, aspirin_topo)


def test_fit_refuses_unknown_length_unit(reference, aspirin_topo):
    train, valid = reference
    with pytest.raises(ValueError, match="bohr"):
        fit(train, valid, aspirin_topo, length_unit="bohr")


# ---------------------------------------------------------------------------
# Refusals: the bundle does not score what it cannot vouch for
# ---------------------------------------------------------------------------

def _saved(bundle, tmp_path):
    path = tmp_path / "aspirin.bundle"
    bundle.save(path)
    return path


def _edit_manifest(path, edit):
    manifest = json.loads((path / "manifest.json").read_text())
    edit(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("version", [0, BUNDLE_FORMAT_VERSION + 1])
def test_load_refuses_another_format_version(bundle, tmp_path, version):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m.update(format_version=version))
    with pytest.raises(BundleError, match="format version"):
        load(path)


def test_load_refuses_a_file_that_is_not_a_bundle(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m.update(format="something-else"))
    with pytest.raises(BundleError, match="not a DetectAna reference bundle"):
        load(path)


def test_load_refuses_distance_based_results(bundle, tmp_path):
    """Old √T² values and new T² values are never mixed (issue 03)."""
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m.update(statistic="distance"))
    with pytest.raises(BundleError, match="statistic"):
        load(path)


def test_load_refuses_changed_descriptor_code(bundle, tmp_path):
    path = _saved(bundle, tmp_path)

    def edit(m):
        m["tracks"]["geometric"]["descriptor"]["code_sha256"] = "0" * 64
    _edit_manifest(path, edit)
    with pytest.raises(BundleError, match="descriptor code"):
        load(path)


def test_load_refuses_when_installed_descriptor_code_changed(bundle, tmp_path, monkeypatch):
    path = _saved(bundle, tmp_path)
    monkeypatch.setattr("detectana.bundle.descriptor_code_hash", lambda: "f" * 64)
    with pytest.raises(BundleError, match="descriptor code"):
        load(path)


def test_load_refuses_unknown_descriptor(bundle, tmp_path):
    path = _saved(bundle, tmp_path)

    def edit(m):
        m["tracks"]["geometric"]["descriptor"]["name"] = "soap"
    _edit_manifest(path, edit)
    with pytest.raises(BundleError, match="soap"):
        load(path)


@pytest.mark.parametrize("units", [
    {"length": "bohr", "angle": "radian"},
    {"length": "angstrom", "angle": "degree"},
    {"length": "angstrom"},
])
def test_load_refuses_other_units(bundle, tmp_path, units):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m.update(units=units))
    with pytest.raises(BundleError, match="unit"):
        load(path)


def test_load_refuses_elements_that_do_not_match_atom_count(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m["molecule"]["elements"].pop())
    with pytest.raises(BundleError, match="elements"):
        load(path)


def test_load_refuses_feature_names_that_do_not_match_topology(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m["tracks"]["geometric"]["feature_names"].pop())
    with pytest.raises(BundleError, match="feature"):
        load(path)


def test_load_refuses_atom_index_out_of_range(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    _edit_manifest(path, lambda m: m["molecule"]["topology"]["bonds"].append([0, 21]))
    with pytest.raises(BundleError, match="atom index"):
        load(path)


def test_load_refuses_missing_state_array(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    with np.load(path / "state.npz") as npz:
        state = {k: npz[k] for k in npz.files if not k.endswith("t2_inv_cov")}
    np.savez(path / "state.npz", **state)
    with pytest.raises(BundleError, match="t2_inv_cov"):
        load(path)


def test_load_refuses_state_of_the_wrong_shape(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    with np.load(path / "state.npz") as npz:
        state = {k: npz[k] for k in npz.files}
    state["geometric/pca_components"] = state["geometric/pca_components"][:, :-1]
    np.savez(path / "state.npz", **state)
    with pytest.raises(BundleError, match="shape"):
        load(path)


def test_load_refuses_a_missing_file(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    (path / "state.npz").unlink()
    with pytest.raises(BundleError, match="state.npz"):
        load(path)


def test_load_refuses_unreadable_manifest(bundle, tmp_path):
    path = _saved(bundle, tmp_path)
    (path / "manifest.json").write_text("{not json")
    with pytest.raises(BundleError, match="manifest"):
        load(path)


# ── frames ──────────────────────────────────────────────────────────────────

def test_validate_frame_accepts_the_reference_order(bundle, reference):
    bundle.validate_frame(bundle.elements, reference[0][0])


def test_validate_frame_refuses_swapped_atoms(bundle, reference):
    elements = list(bundle.elements)
    i = elements.index("C")
    j = elements.index("H")
    elements[i], elements[j] = elements[j], elements[i]
    with pytest.raises(BundleError, match="atom order"):
        bundle.validate_frame(elements, reference[0][0])


def test_validate_frame_refuses_wrong_atom_count(bundle, reference):
    with pytest.raises(BundleError, match="atoms"):
        bundle.validate_frame(bundle.elements[:-1], reference[0][0][:-1])


def test_validate_frame_refuses_wrong_unit(bundle, reference):
    with pytest.raises(BundleError, match="bohr"):
        bundle.validate_frame(bundle.elements, reference[0][0], length_unit="bohr")


def test_validate_frame_refuses_non_finite_positions(bundle, reference):
    pos = reference[0][0].copy()
    pos[4, 1] = np.nan
    with pytest.raises(BundleError, match="NaN"):
        bundle.validate_frame(bundle.elements, pos)


def test_score_refuses_wrong_unit(bundle, reference):
    with pytest.raises(BundleError, match="bohr"):
        bundle.score(reference[0], length_unit="bohr")


def test_score_refuses_wrong_shape(bundle, reference):
    with pytest.raises(BundleError, match="atoms"):
        bundle.score(reference[0][:, :20])


def test_bundle_error_is_a_value_error():
    assert issubclass(BundleError, ValueError)


# ---------------------------------------------------------------------------
# Precomputed descriptors (the batch pipeline caches them)
# ---------------------------------------------------------------------------

def test_score_features_equals_score(bundle, trajectory, aspirin_topo):
    X = compute_descriptor_batch(trajectory, aspirin_topo)
    assert np.array_equal(bundle.score_features(X)["geometric"], bundle.score(trajectory)["geometric"])


def test_score_features_refuses_wrong_width(bundle, trajectory, aspirin_topo):
    X = compute_descriptor_batch(trajectory, aspirin_topo)
    with pytest.raises(BundleError, match="134"):
        bundle.score_features(X[:, :-1])
