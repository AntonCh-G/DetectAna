"""End-to-end tests for the pipeline orchestrator.

The demo config is the subject: it is what CI runs and what the README tells a
reader to run, so these tests assert on the shipped defaults rather than on
invented parameters.

Covered
-------
1.  ``run_pipeline`` on ``config/demo.yaml`` — every output file, and the
    internal consistency of onset_table / frame_aggregate / bead_scores.
2.  manifest.json, including the ``molecule`` and ``onset_design`` blocks.
3.  The saved models: loadable, and re-scoring the cached descriptors with them
    reproduces bead_scores.npy exactly.
4.  The HDF5 branch of ``_process_run``, driven by a synthesised
    ``nvt_trajectory.hdf5`` (see the ``smoke_hdf5`` fixture).
5.  Descriptor-cache reuse and ``force_recompute``.
6.  The two-threshold config (bead_percentile / centroid_percentile), including
    that it reduces to the single-``percentile`` behaviour when the two agree.
7.  Multiple runs, and the failure modes: no bead files, truncated bead file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from detectana.pipeline import resolve_threshold_percentiles, run_pipeline

LINES_PER_FRAME = 23  # 1 count line + 1 comment line + 21 atom lines


def _run(cfg_template: dict, out_dir, **io_overrides) -> dict:
    """Run the pipeline on a copy of ``cfg_template`` writing to ``out_dir``."""
    cfg = copy.deepcopy(cfg_template)
    cfg["io"]["output_dir"] = str(out_dir)
    cfg["io"].update(io_overrides)
    run_pipeline(cfg)
    return cfg


def _read_outputs(out_dir, run_name: str) -> dict:
    run_dir = out_dir / run_name
    return {
        "run_dir": run_dir,
        "onset": pd.read_csv(run_dir / "onset_table.csv"),
        "agg": pd.read_csv(run_dir / "frame_aggregate.csv"),
        "bead_scores": np.load(run_dir / "bead_scores.npy"),
        "centroid_scores": np.load(run_dir / "centroid_scores.npy"),
        "manifest": json.loads((run_dir / "manifest.json").read_text()),
    }


# ---------------------------------------------------------------------------
# Module-scoped runs — the pipeline is executed once and asserted on many times
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_run(demo_config_template, tmp_path_factory):
    """One full run of config/demo.yaml. Returns (output_root, config)."""
    out = tmp_path_factory.mktemp("demo_run") / "outputs"
    cfg = _run(demo_config_template, out)
    return out, cfg


@pytest.fixture(scope="module")
def demo_outputs(demo_run):
    out, _ = demo_run
    return _read_outputs(out, "demo")


@pytest.fixture(scope="module")
def hdf5_run(demo_config_template, smoke_hdf5, tmp_path_factory):
    """One run over the synthetic HDF5 trajectory. Returns (output_root, config)."""
    cfg = copy.deepcopy(demo_config_template)
    cfg["runs"] = [{
        "name": "hdf5run",
        "initial_xyz": cfg["runs"][0]["initial_xyz"],
        "hdf5": str(smoke_hdf5),
        "timestep_fs": 0.2,
        "stride": 50,
    }]
    out = tmp_path_factory.mktemp("hdf5_run") / "outputs"
    cfg = _run(cfg, out)
    return out, cfg


# ---------------------------------------------------------------------------
# 1. Outputs of the demo run
# ---------------------------------------------------------------------------

def test_demo_run_writes_every_documented_output(demo_run):
    out, _ = demo_run
    expected = [
        "models/descriptor_pipeline.pkl",
        "models/scorer.pkl",
        "onset_summary.csv",
        "demo/onset_table.csv",
        "demo/frame_aggregate.csv",
        "demo/manifest.json",
        "demo/bead_scores.npy",
        "demo/centroid_scores.npy",
        "demo/chemistry_flags_bead00.csv",
        "demo/descriptor_cache/bead_00_descriptors.npz",
        "demo/descriptor_cache/centroid_descriptors.npz",
        "demo/plots/score_vs_time.png",
    ]
    missing = [rel for rel in expected if not (out / rel).exists()]
    assert not missing, f"pipeline did not write: {missing}"


def test_frame_aggregate_schema_and_length(demo_outputs):
    agg = demo_outputs["agg"]
    assert list(agg.columns) == [
        "step", "time_ps", "bead_max", "bead_p95", "bead_frac_ood",
        "centroid_score", "centroid_ood",
    ]
    # data/smoke ships 24 trajectory frames.
    assert len(agg) == 24
    assert agg["bead_frac_ood"].between(0.0, 1.0).all()
    assert agg["centroid_score"].gt(0.0).all()


def test_frame_aggregate_time_axis_uses_timestep_times_stride(demo_outputs, demo_run):
    _, cfg = demo_run
    run_cfg = cfg["runs"][0]
    frame_time_fs = run_cfg["timestep_fs"] * run_cfg["stride"]
    agg = demo_outputs["agg"]
    np.testing.assert_allclose(
        agg["time_ps"].to_numpy(),
        agg["step"].to_numpy() * frame_time_fs / 1000.0,
    )


def test_bead_aggregates_match_the_saved_score_matrix(demo_outputs):
    """bead_max / bead_p95 / bead_frac_ood are reductions of bead_scores.npy."""
    agg = demo_outputs["agg"]
    bead_scores = demo_outputs["bead_scores"]
    threshold = demo_outputs["manifest"]["ood_threshold"]

    assert bead_scores.shape == (1, len(agg))  # one bead in the demo data
    np.testing.assert_allclose(agg["bead_max"].to_numpy(), bead_scores.max(axis=0))
    np.testing.assert_allclose(
        agg["bead_p95"].to_numpy(), np.percentile(bead_scores, 95, axis=0)
    )
    np.testing.assert_allclose(
        agg["bead_frac_ood"].to_numpy(), (bead_scores > threshold).mean(axis=0)
    )
    np.testing.assert_allclose(
        agg["centroid_score"].to_numpy(), demo_outputs["centroid_scores"]
    )


def test_onset_table_has_one_row_per_run_with_the_full_schema(demo_outputs):
    from detectana.onset import OnsetResult

    onset = demo_outputs["onset"]
    assert len(onset) == 1
    assert onset["run"].iloc[0] == "demo"

    blank = OnsetResult(None, None, None, None, None, None, None, None)
    assert list(onset.columns) == ["run", *blank.to_dict().keys()]


def test_onset_steps_agree_with_the_frame_indices_they_name(demo_outputs):
    """Every reported step must be the step of the frame at the reported index."""
    onset = demo_outputs["onset"].iloc[0]
    steps = demo_outputs["agg"]["step"].to_numpy()

    for kind in ("first_bead_anomaly", "persistent_bead_anomaly",
                 "centroid_anomaly", "collective_anomaly"):
        frame = onset[f"{kind}_frame"]
        step = onset[f"{kind}_step"]
        if pd.isna(frame):
            assert pd.isna(step), f"{kind}: step without a frame index"
            continue
        assert steps[int(frame)] == int(step), f"{kind}: step/frame disagree"


def test_first_bead_anomaly_never_follows_the_persistent_one(demo_outputs):
    """A single flagged frame is a precondition for a flagged window."""
    onset = demo_outputs["onset"].iloc[0]
    first = onset["first_bead_anomaly_frame"]
    persistent = onset["persistent_bead_anomaly_frame"]
    if not pd.isna(persistent):
        assert not pd.isna(first)
        assert int(first) <= int(persistent)


def test_flagged_bead_index_is_a_real_bead(demo_outputs):
    onset = demo_outputs["onset"].iloc[0]
    n_beads = demo_outputs["bead_scores"].shape[0]
    idx = onset["first_anomaly_bead_idx"]
    if not pd.isna(idx):
        assert 0 <= int(idx) < n_beads


def test_onset_summary_collects_the_runs(demo_run, demo_outputs):
    out, _ = demo_run
    summary = pd.read_csv(out / "onset_summary.csv")
    assert list(summary["run"]) == ["demo"]
    pd.testing.assert_frame_equal(summary, demo_outputs["onset"])


def test_chemistry_flags_are_written_per_frame(demo_outputs):
    chem = pd.read_csv(demo_outputs["run_dir"] / "chemistry_flags_bead00.csv")
    assert len(chem) == len(demo_outputs["agg"])
    assert {"step", "broken_bond", "close_contact"} <= set(chem.columns)


# ---------------------------------------------------------------------------
# 2. Manifest
# ---------------------------------------------------------------------------

def test_manifest_records_the_run_and_the_code_version(demo_outputs):
    from detectana import __version__

    manifest = demo_outputs["manifest"]
    assert manifest["detectana_version"] == __version__
    assert manifest["run"] == "demo"
    assert manifest["timestamp"]
    assert manifest["config"]["descriptor"]["random_seed"] == 42


def test_manifest_molecule_block_describes_the_molecule_that_was_used(demo_outputs):
    molecule = demo_outputs["manifest"]["molecule"]
    assert molecule["n_atoms"] == 21
    assert len(molecule["atom_types"]) == 21
    assert "".join(molecule["atom_types"]).startswith("CCCCCCC")
    # Auto-detected six-membered carbon ring.
    assert len(molecule["ring_atoms"]) == 6
    assert molecule["initial_xyz"].endswith("initial.xyz")


def test_manifest_threshold_block_is_consistent(demo_outputs):
    manifest = demo_outputs["manifest"]
    # demo.yaml sets a single `percentile`, so both tracks share it.
    assert manifest["bead_percentile"] == manifest["centroid_percentile"] == 95.0
    assert manifest["ood_threshold"] == manifest["bead_ood_threshold"]
    assert manifest["ood_threshold"] == manifest["centroid_ood_threshold"]
    assert manifest["pca_n_components"] > 0
    assert 0.0 < manifest["pca_variance_explained"] <= 1.0
    assert manifest["n_features"] > 0


def test_manifest_onset_design_records_the_window_rule_arithmetic(demo_outputs):
    design = demo_outputs["manifest"]["onset_design"]
    n_frames = len(demo_outputs["agg"])

    assert design["false_flag_rate"] == pytest.approx(0.05)  # 95th percentile
    assert design["window_frames"] == 5
    assert design["step_frames"] == 1
    assert design["fraction_threshold"] == pytest.approx(0.20)
    # demo.yaml sets fraction_threshold, not a budget.
    assert design["derived_from_false_alarm_budget"] is None
    assert design["n_windows_tested"] == n_frames - 5 + 1
    assert 1 <= design["flags_needed_per_window"] <= design["effective_trials_per_window"]
    assert 0.0 <= design["false_alarm_probability_per_run"] <= 1.0
    assert (
        design["false_alarm_probability_per_run_nonoverlapping"]
        <= design["false_alarm_probability_per_run"]
    )
    # 'auto' means it was measured, not assumed.
    assert 0.0 <= design["frame_autocorrelation"] < 1.0


# ---------------------------------------------------------------------------
# 3. Saved models
# ---------------------------------------------------------------------------

def test_saved_models_reproduce_the_saved_scores(demo_run, demo_outputs):
    """Loading the pickles and re-scoring the cached descriptors is exact.

    This is the check that matters for anyone reusing the fitted model: the
    artefacts on disk have to be the ones that produced the numbers on disk.
    """
    from detectana.descriptors import DescriptorPipeline
    from detectana.scorer import MahalanobisScorer

    out, _ = demo_run
    pipe = DescriptorPipeline.load(out / "models" / "descriptor_pipeline.pkl")
    scorer = MahalanobisScorer.load(out / "models" / "scorer.pkl")

    cache = np.load(demo_outputs["run_dir"] / "descriptor_cache" / "bead_00_descriptors.npz")
    rescored = scorer.score(pipe.transform(cache["descriptors"]))
    np.testing.assert_array_equal(rescored, demo_outputs["bead_scores"][0])

    centroid_cache = np.load(
        demo_outputs["run_dir"] / "descriptor_cache" / "centroid_descriptors.npz"
    )
    np.testing.assert_array_equal(
        scorer.score(pipe.transform(centroid_cache["descriptors"])),
        demo_outputs["centroid_scores"],
    )


def test_saved_scorer_carries_both_track_thresholds(demo_run, demo_outputs):
    from detectana.scorer import MahalanobisScorer

    out, _ = demo_run
    scorer = MahalanobisScorer.load(out / "models" / "scorer.pkl")
    manifest = demo_outputs["manifest"]

    assert scorer.get_threshold("bead") == pytest.approx(manifest["bead_ood_threshold"])
    assert scorer.get_threshold("centroid") == pytest.approx(manifest["centroid_ood_threshold"])
    # The plain attribute still answers, and answers with the bead threshold.
    assert scorer.threshold == pytest.approx(manifest["ood_threshold"])


def test_loaded_scorer_flags_the_same_frames_as_the_aggregate_table(demo_run, demo_outputs):
    """is_ood() on a reloaded scorer must agree with centroid_ood in the CSV."""
    from detectana.scorer import MahalanobisScorer

    out, _ = demo_run
    scorer = MahalanobisScorer.load(out / "models" / "scorer.pkl")
    flags = scorer.is_ood(demo_outputs["centroid_scores"], track="centroid")
    np.testing.assert_array_equal(flags, demo_outputs["agg"]["centroid_ood"].to_numpy())


# ---------------------------------------------------------------------------
# 4. HDF5 branch
# ---------------------------------------------------------------------------

def test_hdf5_run_produces_one_score_row_per_bead(hdf5_run, smoke_hdf5):
    import h5py

    out, _ = hdf5_run
    outputs = _read_outputs(out, "hdf5run")
    with h5py.File(smoke_hdf5, "r") as fh:
        n_frames, n_beads = fh["bead_positions"].shape[:2]

    assert outputs["bead_scores"].shape == (n_beads, n_frames)
    assert outputs["centroid_scores"].shape == (n_frames,)
    assert len(outputs["agg"]) == n_frames


def test_hdf5_run_steps_are_the_frame_indices(hdf5_run):
    """The HDF5 layout carries no Step field, so steps are 0..n-1."""
    out, _ = hdf5_run
    outputs = _read_outputs(out, "hdf5run")
    np.testing.assert_array_equal(
        outputs["agg"]["step"].to_numpy(), np.arange(len(outputs["agg"]))
    )


def test_hdf5_run_caches_descriptors_for_every_bead(hdf5_run):
    out, _ = hdf5_run
    cache_dir = out / "hdf5run" / "descriptor_cache"
    bead_caches = sorted(p.name for p in cache_dir.glob("bead_*_descriptors.npz"))
    assert bead_caches == ["bead_00_descriptors.npz", "bead_01_descriptors.npz",
                           "bead_02_descriptors.npz"]
    assert (cache_dir / "centroid_descriptors.npz").exists()


def test_hdf5_bead_zero_is_the_undisplaced_trajectory(hdf5_run, demo_outputs):
    """Bead 0 of the fixture is the smoke trajectory, so it scores like the demo.

    Not bit-identical: the demo scores the trajectory as read from the XYZ file,
    while the fixture round-trips it through float64 HDF5. Same geometries, so
    the scores have to agree to floating-point tolerance.
    """
    out, _ = hdf5_run
    outputs = _read_outputs(out, "hdf5run")
    np.testing.assert_allclose(
        outputs["bead_scores"][0], demo_outputs["bead_scores"][0], rtol=1e-6
    )


def test_hdf5_run_writes_chemistry_flags_and_plots(hdf5_run):
    out, _ = hdf5_run
    assert (out / "hdf5run" / "chemistry_flags_bead00.csv").exists()
    assert (out / "hdf5run" / "plots" / "score_vs_time.png").exists()


def test_hdf5_missing_dataset_is_reported(demo_config_template, tmp_path):
    """A file without bead_positions fails in the loader, not deep in the maths."""
    import h5py

    bad = tmp_path / "bad.hdf5"
    with h5py.File(bad, "w") as fh:
        fh.create_dataset("positions", data=np.zeros((4, 21, 3)))

    cfg = copy.deepcopy(demo_config_template)
    cfg["runs"] = [{
        "name": "bad",
        "initial_xyz": cfg["runs"][0]["initial_xyz"],
        "hdf5": str(bad),
        "timestep_fs": 0.2,
        "stride": 50,
    }]
    with pytest.raises(KeyError):
        _run(cfg, tmp_path / "out")


# ---------------------------------------------------------------------------
# 5. Descriptor cache
# ---------------------------------------------------------------------------

def test_second_run_reuses_the_cache_and_gets_the_same_scores(demo_config_template, tmp_path):
    """Cache hit and cache miss must not disagree — else results depend on state."""
    out = tmp_path / "outputs"
    _run(demo_config_template, out)
    first = np.load(out / "demo" / "bead_scores.npy")
    cache = out / "demo" / "descriptor_cache" / "bead_00_descriptors.npz"
    mtime_before = cache.stat().st_mtime_ns

    _run(demo_config_template, out)  # force_recompute stays False → cache is read
    np.testing.assert_array_equal(np.load(out / "demo" / "bead_scores.npy"), first)
    assert cache.stat().st_mtime_ns == mtime_before, "cache was rewritten on a hit"


def test_force_recompute_rewrites_the_cache_without_changing_the_scores(
    demo_config_template, tmp_path
):
    out = tmp_path / "outputs"
    _run(demo_config_template, out)
    first = np.load(out / "demo" / "bead_scores.npy")
    cache = out / "demo" / "descriptor_cache" / "bead_00_descriptors.npz"
    cached_descriptors = np.load(cache)["descriptors"]

    _run(demo_config_template, out, force_recompute=True)
    np.testing.assert_array_equal(np.load(out / "demo" / "bead_scores.npy"), first)
    np.testing.assert_array_equal(np.load(cache)["descriptors"], cached_descriptors)


# ---------------------------------------------------------------------------
# 6. Threshold schema
# ---------------------------------------------------------------------------

def test_resolve_threshold_percentiles_single_value_covers_both_tracks():
    assert resolve_threshold_percentiles({"percentile": 99.0}) == (99.0, 99.0)


def test_resolve_threshold_percentiles_per_track_values():
    cfg = {"bead_percentile": 99.0, "centroid_percentile": 85.0}
    assert resolve_threshold_percentiles(cfg) == (99.0, 85.0)


def test_resolve_threshold_percentiles_falls_back_to_the_shared_value():
    cfg = {"percentile": 99.0, "centroid_percentile": 85.0}
    assert resolve_threshold_percentiles(cfg) == (99.0, 85.0)


def test_resolve_threshold_percentiles_rejects_a_config_with_no_percentile():
    with pytest.raises(KeyError, match="percentile"):
        resolve_threshold_percentiles({"bead_percentile": 99.0})


def test_split_percentiles_reduce_to_the_single_percentile_form(hdf5_config, tmp_path):
    """bead=centroid=p must give byte-identical results to percentile=p.

    This is what keeps the two-threshold feature from changing existing runs.
    """
    shared = copy.deepcopy(hdf5_config)
    shared["threshold"] = {"percentile": 95.0}
    split = copy.deepcopy(hdf5_config)
    split["threshold"] = {"bead_percentile": 95.0, "centroid_percentile": 95.0}

    out_shared = tmp_path / "shared"
    out_split = tmp_path / "split"
    _run(shared, out_shared)
    _run(split, out_split)

    for name in ("frame_aggregate.csv", "onset_table.csv"):
        assert (out_shared / "hdf5run" / name).read_bytes() == \
               (out_split / "hdf5run" / name).read_bytes(), name
    for name in ("bead_scores.npy", "centroid_scores.npy"):
        np.testing.assert_array_equal(
            np.load(out_shared / "hdf5run" / name),
            np.load(out_split / "hdf5run" / name),
        )


def test_split_percentiles_flag_the_two_tracks_differently(hdf5_config, tmp_path):
    """A strict centroid threshold must flag fewer centroid frames than a loose one."""
    loose = copy.deepcopy(hdf5_config)
    loose["threshold"] = {"bead_percentile": 50.0, "centroid_percentile": 50.0}
    strict = copy.deepcopy(hdf5_config)
    strict["threshold"] = {"bead_percentile": 50.0, "centroid_percentile": 99.0}

    _run(loose, tmp_path / "loose")
    _run(strict, tmp_path / "strict")

    loose_out = _read_outputs(tmp_path / "loose", "hdf5run")
    strict_out = _read_outputs(tmp_path / "strict", "hdf5run")

    # Same bead percentile → identical bead track.
    np.testing.assert_array_equal(loose_out["bead_scores"], strict_out["bead_scores"])
    np.testing.assert_allclose(
        loose_out["agg"]["bead_frac_ood"], strict_out["agg"]["bead_frac_ood"]
    )
    # Stricter centroid threshold → no more centroid flags than the loose one.
    assert (
        strict_out["agg"]["centroid_ood"].sum() <= loose_out["agg"]["centroid_ood"].sum()
    )
    assert (
        strict_out["manifest"]["centroid_ood_threshold"]
        > strict_out["manifest"]["bead_ood_threshold"]
    )


def test_split_percentiles_record_both_tracks_in_the_onset_design(hdf5_config, tmp_path):
    cfg = copy.deepcopy(hdf5_config)
    cfg["threshold"] = {"bead_percentile": 50.0, "centroid_percentile": 99.0}
    _run(cfg, tmp_path / "out")

    design = _read_outputs(tmp_path / "out", "hdf5run")["manifest"]["onset_design"]
    assert set(design) == {"fraction_threshold_used", "bead", "centroid"}
    assert design["bead"]["false_flag_rate"] == pytest.approx(0.50)
    assert design["centroid"]["false_flag_rate"] == pytest.approx(0.01)
    assert design["fraction_threshold_used"] == pytest.approx(
        max(design["bead"]["fraction_threshold"], design["centroid"]["fraction_threshold"])
    )


# ---------------------------------------------------------------------------
# 7. Several runs, and the failure modes
# ---------------------------------------------------------------------------

def test_two_runs_are_scored_against_one_reference_fit(demo_config_template, tmp_path):
    """Runs are the independent units, so both appear in onset_summary.csv."""
    cfg = copy.deepcopy(demo_config_template)
    second = copy.deepcopy(cfg["runs"][0])
    second["name"] = "demo_again"
    cfg["runs"].append(second)

    out = tmp_path / "outputs"
    _run(cfg, out)

    summary = pd.read_csv(out / "onset_summary.csv")
    assert list(summary["run"]) == ["demo", "demo_again"]
    # Same trajectory scored twice against the same fit → same onset.
    assert summary.drop(columns="run").iloc[0].equals(summary.drop(columns="run").iloc[1])
    np.testing.assert_array_equal(
        np.load(out / "demo" / "bead_scores.npy"),
        np.load(out / "demo_again" / "bead_scores.npy"),
    )


def test_a_second_run_with_a_different_molecule_is_rejected(
    demo_config_template, ethanol_xyz, tmp_path
):
    """The first run's initial.xyz defines the molecule; run 2 must match it."""
    cfg = copy.deepcopy(demo_config_template)
    second = copy.deepcopy(cfg["runs"][0])
    second["name"] = "ethanol"
    second["initial_xyz"] = str(ethanol_xyz)
    cfg["runs"].append(second)

    with pytest.raises(ValueError, match="atoms"):
        _run(cfg, tmp_path / "outputs")


def test_a_bead_glob_that_matches_nothing_fails_loudly(demo_config_template, tmp_path):
    cfg = copy.deepcopy(demo_config_template)
    cfg["runs"][0]["bead_glob"] = str(tmp_path / "no_such_bead_*.xyz")
    with pytest.raises(FileNotFoundError, match="No bead files matched"):
        _run(cfg, tmp_path / "outputs")


def _run_with_a_short_bead(demo_config_template, tmp_path) -> tuple[Path, int, int]:
    """Two-bead run where bead 01 stops halfway. Returns (out_dir, full, short)."""
    source = Path(demo_config_template["runs"][0]["bead_glob"]).parent / "aspirin.pos_00.xyz"
    lines = source.read_text().splitlines(keepends=True)

    bead_dir = tmp_path / "beads"
    bead_dir.mkdir()
    (bead_dir / "aspirin.pos_00.xyz").write_text("".join(lines))
    # Half the frames, cut on a frame boundary.
    n_frames = len(lines) // LINES_PER_FRAME
    n_short = n_frames // 2
    (bead_dir / "aspirin.pos_01.xyz").write_text("".join(lines[: n_short * LINES_PER_FRAME]))

    cfg = copy.deepcopy(demo_config_template)
    cfg["runs"][0]["bead_glob"] = str(bead_dir / "aspirin.pos_*.xyz")
    out = tmp_path / "outputs"
    _run(cfg, out)
    return out, n_frames, n_short


def test_a_truncated_bead_file_costs_frames_not_the_run(demo_config_template, tmp_path):
    """A short bead file trims the analysis to the common range instead of failing."""
    out, _, n_short = _run_with_a_short_bead(demo_config_template, tmp_path)
    outputs = _read_outputs(out, "demo")

    assert outputs["bead_scores"].shape == (2, n_short)
    assert len(outputs["agg"]) == n_short
    assert len(outputs["onset"]) == 1


def test_the_truncation_is_reported_in_the_manifest(demo_config_template, tmp_path):
    """The trim is provenance: how many frames were used, and what was on disk."""
    out, n_frames, n_short = _run_with_a_short_bead(demo_config_template, tmp_path)
    alignment = _read_outputs(out, "demo")["manifest"]["frame_alignment"]

    assert alignment["truncated_to_common_range"] is True
    assert alignment["n_frames_used"] == n_short
    assert alignment["frame_counts"] == [n_frames, n_short]
    assert alignment["n_frames_dropped"] == n_frames - n_short
    assert alignment["centroid_frame_count"] == n_frames


def test_the_truncation_is_warned_about(demo_config_template, tmp_path, caplog):
    """A quiet trim would read as a complete analysis, so it has to be logged."""
    with caplog.at_level("WARNING", logger="detectana.pipeline"):
        _run_with_a_short_bead(demo_config_template, tmp_path)
    assert any("frame counts differ" in rec.getMessage() for rec in caplog.records)


def test_equal_length_beads_report_no_truncation(demo_outputs):
    """The report is written on every run, not only the trimmed ones."""
    alignment = demo_outputs["manifest"]["frame_alignment"]
    assert alignment["truncated_to_common_range"] is False
    assert alignment["n_frames_dropped"] == 0
    assert alignment["n_frames_used"] == len(demo_outputs["agg"])


def test_beads_that_disagree_on_step_numbering_still_fail(demo_config_template, tmp_path):
    """Trimming cannot rescue beads whose frame *i* is a different timestep."""
    source = Path(demo_config_template["runs"][0]["bead_glob"]).parent / "aspirin.pos_00.xyz"
    lines = source.read_text().splitlines(keepends=True)

    bead_dir = tmp_path / "beads"
    bead_dir.mkdir()
    (bead_dir / "aspirin.pos_00.xyz").write_text("".join(lines))
    # The same trajectory at twice the stride: shorter *and* differently numbered.
    n_frames = len(lines) // LINES_PER_FRAME
    strided: list[str] = []
    for frame in range(0, n_frames, 2):
        strided += lines[frame * LINES_PER_FRAME:(frame + 1) * LINES_PER_FRAME]
    (bead_dir / "aspirin.pos_01.xyz").write_text("".join(strided))

    cfg = copy.deepcopy(demo_config_template)
    cfg["runs"][0]["bead_glob"] = str(bead_dir / "aspirin.pos_*.xyz")
    with pytest.raises(ValueError, match="disagrees on step numbering"):
        _run(cfg, tmp_path / "outputs")


def test_plot_failure_does_not_lose_the_results(demo_config_template, tmp_path, monkeypatch):
    """Plotting is cosmetic: a failure there must not cost the run its tables."""
    import detectana.pipeline as pipeline_mod

    def boom(*args, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(pipeline_mod, "_make_plots", boom)
    out = tmp_path / "outputs"
    _run(demo_config_template, out)

    assert (out / "demo" / "onset_table.csv").exists()
    assert not (out / "demo" / "plots" / "score_vs_time.png").exists()
