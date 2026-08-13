"""End-to-end tests for the optional embedding OOD track.

The embedding track scores pre-computed MLFF invariant features per atom, so it
needs HDF5 files that a force-field inference run would normally produce. These
tests synthesise them: the point here is the wiring — that the track is fitted on
reference embeddings only, merged onto the right frames, and reported separately
from the geometric track — not that the features mean anything.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from detectana.pipeline import run_pipeline

N_EMB_FEATURES = 6


def _write_embeddings(path, n_frames, steps=None, n_atoms=21, seed=0, shift=0.0):
    """Write one ``inv_features`` + ``steps`` file in the documented layout."""
    import h5py

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_frames, n_atoms, N_EMB_FEATURES)) + shift
    if steps is None:
        steps = np.arange(n_frames, dtype=np.int64)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("inv_features", data=features)
        fh.create_dataset("steps", data=np.asarray(steps, dtype=np.int64))
    return path


@pytest.fixture
def embedding_files(tmp_path, smoke_hdf5):
    """Reference, per-bead and centroid embedding files for the HDF5 run."""
    import h5py

    with h5py.File(smoke_hdf5, "r") as fh:
        n_frames, n_beads = fh["bead_positions"].shape[:2]

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    _write_embeddings(emb_dir / "reference_train.h5", 64, seed=1)
    _write_embeddings(emb_dir / "reference_valid.h5", 32, seed=2)
    for bead in range(n_beads):
        # Shifted away from the reference cloud so the track actually flags frames.
        _write_embeddings(
            emb_dir / f"bead_{bead:02d}.h5", n_frames, seed=10 + bead, shift=3.0
        )
    _write_embeddings(emb_dir / "centroid.h5", n_frames, seed=99, shift=3.0)
    return emb_dir, n_frames, n_beads


@pytest.fixture
def embedding_config(demo_config_template, smoke_hdf5, embedding_files, tmp_path):
    emb_dir, _, _ = embedding_files
    cfg = copy.deepcopy(demo_config_template)
    cfg["io"]["output_dir"] = str(tmp_path / "outputs")
    cfg["runs"] = [{
        "name": "embrun",
        "initial_xyz": cfg["runs"][0]["initial_xyz"],
        "hdf5": str(smoke_hdf5),
        "timestep_fs": 0.2,
        "stride": 50,
        "embedding_glob": str(emb_dir / "bead_*.h5"),
        "centroid_embedding_h5": str(emb_dir / "centroid.h5"),
    }]
    cfg["embedding"] = {
        "enabled": True,
        "reference_train_h5": str(emb_dir / "reference_train.h5"),
        "reference_valid_h5": str(emb_dir / "reference_valid.h5"),
    }
    return cfg


def _outputs(cfg):
    from pathlib import Path

    out = Path(cfg["io"]["output_dir"])
    return out, out / cfg["runs"][0]["name"]


def test_embedding_track_adds_its_own_columns_and_files(embedding_config, embedding_files):
    _, n_frames, n_beads = embedding_files
    run_pipeline(embedding_config)
    out, run_dir = _outputs(embedding_config)

    agg = pd.read_csv(run_dir / "frame_aggregate.csv")
    assert {"emb_bead_max", "emb_bead_p95", "emb_bead_frac_ood",
            "emb_centroid_score", "emb_centroid_ood"} <= set(agg.columns)
    assert len(agg) == n_frames

    emb_bead = np.load(run_dir / "emb_bead_scores.npy")
    emb_centroid = np.load(run_dir / "emb_centroid_scores.npy")
    assert emb_bead.shape == (n_beads, n_frames)
    assert emb_centroid.shape == (n_frames,)
    np.testing.assert_allclose(agg["emb_bead_max"].to_numpy(), emb_bead.max(axis=0))
    np.testing.assert_allclose(agg["emb_centroid_score"].to_numpy(), emb_centroid)

    assert (out / "models" / "embedding_pipeline.pkl").exists()


def test_embedding_track_is_reported_separately_from_the_geometric_one(embedding_config):
    """Two tracks, two answers — the geometric onset must not absorb the embedding one."""
    run_pipeline(embedding_config)
    _, run_dir = _outputs(embedding_config)
    onset = pd.read_csv(run_dir / "onset_table.csv").iloc[0]

    assert not pd.isna(onset["embedding_persistent_bead_onset_frame"])
    assert not pd.isna(onset["embedding_centroid_onset_frame"])
    # The embedding features are synthetic noise shifted off the reference cloud,
    # so every frame is OOD on that track and onset lands on the first window.
    assert int(onset["embedding_persistent_bead_onset_frame"]) == 0


def test_embedding_threshold_comes_from_the_reference_valid_file(embedding_config):
    from detectana.embedding_scorer import EmbeddingPipeline

    run_pipeline(embedding_config)
    out, run_dir = _outputs(embedding_config)

    emb_pipe = EmbeddingPipeline.load(out / "models" / "embedding_pipeline.pkl")
    assert emb_pipe.n_atoms == 21
    assert emb_pipe.n_features == N_EMB_FEATURES
    assert emb_pipe.threshold > 0

    agg = pd.read_csv(run_dir / "frame_aggregate.csv")
    scores = np.load(run_dir / "emb_centroid_scores.npy")
    np.testing.assert_array_equal(
        agg["emb_centroid_ood"].to_numpy().astype(bool), scores > emb_pipe.threshold
    )


def test_embedding_plot_gains_its_panels(embedding_config):
    run_pipeline(embedding_config)
    _, run_dir = _outputs(embedding_config)
    plot = run_dir / "plots" / "score_vs_time.png"
    assert plot.exists() and plot.stat().st_size > 0


def test_manifest_still_describes_the_geometric_run(embedding_config):
    run_pipeline(embedding_config)
    _, run_dir = _outputs(embedding_config)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config"]["embedding"]["enabled"] is True
    assert manifest["ood_threshold"] > 0


def test_embedding_enabled_without_run_paths_skips_the_track(embedding_config):
    """A run with no embedding files is a warning, not a crash."""
    embedding_config["runs"][0].pop("embedding_glob")
    embedding_config["runs"][0].pop("centroid_embedding_h5")
    run_pipeline(embedding_config)

    _, run_dir = _outputs(embedding_config)
    agg = pd.read_csv(run_dir / "frame_aggregate.csv")
    assert "emb_bead_max" not in agg.columns
    assert not (run_dir / "emb_bead_scores.npy").exists()


def test_an_embedding_glob_that_matches_nothing_fails_loudly(embedding_config, tmp_path):
    embedding_config["runs"][0]["embedding_glob"] = str(tmp_path / "absent_*.h5")
    with pytest.raises(FileNotFoundError, match="No embedding bead files matched"):
        run_pipeline(embedding_config)


def test_mismatched_embedding_bead_lengths_trim_the_track(embedding_config, embedding_files):
    """A short embedding file costs the track frames, as on the geometric side."""
    emb_dir, n_frames, n_beads = embedding_files
    _write_embeddings(emb_dir / "bead_01.h5", n_frames - 2, seed=77, shift=3.0)

    run_pipeline(embedding_config)
    _, run_dir = _outputs(embedding_config)

    emb_bead = np.load(run_dir / "emb_bead_scores.npy")
    assert emb_bead.shape == (n_beads, n_frames - 2)

    # The geometric track keeps all its frames; the embedding columns go NaN on
    # the two frames the embedding files no longer cover.
    agg = pd.read_csv(run_dir / "frame_aggregate.csv")
    assert len(agg) == n_frames
    assert agg["emb_bead_max"].notna().sum() == n_frames - 2

    alignment = json.loads((run_dir / "manifest.json").read_text())["frame_alignment"]
    assert alignment["truncated_to_common_range"] is False  # geometric track intact
    assert alignment["embedding"]["truncated_to_common_range"] is True
    assert alignment["embedding"]["n_frames_used"] == n_frames - 2
    assert alignment["embedding"]["frame_counts"][1] == n_frames - 2


def test_embedding_frames_outside_the_trajectory_are_left_unscored(
    embedding_config, embedding_files
):
    """Embedding inference may cover a strided subset; the rest must stay NaN."""
    emb_dir, n_frames, n_beads = embedding_files
    strided = np.arange(0, n_frames, 2, dtype=np.int64)
    for bead in range(n_beads):
        _write_embeddings(
            emb_dir / f"bead_{bead:02d}.h5", len(strided), steps=strided,
            seed=30 + bead, shift=3.0,
        )
    _write_embeddings(
        emb_dir / "centroid.h5", len(strided), steps=strided, seed=31, shift=3.0
    )

    run_pipeline(embedding_config)
    _, run_dir = _outputs(embedding_config)
    agg = pd.read_csv(run_dir / "frame_aggregate.csv")

    scored = agg["emb_bead_max"].notna()
    np.testing.assert_array_equal(agg.loc[scored, "step"].to_numpy(), strided)
    assert agg["emb_bead_max"].isna().sum() == n_frames - len(strided)
