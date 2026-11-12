"""Runs read from any file ASE reads (``trajectory:``), e.g. ``.traj`` or extxyz.

The smoke bead file is converted with ASE, so the same frames go through the
i-PI XYZ path and the ASE path, and the scores must agree.
"""

from __future__ import annotations

import copy
import glob
import logging

import ase.io
import numpy as np
import pandas as pd
import pytest

from detectana.io import MoleculeSpec, load_ase_trajectory, load_trajectory_positions
from detectana.pipeline import check_run_source, run_pipeline


@pytest.fixture(scope="module")
def bead_xyz(demo_config_template) -> str:
    """The demo run's single i-PI bead file."""
    (path,) = glob.glob(demo_config_template["runs"][0]["bead_glob"])
    return path


@pytest.fixture(scope="module", params=["traj", "extxyz"])
def converted(request, bead_xyz, tmp_path_factory):
    """The bead file written by ASE as ``.traj`` or extxyz."""
    path = tmp_path_factory.mktemp("ase") / f"run.{request.param}"
    ase.io.write(path, ase.io.read(bead_xyz, ":", format="xyz"))
    return path


@pytest.fixture(scope="module")
def spec(initial_xyz) -> MoleculeSpec:
    return MoleculeSpec.from_xyz(initial_xyz)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_loader_gives_the_positions_of_the_ipi_reader(converted, bead_xyz, spec):
    positions, steps = load_ase_trajectory(converted, spec=spec)
    expected, _ = load_trajectory_positions(bead_xyz, pimd=True)
    # .traj stores the floats themselves; ASE writes extxyz with 8 decimals.
    atol = 0.0 if converted.suffix == ".traj" else 1e-8
    np.testing.assert_allclose(positions, expected, rtol=0, atol=atol)
    np.testing.assert_array_equal(steps, np.arange(len(expected)))
    assert positions.dtype == np.float64 and steps.dtype == np.int64


def test_loader_refuses_reordered_atoms(bead_xyz, spec, tmp_path):
    frames = ase.io.read(bead_xyz, ":", format="xyz")
    symbols = frames[3].get_chemical_symbols()
    j = next(j for j, s in enumerate(symbols) if s != symbols[0])
    order = list(range(len(symbols)))
    order[0], order[j] = j, 0
    frames[3] = frames[3][order]
    path = tmp_path / "swapped.traj"
    ase.io.write(path, frames)
    with pytest.raises(ValueError, match="Frame 3"):
        load_ase_trajectory(path, spec=spec)


def test_loader_warns_that_periodic_frames_are_not_unwrapped(bead_xyz, tmp_path, caplog):
    frames = ase.io.read(bead_xyz, ":2", format="xyz")
    for atoms in frames:
        atoms.set_cell([30.0, 30.0, 30.0])
        atoms.pbc = True
    path = tmp_path / "box.traj"
    ase.io.write(path, frames)
    with caplog.at_level(logging.WARNING, logger="detectana.io"):
        load_ase_trajectory(path)
    assert "not unwrapped" in caplog.text


def test_loader_refuses_an_empty_file(tmp_path):
    path = tmp_path / "empty.extxyz"
    path.write_text("")
    with pytest.raises(ValueError, match="No frames"):
        load_ase_trajectory(path, format="extxyz")


# ---------------------------------------------------------------------------
# Run source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "run, source",
    [
        ({"bead_glob": "b*", "centroid_xyz": "c"}, "bead_glob"),
        ({"hdf5": "h"}, "hdf5"),
        ({"trajectory": "t"}, "trajectory"),
    ],
)
def test_a_run_names_one_source(run, source):
    assert check_run_source({"name": "r", **run}) == source


@pytest.mark.parametrize(
    "run",
    [{}, {"hdf5": "h", "trajectory": "t"}, {"bead_glob": "b*"}],
)
def test_a_run_with_no_source_two_sources_or_no_centroid_is_refused(run):
    with pytest.raises(ValueError, match="r"):
        check_run_source({"name": "r", **run})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run(cfg: dict, out) -> None:
    cfg["io"]["output_dir"] = str(out)
    run_pipeline(cfg)


@pytest.fixture(scope="module")
def xyz_and_traj_runs(demo_config_template, bead_xyz, tmp_path_factory):
    """The demo run as i-PI XYZ (bead file = centroid) and as one .traj file."""
    base = tmp_path_factory.mktemp("traj_runs")
    xyz_cfg = copy.deepcopy(demo_config_template)
    xyz_cfg["runs"][0]["centroid_xyz"] = bead_xyz
    _run(xyz_cfg, base / "xyz")

    traj = base / "run.traj"
    ase.io.write(traj, ase.io.read(bead_xyz, ":", format="xyz"))
    traj_cfg = copy.deepcopy(demo_config_template)
    run = traj_cfg["runs"][0]
    for key in ("bead_glob", "centroid_xyz"):
        del run[key]
    run["trajectory"] = str(traj)
    _run(traj_cfg, base / "traj")
    return base / "xyz" / "demo", base / "traj" / "demo", run


def test_traj_run_scores_equal_the_xyz_run(xyz_and_traj_runs):
    xyz_dir, traj_dir, _ = xyz_and_traj_runs
    for name in ("bead_scores.npy", "centroid_scores.npy"):
        np.testing.assert_array_equal(np.load(traj_dir / name), np.load(xyz_dir / name))
    onset_cols = ["persistent_bead_anomaly_frame", "centroid_anomaly_frame", "collective_anomaly_frame"]
    pd.testing.assert_frame_equal(
        pd.read_csv(traj_dir / "onset_table.csv")[onset_cols],
        pd.read_csv(xyz_dir / "onset_table.csv")[onset_cols],
    )


def test_traj_run_has_one_bead_that_is_also_the_centroid(xyz_and_traj_runs):
    _, traj_dir, _ = xyz_and_traj_runs
    bead = np.load(traj_dir / "bead_scores.npy")
    assert bead.shape[0] == 1
    np.testing.assert_array_equal(bead[0], np.load(traj_dir / "centroid_scores.npy"))
    assert (traj_dir / "chemistry_flags_bead00.csv").exists()


def test_traj_time_axis_is_the_frame_index_times_timestep_times_stride(xyz_and_traj_runs):
    _, traj_dir, run = xyz_and_traj_runs
    agg = pd.read_csv(traj_dir / "frame_aggregate.csv")
    np.testing.assert_array_equal(agg["step"].to_numpy(), np.arange(len(agg)))
    np.testing.assert_allclose(
        agg["time_ps"].to_numpy(),
        np.arange(len(agg)) * run["timestep_fs"] * run["stride"] / 1000.0,
    )
