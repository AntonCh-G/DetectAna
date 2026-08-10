"""Shared pytest fixtures for the detectana test suite."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_DIR = PROJECT_ROOT / "data" / "smoke"
DEMO_CONFIG = PROJECT_ROOT / "config" / "demo.yaml"

INITIAL_XYZ = Path(os.environ.get("DETECTANA_INITIAL_XYZ", SMOKE_DATA_DIR / "initial.xyz"))
TRAIN_XYZ = Path(os.environ.get("DETECTANA_TRAIN_XYZ", SMOKE_DATA_DIR / "reference_train.xyz"))
VALID_XYZ = Path(os.environ.get("DETECTANA_VALID_XYZ", SMOKE_DATA_DIR / "reference_valid.xyz"))
BEAD_00_XYZ = Path(os.environ.get("DETECTANA_BEAD_00_XYZ", SMOKE_DATA_DIR / "aspirin.pos_00.xyz"))


@pytest.fixture(scope="module")
def initial_xyz():
    """Path to the aspirin geometry that defines the molecule for the tests."""
    return INITIAL_XYZ


@pytest.fixture(scope="module")
def topo():
    from detectana.topology import build_topology
    return build_topology(INITIAL_XYZ)


@pytest.fixture(scope="module")
def ethanol_xyz(tmp_path_factory):
    """A second, ring-less molecule (9 atoms) for the generality tests."""
    from ase.build import molecule
    from ase.io import write

    path = tmp_path_factory.mktemp("ethanol") / "initial.xyz"
    write(str(path), molecule("CH3CH2OH"), format="xyz")
    return path


@pytest.fixture(scope="module")
def train_positions():
    """All reference frames shipped in data/smoke (train + valid concatenated).

    The two files are kept disjoint on disk so the demo config can calibrate a
    threshold without leakage; unit tests need the full set to keep the fitted
    PCA subspace wide enough for the descriptor-sensitivity checks.
    """
    import numpy as np

    from detectana.io import load_reference_frames
    train_pos, _, _ = load_reference_frames(TRAIN_XYZ)
    valid_pos, _, _ = load_reference_frames(VALID_XYZ)
    return np.concatenate([train_pos, valid_pos], axis=0)  # (n_frames, 21, 3)


@pytest.fixture(scope="module")
def fitted_pipeline(topo, train_positions):
    from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
    X_train = compute_descriptor_batch(train_positions, topo)
    pipe = DescriptorPipeline(pca_variance=0.95, random_seed=42)
    pipe.fit(X_train)
    return pipe, X_train


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------

def _absolutise(cfg: dict) -> dict:
    """Make every data path in a config absolute, relative to the repo root.

    The shipped configs use paths relative to the repository root; a test may run
    from anywhere, so resolve them here rather than depending on the cwd.
    """
    def fix(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (PROJECT_ROOT / path))

    ref = cfg["reference"]
    for key in ("train", "valid", "test"):
        if key in ref:
            ref[key] = fix(ref[key])
    for run in cfg["runs"]:
        for key in ("initial_xyz", "bead_glob", "centroid_xyz", "hdf5"):
            if key in run:
                run[key] = fix(run[key])
    return cfg


@pytest.fixture(scope="session")
def demo_config_template() -> dict:
    """``config/demo.yaml`` as loaded from disk, with data paths absolute.

    Session-scoped and never handed out directly — use the ``demo_config``
    fixture, which copies it, so one test cannot mutate another's config.
    """
    import yaml

    with open(DEMO_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    return _absolutise(cfg)


@pytest.fixture
def demo_config(demo_config_template, tmp_path) -> dict:
    """The demo config, writing into this test's ``tmp_path``.

    Everything else — percentiles, window rule, seed — is exactly what the demo
    run uses, so a test asserting on these outputs is asserting on the shipped
    defaults.
    """
    cfg = copy.deepcopy(demo_config_template)
    cfg["io"]["output_dir"] = str(tmp_path / "outputs")
    return cfg


@pytest.fixture(scope="session")
def smoke_bead_positions():
    """Positions from the smoke bead trajectory: (n_frames, 21, 3) Å."""
    from detectana.io import load_trajectory_positions

    positions, _ = load_trajectory_positions(BEAD_00_XYZ, pimd=True)
    return positions


@pytest.fixture(scope="session")
def smoke_hdf5(tmp_path_factory, smoke_bead_positions) -> Path:
    """A small ``nvt_trajectory.hdf5`` in the layout the HDF5 loader expects.

    Real smoke geometries are used for bead 0, and the other beads are that
    trajectory displaced by a fixed, tiny (0.01 Å scale) amount — enough to make
    the beads distinguishable without turning them into different molecules. The
    centroid is the bead mean, as it is in a real PIMD run.
    """
    import h5py
    import numpy as np

    n_beads = 3
    base = np.asarray(smoke_bead_positions, dtype=np.float64)
    rng = np.random.default_rng(20260810)
    jitter = rng.normal(scale=0.01, size=(n_beads, *base.shape))
    jitter[0] = 0.0
    bead_positions = base[None, ...] + jitter          # (n_beads, n_frames, n_atoms, 3)
    bead_positions = bead_positions.transpose(1, 0, 2, 3)  # → (n_frames, n_beads, …)
    centroid_positions = bead_positions.mean(axis=1)
    n_frames = bead_positions.shape[0]

    path = tmp_path_factory.mktemp("hdf5") / "nvt_trajectory.hdf5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("bead_positions", data=bead_positions)
        fh.create_dataset("positions", data=centroid_positions)
        fh.create_dataset("potential", data=np.linspace(-100.0, -99.0, n_frames))
    return path


@pytest.fixture
def hdf5_config(demo_config, smoke_hdf5) -> dict:
    """The demo config with its XYZ run replaced by the synthetic HDF5 run."""
    run = {
        "name": "hdf5run",
        "initial_xyz": str(INITIAL_XYZ),
        "hdf5": str(smoke_hdf5),
        "timestep_fs": 0.2,
        "stride": 50,
    }
    demo_config["runs"] = [run]
    return demo_config
