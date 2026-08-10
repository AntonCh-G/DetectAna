"""Shared pytest fixtures for the detectana test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_DIR = PROJECT_ROOT / "data" / "smoke"

INITIAL_XYZ = Path(os.environ.get("DETECTANA_INITIAL_XYZ", SMOKE_DATA_DIR / "initial.xyz"))
TRAIN_XYZ = Path(os.environ.get("DETECTANA_TRAIN_XYZ", SMOKE_DATA_DIR / "reference_train.xyz"))
VALID_XYZ = Path(os.environ.get("DETECTANA_VALID_XYZ", SMOKE_DATA_DIR / "reference_valid.xyz"))


@pytest.fixture(scope="module")
def topo():
    from detectana.topology import build_topology
    return build_topology(INITIAL_XYZ)


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
