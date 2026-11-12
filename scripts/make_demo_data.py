#!/usr/bin/env python
"""Generate the synthetic demo dataset in ``data/smoke/``.

The demo data is not simulation output. It is built from the single equilibrium
aspirin geometry in ``data/smoke/initial.xyz`` by thermal-style perturbation:
the two flexible torsions are drawn from wrapped normals, every atom is rattled
with an element-dependent amplitude, and the molecule is randomly oriented. No
force field and no reference calculation is involved, so the files carry
positions only — the geometric pipeline never reads forces.

That is enough to exercise every code path end to end and to check an install,
and it is not enough to mean anything scientifically: 64 training frames for a
134-dimensional descriptor is far below what the covariance needs.

Deterministic given the seeds below. Run from the repository root:

    python scripts/make_demo_data.py

Files written (all overwritten):

    data/smoke/reference_train.xyz   64 frames, extended XYZ
    data/smoke/reference_valid.xyz   32 frames, extended XYZ
    data/smoke/aspirin.pos_00.xyz    24 frames, i-PI XYZ, bead 0
    data/smoke/aspirin.xc.xyz        24 frames, i-PI XYZ, centroid

``initial.xyz`` is an input, not an output, and is left alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

from detectana.evaluation import dihedral_angles_deg, perturb_dihedral
from detectana.synthetic import (  # noqa: F401  (re-exported for older callers)
    CARBOXYL,
    CARBOXYL_SIGMA_DEG,
    ESTER,
    ESTER_SIGMA_DEG,
    RATTLE_H_A,
    RATTLE_HEAVY_A,
    SEED_TRAIN,
    SEED_TRAJ,
    SEED_VALID,
    random_orientation,
    rattle,
    reference_frames,
)
from detectana.topology import build_topology

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = PROJECT_ROOT / "data" / "smoke"

# Torsions, spreads, rattle amplitudes and seeds live in detectana.synthetic,
# which the benchmark also uses; re-exported here under their old names.
N_TRAIN = 64
N_VALID = 32
N_TRAJ = 24
STRIDE = 50  # i-PI output interval, matches config/demo.yaml
# Nonzero so step numbers and frame indices differ, as they do in any restarted
# i-PI run. tests/test_io.py relies on the distinction.
START_STEP = 100_000

# The trajectory walks the ester torsion out of its reference basin so the demo
# has an onset to find rather than a flat score trace.
TRAJ_DRIFT_DEG = 130.0
BEAD_JITTER_A = 0.05  # bead 0 around the centroid

CELL_A = 105.83544  # i-PI writes a cell even for a gas-phase molecule


_rattle = rattle
_random_orientation = random_orientation


def _reference_frames(
    equilibrium: np.ndarray,
    symbols: list[str],
    topo,
    n_frames: int,
    seed: int,
    motion: str = "rattle",
) -> np.ndarray:
    """Synthetic reference frames; see ``detectana.synthetic.reference_frames``."""
    return reference_frames(equilibrium, topo, n_frames, seed, motion=motion)


def _trajectory_frames(
    equilibrium: np.ndarray,
    symbols: list[str],
    topo,
    n_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and bead-0 trajectories that drift out of the reference basin."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0, TRAJ_DRIFT_DEG, n_frames)

    centroid, bead = [], []
    for step_deg in drift:
        pos = equilibrium.copy()
        pos = perturb_dihedral(
            pos, topo, float(step_deg + rng.normal(scale=4.0)), dihedral=ESTER
        )
        pos = perturb_dihedral(
            pos, topo, float(rng.normal(scale=CARBOXYL_SIGMA_DEG)), dihedral=CARBOXYL
        )
        pos = _random_orientation(_rattle(pos, symbols, rng), rng)
        centroid.append(pos)
        # One bead of a ring polymer fluctuates around the centroid.
        bead.append(pos + rng.normal(scale=BEAD_JITTER_A, size=pos.shape))
    return np.stack(centroid), np.stack(bead)


def _write_extxyz(path: Path, frames: np.ndarray, symbols: list[str]) -> None:
    with path.open("w") as fh:
        for i, pos in enumerate(frames):
            fh.write(f"{len(symbols)}\n")
            fh.write(
                'Properties=species:S:1:pos:R:3 '
                f'frame={i} pbc="F F F" '
                'comment="synthetic demo geometry, see scripts/make_demo_data.py"\n'
            )
            for s, (x, y, z) in zip(symbols, pos):
                fh.write(f"{s:2s} {x:20.12f} {y:20.12f} {z:20.12f}\n")


def _write_ipi_xyz(path: Path, frames: np.ndarray, symbols: list[str], bead: int) -> None:
    with path.open("w") as fh:
        for i, pos in enumerate(frames):
            fh.write(f"{len(symbols)}\n")
            fh.write(
                f"# CELL(abcABC): {CELL_A:11.5f} {CELL_A:11.5f} {CELL_A:11.5f}"
                f" {90.0:11.5f} {90.0:11.5f} {90.0:11.5f}"
                f"  Step: {START_STEP + i * STRIDE:11d}  Bead: {bead:8d}"
                " positions{angstrom}  cell{angstrom}\n"
            )
            for s, (x, y, z) in zip(symbols, pos):
                fh.write(f"{s:2s} {x:20.12f} {y:20.12f} {z:20.12f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial",
        type=Path,
        default=SMOKE_DIR / "initial.xyz",
        help="equilibrium geometry the demo data is built from",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SMOKE_DIR, help="where to write the demo files"
    )
    args = parser.parse_args()

    atoms = read(str(args.initial), index=0, format="xyz")
    equilibrium = np.asarray(atoms.get_positions(), dtype=np.float64)
    symbols = list(atoms.get_chemical_symbols())
    topo = build_topology(args.initial)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _reference_frames(equilibrium, symbols, topo, N_TRAIN, SEED_TRAIN)
    valid = _reference_frames(equilibrium, symbols, topo, N_VALID, SEED_VALID)
    centroid, bead = _trajectory_frames(equilibrium, symbols, topo, N_TRAJ, SEED_TRAJ)

    _write_extxyz(args.out_dir / "reference_train.xyz", train, symbols)
    _write_extxyz(args.out_dir / "reference_valid.xyz", valid, symbols)
    _write_ipi_xyz(args.out_dir / "aspirin.xc.xyz", centroid, symbols, bead=0)
    _write_ipi_xyz(args.out_dir / "aspirin.pos_00.xyz", bead, symbols, bead=0)

    # The byte-offset caches belong to the old files.
    for stale in args.out_dir.glob("*.frameindex.npz"):
        stale.unlink()

    ester = dihedral_angles_deg(centroid, ESTER)
    print(f"wrote {N_TRAIN} train, {N_VALID} valid, {N_TRAJ} trajectory frames")
    print(f"ester torsion over the trajectory: {ester[0]:.1f}° to {ester[-1]:.1f}°")


if __name__ == "__main__":
    main()
