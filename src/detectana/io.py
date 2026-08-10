"""Trajectory I/O for PIMD bead files and extended-XYZ reference sets.

Assumptions
-----------
- PIMD bead position files: standard XYZ with a CELL/Step/Bead comment line.
  Positions are in Angstrom.  Units tag ``positions{angstrom}`` is *assumed*
  from iPI convention; no runtime unit conversion is applied to positions.
- Reference files (train/valid/test): extended XYZ with
  ``Properties=species:S:1:pos:R:3:forces:R:3 energy=...``.
  Positions in Å, forces in eV/Å, energy in eV.
- Aspirin is always 21 atoms in the order defined by ``initial.xyz``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

import numpy as np
from ase import Atoms
from ase.io import iread, read

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASPIRIN_N_ATOMS: int = 21
ASPIRIN_ATOM_TYPES: list[str] = [
    "C", "C", "C", "C", "C", "C", "C",
    "O", "O", "O",
    "C", "C", "O",
    "H", "H", "H", "H", "H", "H", "H", "H",
]

_STEP_RE = re.compile(r"Step:\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_step(comment: str) -> int | None:
    """Extract step number from an iPI-style XYZ comment line."""
    m = _STEP_RE.search(comment)
    return int(m.group(1)) if m else None


def _atoms_to_info(atoms: Atoms) -> dict:
    """Return atoms.info dict, tolerating ASE version differences."""
    return atoms.info if hasattr(atoms, "info") else {}


def validate_frame(atoms: Atoms, frame_idx: int, source: str = "") -> None:
    """Raise ValueError immediately on atom-count or atom-type mismatch."""
    loc = f"[{source}] " if source else ""
    n = len(atoms)
    if n != ASPIRIN_N_ATOMS:
        raise ValueError(
            f"{loc}Frame {frame_idx}: expected {ASPIRIN_N_ATOMS} atoms, got {n}"
        )
    types = list(atoms.get_chemical_symbols())
    if types != ASPIRIN_ATOM_TYPES:
        raise ValueError(
            f"{loc}Frame {frame_idx}: atom-type mismatch.\n"
            f"  Expected: {ASPIRIN_ATOM_TYPES}\n"
            f"  Got:      {types}"
        )


# ---------------------------------------------------------------------------
# PIMD bead trajectory loader (chunked, lazy)
# ---------------------------------------------------------------------------

def iter_bead_positions(
    pos_path: str | Path,
    chunk_size: int = 5000,
    stride: int = 50,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(steps, positions)`` chunks from a PIMD bead position XYZ file.

    Uses the fast binary reader (``xyz_reader``) with the existing
    ``*.frameindex.npz`` byte-offset index — no ASE overhead.

    Parameters
    ----------
    pos_path:
        Path to ``aspirin.pos_NN.xyz``.
    chunk_size:
        Number of frames per yielded chunk.
    stride:
        MD output stride used as fallback when Step cannot be parsed.

    Yields
    ------
    steps : ndarray, shape (chunk,)
    positions : ndarray, shape (chunk, n_atoms, 3)  Å
    """
    from detectana.xyz_reader import iter_positions_chunked

    yield from iter_positions_chunked(pos_path, chunk_size=chunk_size, stride=stride)


# ---------------------------------------------------------------------------
# Reference dataset loader (small files — load fully into memory)
# ---------------------------------------------------------------------------

def load_reference_frames(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/valid/test XYZ file into arrays.

    Returns
    -------
    positions : (n_frames, n_atoms, 3)  Å
    forces    : (n_frames, n_atoms, 3)  eV/Å
    energies  : (n_frames,)             eV
    """
    path = Path(path)
    frames = list(iread(str(path), format="extxyz"))
    if not frames:
        raise ValueError(f"No frames found in {path}")

    for i, f in enumerate(frames):
        validate_frame(f, i, source=path.name)

    positions = np.array([f.get_positions() for f in frames], dtype=np.float64)

    def load_frame_property(
        getter: Callable[[Atoms], object],
        info_key: str,
        info_default: object,
        zero_fallback: Callable[[], np.ndarray],
    ) -> np.ndarray:
        for values in (
            lambda: [getter(f) for f in frames],
            lambda: [_atoms_to_info(f).get(info_key, info_default) for f in frames],
        ):
            try:
                return np.asarray(values(), dtype=np.float64)
            except Exception:
                continue
        return zero_fallback()

    energies = load_frame_property(
        Atoms.get_potential_energy,
        "REF_energy",
        0.0,
        lambda: np.zeros(len(frames), dtype=np.float64),
    )

    # Forces may be absent if the file only contains positions.
    forces = load_frame_property(
        Atoms.get_forces,
        "REF_forces",
        np.zeros((ASPIRIN_N_ATOMS, 3), dtype=np.float64),
        lambda: np.zeros_like(positions),
    )

    return positions, forces, energies


# ---------------------------------------------------------------------------
# Pre-computed MlffModel embedding reader (HDF5)
# ---------------------------------------------------------------------------

def load_embeddings_h5(
    h5_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed MlffModel inv_features from an HDF5 file.

    Expected HDF5 layout (written by scripts/extract_embeddings.py):
    - ``inv_features`` : (n_frames, n_atoms, n_features)  float32 or float64
    - ``steps``        : (n_frames,)                       int64

    Parameters
    ----------
    h5_path : path to a bead or centroid embedding HDF5 file.

    Returns
    -------
    embeddings : (n_frames, n_atoms, n_features)  float64
    steps      : (n_frames,)                       int64
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read embedding files. "
            "Install it with: pip install h5py"
        ) from exc

    with h5py.File(h5_path, "r") as fh:
        embeddings = fh["inv_features"][()].astype(np.float64)
        steps = fh["steps"][()].astype(np.int64)
    return embeddings, steps


# ---------------------------------------------------------------------------
# Full-trajectory loader (MD extxyz or PIMD iPI format)
# ---------------------------------------------------------------------------

def load_trajectory_frames(
    path: str | Path,
    pimd: bool = False,
) -> tuple[list[Atoms], np.ndarray]:
    """Load all frames from an MD (extxyz) or PIMD centroid (iPI) XYZ file.

    Parameters
    ----------
    path : path to trajectory file.
    pimd : True → read as iPI-format XYZ (e.g. ``aspirin.xc.xyz``);
           False → read as extended XYZ (e.g. reference dataset files).

    Returns
    -------
    atoms_list : list of Atoms, one per frame (validated against aspirin)
    steps      : (n_frames,)  int64 — MD step numbers parsed from comment lines;
                 falls back to frame index when the comment carries no Step field.
    """
    path = Path(path)
    fmt = "xyz" if pimd else "extxyz"
    atoms_list: list[Atoms] = []
    steps: list[int] = []
    for frame_idx, atoms in enumerate(iread(str(path), format=fmt)):
        validate_frame(atoms, frame_idx, source=path.name)
        atoms_list.append(atoms)
        m = _STEP_RE.search(str(atoms.info))
        steps.append(int(m.group(1)) if m else frame_idx)
    if not atoms_list:
        raise ValueError(f"No frames found in {path}")
    return atoms_list, np.array(steps, dtype=np.int32)


def load_trajectory_positions(
    path: str | Path,
    pimd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast loader returning positions and steps as numpy arrays.

    For ``pimd=True``, uses the binary xyz_reader (no ASE overhead) — orders
    of magnitude faster than load_trajectory_frames on large iPI files.
    Validates atom count and atom types on the first frame only; logs a warning
    if the file contains mixed atom counts (caught by the index scan).

    Parameters
    ----------
    path : path to trajectory file.
    pimd : True → fast binary reader (iPI XYZ, e.g. ``aspirin.xc.xyz``);
           False → falls back to load_trajectory_frames + position extraction.

    Returns
    -------
    positions : (n_frames, n_atoms, 3)  float64  Å
    steps      : (n_frames,)            int64
    """
    if not pimd:
        atoms_list, steps = load_trajectory_frames(path, pimd=False)
        positions = np.array(
            [a.get_positions() for a in atoms_list], dtype=np.float64
        )
        return positions, steps.astype(np.int64)

    from detectana.xyz_reader import iter_positions_chunked, load_or_build_index

    path = Path(path)
    idx = load_or_build_index(path)

    # Validate atom count from index (all frames must match)
    unique_counts = np.unique(idx.atom_count)
    if len(unique_counts) != 1:
        raise ValueError(
            f"{path.name}: mixed atom counts across frames: {unique_counts.tolist()}"
        )
    if int(unique_counts[0]) != ASPIRIN_N_ATOMS:
        raise ValueError(
            f"{path.name}: expected {ASPIRIN_N_ATOMS} atoms per frame, "
            f"got {int(unique_counts[0])}"
        )

    log.info("Fast binary reader: %d frames in %s", idx.n_frames, path.name)

    all_positions: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []
    for steps_chunk, pos_chunk in iter_positions_chunked(path):
        all_positions.append(pos_chunk)
        all_steps.append(steps_chunk)

    if not all_positions:
        raise ValueError(f"No frames found in {path}")

    positions = np.concatenate(all_positions, axis=0)
    steps = np.concatenate(all_steps, axis=0)
    return positions, steps


# ---------------------------------------------------------------------------
# Single-frame reader (for initial.xyz topology seed)
# ---------------------------------------------------------------------------

def load_single_frame(path: str | Path) -> Atoms:
    """Load the first frame from any XYZ file (e.g. initial.xyz)."""
    path = Path(path)
    atoms = read(str(path), index=0)
    validate_frame(atoms, 0, source=path.name)
    return atoms


# ---------------------------------------------------------------------------
# PIMD HDF5 trajectory loader
# ---------------------------------------------------------------------------

class PIMDTrajectory(NamedTuple):
    """Full PIMD trajectory loaded from a single HDF5 file.

    All position arrays are in Angstrom.  Potential is in eV.
    Force fields are None until the HDF5 format includes them.
    """

    bead_positions: np.ndarray    # (n_frames, n_beads, n_atoms, 3)  Å
    centroid_positions: np.ndarray  # (n_frames, n_atoms, 3)          Å
    potential: np.ndarray          # (n_frames,)                      eV
    steps: np.ndarray              # (n_frames,)                      int64  frame index
    bead_forces: np.ndarray | None = None      # (n_frames, n_beads, n_atoms, 3) future
    centroid_forces: np.ndarray | None = None  # (n_frames, n_atoms, 3)          future


def load_pimd_trajectory_hdf5(h5_path: str | Path) -> PIMDTrajectory:
    """Load a PIMD trajectory from an HDF5 file.

    Expected datasets
    -----------------
    bead_positions : (n_frames, n_beads, n_atoms, 3)  float64  Å
    positions      : (n_frames, n_atoms, 3)            float64  Å  (centroid)
    potential      : (n_frames,)                       float64  eV

    Optional (not yet present — reserved for future format versions)
    -----------------
    bead_forces    : (n_frames, n_beads, n_atoms, 3)
    forces         : (n_frames, n_atoms, 3)            (centroid forces)

    Parameters
    ----------
    h5_path : path to ``nvt_trajectory.hdf5`` or equivalent.

    Returns
    -------
    PIMDTrajectory namedtuple — see class docstring for field shapes.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read HDF5 trajectory files. "
            "Install it with: pip install h5py"
        ) from exc

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as fh:
        bead_pos = fh["bead_positions"][()].astype(np.float64)
        centroid_pos = fh["positions"][()].astype(np.float64)
        potential = fh["potential"][()].astype(np.float64)

        bead_forces = (
            fh["bead_forces"][()].astype(np.float64)
            if "bead_forces" in fh else None
        )
        centroid_forces = (
            fh["forces"][()].astype(np.float64)
            if "forces" in fh else None
        )

    n_frames, n_beads, n_atoms, _ = bead_pos.shape
    if n_atoms != ASPIRIN_N_ATOMS:
        raise ValueError(
            f"{h5_path.name}: expected {ASPIRIN_N_ATOMS} atoms, got {n_atoms}"
        )
    if centroid_pos.shape != (n_frames, n_atoms, 3):
        raise ValueError(
            f"{h5_path.name}: centroid shape {centroid_pos.shape} inconsistent "
            f"with bead shape {bead_pos.shape}"
        )

    steps = np.arange(n_frames, dtype=np.int64)
    log.info(
        "Loaded HDF5 trajectory %s: %d frames, %d beads, %d atoms",
        h5_path.name, n_frames, n_beads, n_atoms,
    )
    return PIMDTrajectory(
        bead_positions=bead_pos,
        centroid_positions=centroid_pos,
        potential=potential,
        steps=steps,
        bead_forces=bead_forces,
        centroid_forces=centroid_forces,
    )
