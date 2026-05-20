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

import re
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from ase import Atoms
from ase.io import iread, read

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
# Single-frame reader (for initial.xyz topology seed)
# ---------------------------------------------------------------------------

def load_single_frame(path: str | Path) -> Atoms:
    """Load the first frame from any XYZ file (e.g. initial.xyz)."""
    path = Path(path)
    atoms = read(str(path), index=0)
    validate_frame(atoms, 0, source=path.name)
    return atoms
