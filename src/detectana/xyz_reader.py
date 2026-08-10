"""Fast binary XYZ reader for iPI-format trajectory files.

Ported from ConfAna/src/io_xyz.py.  Key design choices:
- Binary mode only — reliable byte offsets, no platform newline translation.
- Byte-offset index (NPZ) — O(1) random access; loaded from existing iPI-
  generated ``*.frameindex.npz`` files or built by a one-pass scan.
- No ASE objects — returns raw numpy arrays (steps, positions) per chunk.
- Coordinate-only parsing — element symbols are skipped, not read, so atom
  ordering has to be validated elsewhere (see ``io.MoleculeSpec``).

Frame format assumed (iPI standard XYZ):
    N
    # CELL(abcABC): ...  Step:   NNNNNNNN  Bead:   NN  positions{angstrom} ...
    C   x  y  z
    ...  (N lines)
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_RE_STEP = re.compile(rb"Step:\s*(\d+)")

# ---------------------------------------------------------------------------
# Index structures (mirrored from ConfAna for NPZ compatibility)
# ---------------------------------------------------------------------------

class _FrameIndex:
    """Lightweight container for per-file byte-offset index."""

    def __init__(
        self,
        source_file: str,
        file_size: int,
        file_mtime: float,
        frame_number: np.ndarray,
        byte_offset: np.ndarray,
        atom_count: np.ndarray,
        comment_raw: np.ndarray,
    ) -> None:
        self.source_file = source_file
        self.file_size = file_size
        self.file_mtime = file_mtime
        self.frame_number = frame_number   # (n_frames,) int64
        self.byte_offset = byte_offset     # (n_frames,) int64
        self.atom_count = atom_count       # (n_frames,) int32
        # comment_raw: flat uint8 of all comment lines joined by '\n'
        self._comment_raw = comment_raw
        self._comments: list[bytes] | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frame_number)

    def comment(self, i: int) -> bytes:
        """Return raw comment bytes for frame i."""
        if self._comments is None:
            self._comments = self._comment_raw.tobytes().split(b"\n")
        return self._comments[i]


# ---------------------------------------------------------------------------
# Index loading / building
# ---------------------------------------------------------------------------

def load_or_build_index(
    xyz_path: str | Path,
    cache_path: str | Path | None = None,
) -> _FrameIndex:
    """Load a byte-offset index from NPZ cache, or build it by scanning.

    The cache is considered valid when source-file path, size, and mtime
    all match.  Compatible with the iPI-generated ``*.frameindex.npz``
    files (same key layout as ConfAna).

    Parameters
    ----------
    xyz_path : path to the XYZ trajectory file.
    cache_path : NPZ file path.  Defaults to ``<xyz_path>.frameindex.npz``.
    """
    xyz_path = Path(xyz_path).resolve()
    if cache_path is None:
        cache_path = Path(str(xyz_path) + ".frameindex.npz")
    else:
        cache_path = Path(cache_path)

    stat = os.stat(xyz_path)

    if cache_path.exists():
        try:
            idx = _load_index_cache(cache_path)
            if (
                idx.source_file == str(xyz_path)
                and idx.file_size == stat.st_size
                and abs(idx.file_mtime - stat.st_mtime) < 1.0
            ):
                return idx
        except Exception:
            pass  # stale or corrupt cache — fall through to rebuild

    idx = _scan_index(xyz_path, stat)
    try:
        _save_index_cache(idx, cache_path)
    except Exception:
        pass  # non-fatal: proceed with in-memory index
    return idx


def _load_index_cache(cache_path: Path) -> _FrameIndex:
    data = np.load(cache_path, allow_pickle=False)
    return _FrameIndex(
        source_file=str(data["_source_file"]),
        file_size=int(data["_file_size"]),
        file_mtime=float(data["_file_mtime"]),
        frame_number=data["frame_number"],
        byte_offset=data["byte_offset"],
        atom_count=data["atom_count"],
        comment_raw=data["comment_raw"],
    )


def _save_index_cache(idx: _FrameIndex, path: Path) -> None:
    np.savez(
        path,
        _source_file=np.array(idx.source_file),
        _file_size=np.array(idx.file_size, dtype=np.int64),
        _file_mtime=np.array(idx.file_mtime, dtype=np.float64),
        frame_number=idx.frame_number,
        byte_offset=idx.byte_offset,
        atom_count=idx.atom_count,
        comment_raw=idx._comment_raw,
    )


def _scan_index(xyz_path: Path, stat: os.stat_result) -> _FrameIndex:
    """One-pass binary scan to build byte-offset index (no coord parsing)."""
    frame_numbers: list[int] = []
    byte_offsets: list[int] = []
    atom_counts: list[int] = []
    comments: list[bytes] = []

    frame_idx = 0
    with open(xyz_path, "rb") as fh:
        while True:
            offset = fh.tell()
            header = fh.readline()
            if not header:
                break
            header_str = header.strip()
            if not header_str:
                continue
            try:
                n_atoms = int(header_str)
            except ValueError as exc:
                raise ValueError(
                    f"{xyz_path.name}: frame {frame_idx}: "
                    f"expected atom count, got {header_str!r}"
                ) from exc

            comment = fh.readline()
            if not comment:
                raise ValueError(f"{xyz_path.name}: truncated at frame {frame_idx}")
            comment = comment.rstrip(b"\r\n")

            for _ in range(n_atoms):
                line = fh.readline()
                if not line:
                    raise ValueError(
                        f"{xyz_path.name}: truncated inside frame {frame_idx}"
                    )

            frame_numbers.append(frame_idx)
            byte_offsets.append(offset)
            atom_counts.append(n_atoms)
            comments.append(comment)
            frame_idx += 1

    comment_raw = np.frombuffer(b"\n".join(comments), dtype=np.uint8)
    return _FrameIndex(
        source_file=str(xyz_path),
        file_size=stat.st_size,
        file_mtime=stat.st_mtime,
        frame_number=np.array(frame_numbers, dtype=np.int64),
        byte_offset=np.array(byte_offsets, dtype=np.int64),
        atom_count=np.array(atom_counts, dtype=np.int32),
        comment_raw=comment_raw,
    )


# ---------------------------------------------------------------------------
# Coordinate parsing helpers
# ---------------------------------------------------------------------------

def _parse_step(comment: bytes) -> int | None:
    m = _RE_STEP.search(comment)
    return int(m.group(1)) if m else None


def _parse_atom_lines(lines: list[bytes], n_atoms: int) -> np.ndarray:
    """Parse n_atoms coordinate lines into (n_atoms, 3) float64 array."""
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    for row, line in enumerate(lines):
        parts = line.split()
        coords[row, 0] = float(parts[1])
        coords[row, 1] = float(parts[2])
        coords[row, 2] = float(parts[3])
    return coords


# ---------------------------------------------------------------------------
# Public iterators
# ---------------------------------------------------------------------------

def iter_positions_chunked(
    xyz_path: str | Path,
    chunk_size: int = 5000,
    stride: int = 50,
    cache_path: str | Path | None = None,
    expected_n_atoms: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(steps, positions)`` chunks from an iPI bead XYZ file.

    Uses the ConfAna-format byte-offset index (loads existing
    ``*.frameindex.npz`` or builds one on first call).

    Parameters
    ----------
    xyz_path : path to a bead or centroid trajectory file.
    chunk_size : frames per yielded chunk.
    stride : used as fallback when Step cannot be parsed from comment.
    cache_path : optional explicit path for the frameindex NPZ.
    expected_n_atoms : when given, the file's atom count must equal it.

    Yields
    ------
    steps : ndarray int64 (chunk,)
    positions : ndarray float64 (chunk, n_atoms, 3)  — Angstrom
    """
    xyz_path = Path(xyz_path)
    idx = load_or_build_index(xyz_path, cache_path)
    n_frames = idx.n_frames

    # All frames must hold the same molecule: the parser reuses one atom count.
    unique_counts = np.unique(idx.atom_count)
    if len(unique_counts) != 1:
        raise ValueError(
            f"{xyz_path.name}: mixed atom counts across frames: "
            f"{unique_counts.tolist()}"
        )
    n_atoms = int(unique_counts[0])
    if expected_n_atoms is not None and n_atoms != expected_n_atoms:
        raise ValueError(
            f"{xyz_path.name}: expected {expected_n_atoms} atoms per frame, "
            f"got {n_atoms}"
        )

    steps_buf: list[int] = []
    pos_buf: list[np.ndarray] = []

    with open(xyz_path, "rb") as fh:
        for fi in range(n_frames):
            fh.seek(int(idx.byte_offset[fi]))
            fh.readline()               # skip atom-count line
            comment = fh.readline().rstrip(b"\r\n")
            atom_lines = [fh.readline() for _ in range(n_atoms)]

            step = _parse_step(comment)
            if step is None:
                step = fi * stride

            steps_buf.append(step)
            pos_buf.append(_parse_atom_lines(atom_lines, n_atoms))

            if len(pos_buf) >= chunk_size:
                yield (
                    np.array(steps_buf, dtype=np.int64),
                    np.array(pos_buf, dtype=np.float64),
                )
                steps_buf, pos_buf = [], []

    if pos_buf:
        yield (
            np.array(steps_buf, dtype=np.int64),
            np.array(pos_buf, dtype=np.float64),
        )
