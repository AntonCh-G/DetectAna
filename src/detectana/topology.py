"""Aspirin molecular topology: bond graph, angles, dihedrals, ring detection.

Hard-chemistry checks (bond breaking, close contacts, ring planarity) are
computed here and are separate from the statistical OOD score.

The topology is built once from ``initial.xyz`` and frozen for the entire run.
Atom ordering and count are validated before any descriptor is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AspirinTopology:
    """Fixed molecular topology derived from the initial geometry.

    Attributes
    ----------
    bonds : list of (i, j) with i < j
    angles : list of (i, j, k)  — j is the vertex
    dihedrals : list of (i, j, k, l)
    ring_atoms : indices of the 6 benzene-ring carbons
    n_atoms : always 21
    atom_types : list of element symbols
    bond_names : human-readable label per bond, e.g. "C0-C5"
    angle_names : human-readable label per angle
    dihedral_names : pairs of sin/cos feature labels
    planarity_name : label for ring-planarity feature
    """

    bonds: list[tuple[int, int]] = field(default_factory=list)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    ring_atoms: list[int] = field(default_factory=list)
    n_atoms: int = 21
    atom_types: list[str] = field(default_factory=list)

    # Feature names (populated by _build_names)
    bond_names: list[str] = field(default_factory=list)
    angle_names: list[str] = field(default_factory=list)
    dihedral_names: list[str] = field(default_factory=list)
    planarity_name: str = "ring_planarity_rmsd_Å"

    # Precomputed numpy index arrays for vectorised batch descriptor computation.
    # Populated by build_topology(); None until then.
    bond_idx: np.ndarray | None = field(default=None)       # (n_bonds, 2)
    angle_idx: np.ndarray | None = field(default=None)      # (n_angles, 3)
    dihedral_idx: np.ndarray | None = field(default=None)   # (n_dihedrals, 4)
    ring_idx: np.ndarray | None = field(default=None)       # (6,)

    @property
    def n_bond_features(self) -> int:
        return len(self.bonds)

    @property
    def n_angle_features(self) -> int:
        return len(self.angles)

    @property
    def n_dihedral_features(self) -> int:
        return 2 * len(self.dihedrals)  # sin + cos per dihedral

    @property
    def n_features(self) -> int:
        return self.n_bond_features + self.n_angle_features + self.n_dihedral_features + 1

    @property
    def feature_names(self) -> list[str]:
        names = list(self.bond_names)
        names += self.angle_names
        names += self.dihedral_names   # already 2 per dihedral (sin_, cos_)
        names += [self.planarity_name]
        return names


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_topology(
    initial_xyz: str | Path,
    nl_mult: float = 1.1,
) -> AspirinTopology:
    """Build topology from the first frame of ``initial_xyz``.

    Parameters
    ----------
    initial_xyz : path to initial geometry (21-atom aspirin XYZ).
    nl_mult : NeighborList covalent-radii multiplier (default 1.1).
    """
    from detectana.io import load_single_frame

    atoms = load_single_frame(initial_xyz)
    topo = AspirinTopology(
        n_atoms=len(atoms),
        atom_types=list(atoms.get_chemical_symbols()),
    )

    # ── Bond graph ──────────────────────────────────────────────────────────
    cutoffs = natural_cutoffs(atoms, mult=nl_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    conn = nl.get_connectivity_matrix(sparse=False)

    topo.bonds = [
        (i, j)
        for i in range(topo.n_atoms)
        for j in range(i + 1, topo.n_atoms)
        if conn[i, j] > 0
    ]

    if not topo.bonds:
        raise ValueError("No bonds detected in initial.xyz with nl_mult={nl_mult}")

    # ── Adjacency dict ───────────────────────────────────────────────────────
    adj: dict[int, list[int]] = {i: [] for i in range(topo.n_atoms)}
    for i, j in topo.bonds:
        adj[i].append(j)
        adj[j].append(i)

    # ── Angles ───────────────────────────────────────────────────────────────
    seen_angles: set[tuple[int, int, int]] = set()
    for j in range(topo.n_atoms):
        nbrs = adj[j]
        for a, i in enumerate(nbrs):
            for k in nbrs[a + 1 :]:
                angle = (min(i, k), j, max(i, k))
                if angle not in seen_angles:
                    seen_angles.add(angle)
                    topo.angles.append(angle)

    # ── Dihedrals ────────────────────────────────────────────────────────────
    seen_dih: set[tuple[int, int, int, int]] = set()
    for j, k in topo.bonds:
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                fwd = (i, j, k, l)
                rev = (l, k, j, i)
                canonical = min(fwd, rev)
                if canonical not in seen_dih:
                    seen_dih.add(canonical)
                    topo.dihedrals.append(canonical)

    # ── Benzene ring (6-membered all-carbon ring) ────────────────────────────
    topo.ring_atoms = _find_benzene_ring(adj, topo.atom_types)

    # ── Feature names ────────────────────────────────────────────────────────
    sym = topo.atom_types
    topo.bond_names = [f"bond_{sym[i]}{i}-{sym[j]}{j}_Å" for i, j in topo.bonds]
    topo.angle_names = [
        f"angle_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}_rad"
        for i, j, k in topo.angles
    ]
    topo.dihedral_names = [
        name
        for i, j, k, l in topo.dihedrals
        for name in (
            f"sin_dih_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}-{sym[l]}{l}",
            f"cos_dih_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}-{sym[l]}{l}",
        )
    ]

    # ── Precomputed numpy index arrays for vectorised descriptors ─────────────
    topo.bond_idx = np.array(topo.bonds, dtype=np.int32)            # (n_bonds, 2)
    topo.angle_idx = np.array(topo.angles, dtype=np.int32)          # (n_angles, 3)
    topo.dihedral_idx = np.array(topo.dihedrals, dtype=np.int32)    # (n_dihedrals, 4)
    topo.ring_idx = np.array(topo.ring_atoms, dtype=np.int32)       # (6,)

    return topo


def _find_benzene_ring(
    adj: dict[int, list[int]],
    atom_types: list[str],
) -> list[int]:
    """Return indices of the 6 benzene-ring carbons (DFS cycle search)."""
    n = len(atom_types)
    carbon_set = {i for i in range(n) if atom_types[i] == "C"}

    def dfs(
        start: int,
        current: int,
        path: list[int],
        visited: set[int],
    ) -> list[int] | None:
        for nbr in adj[current]:
            if nbr not in carbon_set:
                continue
            if nbr == start and len(path) == 6:
                return path[:]
            if nbr in visited or len(path) >= 6:
                continue
            visited.add(nbr)
            result = dfs(start, nbr, path + [nbr], visited)
            visited.discard(nbr)
            if result is not None:
                return result
        return None

    for start in sorted(carbon_set):
        result = dfs(start, start, [start], {start})
        if result is not None:
            return sorted(result)

    raise ValueError(
        "No 6-membered all-carbon ring found in topology. "
        "Check initial.xyz atom ordering."
    )


# ---------------------------------------------------------------------------
# Hard-chemistry checks
# ---------------------------------------------------------------------------

@dataclass
class ChemistryFlags:
    """Per-frame hard-chemistry check results."""

    broken_bonds: list[tuple[int, int, float]]   # (i, j, distance_Å)
    close_contacts: list[tuple[int, int, float]]  # (i, j, distance_Å)
    ring_planarity_rmsd: float                    # Å

    @property
    def has_broken_bond(self) -> bool:
        return len(self.broken_bonds) > 0

    @property
    def has_close_contact(self) -> bool:
        return len(self.close_contacts) > 0

    def to_dict(self) -> dict:
        return {
            "broken_bond": self.has_broken_bond,
            "n_broken_bonds": len(self.broken_bonds),
            "close_contact": self.has_close_contact,
            "n_close_contacts": len(self.close_contacts),
            "ring_planarity_rmsd_Å": self.ring_planarity_rmsd,
        }


def check_chemistry(
    positions: np.ndarray,
    topo: AspirinTopology,
    bond_break_cutoff: float = 2.0,
    close_contact_cutoff: float = 1.2,
) -> ChemistryFlags:
    """Run hard-chemistry checks on a single frame.

    Parameters
    ----------
    positions : (n_atoms, 3) Å
    topo : topology from build_topology
    bond_break_cutoff : bond longer than this (Å) is flagged broken
    close_contact_cutoff : non-bonded pair closer than this (Å) is flagged
    """
    broken: list[tuple[int, int, float]] = []
    for i, j in topo.bonds:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if d > bond_break_cutoff:
            broken.append((i, j, d))

    bonded_set = set(map(frozenset, topo.bonds))  # type: ignore[arg-type]
    close: list[tuple[int, int, float]] = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset({i, j}) in bonded_set:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < close_contact_cutoff:
                close.append((i, j, d))

    planarity = _ring_planarity_rmsd(positions, topo.ring_atoms)
    return ChemistryFlags(broken, close, planarity)


def check_chemistry_batch(
    positions: np.ndarray,
    topo: AspirinTopology,
    bond_break_cutoff: float = 2.0,
    close_contact_cutoff: float = 1.2,
) -> list[ChemistryFlags]:
    """Run hard-chemistry checks on a batch of frames.

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3) Å
    """
    return [
        check_chemistry(positions[i], topo, bond_break_cutoff, close_contact_cutoff)
        for i in range(len(positions))
    ]


def _ring_planarity_rmsd(positions: np.ndarray, ring_indices: Sequence[int]) -> float:
    """RMSD of ring atoms from their best-fit plane (Å)."""
    ring_pos = positions[list(ring_indices)]   # (6, 3)
    centroid = ring_pos.mean(axis=0)
    centered = ring_pos - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    distances = np.abs(centered @ normal)
    return float(np.sqrt(np.mean(distances**2)))
