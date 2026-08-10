"""Molecular topology: bond graph, angles, dihedrals, ring detection.

Hard-chemistry checks (bond breaking, close contacts, ring planarity) are
computed here and are separate from the statistical OOD score.

The topology is built once from ``initial.xyz`` and frozen for the entire run.
Nothing about the molecule is hard-coded: atom count, element order, the bond
graph and the ring all come from that file.  Atom ordering and count are then
validated on every frame before any descriptor is computed.

A molecule with no ring is supported — the ring-planarity feature and flag are
simply dropped, which changes the descriptor length.  Fit and threshold must
therefore come from the same topology as the frames being scored.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MoleculeTopology:
    """Fixed molecular topology derived from the initial geometry.

    Attributes
    ----------
    bonds : list of (i, j) with i < j
    angles : list of (i, j, k)  — j is the vertex
    dihedrals : list of (i, j, k, l)
    ring_atoms : indices of the carbon ring used for the planarity feature;
                 empty when the molecule has no such ring
    n_atoms : atom count taken from the initial geometry
    atom_types : list of element symbols
    bond_names : human-readable label per bond, e.g. "C0-C5"
    angle_names : human-readable label per angle
    dihedral_names : sin labels for every dihedral, then cos labels, matching
                     the descriptor column order
    planarity_name : label for ring-planarity feature
    """

    bonds: list[tuple[int, int]] = field(default_factory=list)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    ring_atoms: list[int] = field(default_factory=list)
    n_atoms: int = 0
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
    ring_idx: np.ndarray | None = field(default=None)       # (ring_size,) or None

    @property
    def has_ring(self) -> bool:
        """True when a ring was found or configured, so planarity is a feature."""
        return len(self.ring_atoms) > 0

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
    def n_planarity_features(self) -> int:
        return 1 if self.has_ring else 0

    @property
    def n_features(self) -> int:
        return (
            self.n_bond_features
            + self.n_angle_features
            + self.n_dihedral_features
            + self.n_planarity_features
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.bond_names)
        names += self.angle_names
        names += self.dihedral_names   # sin block then cos block
        if self.has_ring:
            names += [self.planarity_name]
        return names


# Name kept for backwards compatibility with earlier, aspirin-only releases.
AspirinTopology = MoleculeTopology


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_topology(
    initial_xyz: str | Path,
    nl_mult: float = 1.1,
    ring_atoms: Sequence[int] | None = None,
    ring_size: int = 6,
) -> MoleculeTopology:
    """Build topology from the first frame of ``initial_xyz``.

    Parameters
    ----------
    initial_xyz : path to the initial geometry. Its atom count and element order
        define the molecule for the whole run.
    nl_mult : NeighborList covalent-radii multiplier (default 1.1).
    ring_atoms : explicit 0-based indices of the ring used for the planarity
        feature. Give this when the molecule has more than one candidate ring,
        since auto-detection would otherwise pick one of them arbitrarily.
        Pass an empty sequence to switch the planarity feature off.
    ring_size : ring length to search for when ``ring_atoms`` is None
        (default 6, i.e. a benzene-like carbon ring).
    """
    from detectana.io import load_single_frame

    atoms = load_single_frame(initial_xyz)
    topo = MoleculeTopology(
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
        raise ValueError(
            f"No bonds detected in {Path(initial_xyz).name} with nl_mult={nl_mult}"
        )

    # ── Adjacency dict ───────────────────────────────────────────────────────
    adj: dict[int, list[int]] = {i: [] for i in range(topo.n_atoms)}
    for i, j in topo.bonds:
        adj[i].append(j)
        adj[j].append(i)

    n_fragments = _count_connected_components(adj, topo.n_atoms)
    if n_fragments > 1:
        log.warning(
            "%s: bond graph has %d disconnected fragments. Internal coordinates "
            "are computed on raw positions with no periodic images, so results "
            "are only meaningful for a single gas-phase molecule.",
            Path(initial_xyz).name, n_fragments,
        )

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

    # ── Ring used for the planarity feature ──────────────────────────────────
    if ring_atoms is None:
        topo.ring_atoms = _find_carbon_ring(adj, topo.atom_types, ring_size)
        if topo.ring_atoms:
            log.info("Ring for planarity feature (auto-detected): %s", topo.ring_atoms)
        else:
            log.info(
                "No %d-membered all-carbon ring found — ring-planarity feature "
                "and flag are omitted.", ring_size,
            )
    else:
        topo.ring_atoms = _validate_ring_atoms(ring_atoms, adj, topo.n_atoms)
        if topo.ring_atoms:
            log.info("Ring for planarity feature (from config): %s", topo.ring_atoms)
        else:
            log.info("Ring-planarity feature disabled by configuration.")

    # ── Feature names ────────────────────────────────────────────────────────
    sym = topo.atom_types
    topo.bond_names = [f"bond_{sym[i]}{i}-{sym[j]}{j}_Å" for i, j in topo.bonds]
    topo.angle_names = [
        f"angle_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}_rad"
        for i, j, k in topo.angles
    ]
    # Column order matches the descriptor: all sines first, then all cosines.
    dih_labels = [
        f"dih_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}-{sym[l]}{l}"
        for i, j, k, l in topo.dihedrals
    ]
    topo.dihedral_names = (
        [f"sin_{label}" for label in dih_labels]
        + [f"cos_{label}" for label in dih_labels]
    )

    # ── Precomputed numpy index arrays for vectorised descriptors ─────────────
    topo.bond_idx = np.array(topo.bonds, dtype=np.int32)            # (n_bonds, 2)
    topo.angle_idx = np.array(topo.angles, dtype=np.int32)          # (n_angles, 3)
    topo.dihedral_idx = np.array(topo.dihedrals, dtype=np.int32)    # (n_dihedrals, 4)
    topo.ring_idx = (
        np.array(topo.ring_atoms, dtype=np.int32) if topo.ring_atoms else None
    )

    return topo


def _count_connected_components(adj: dict[int, list[int]], n_atoms: int) -> int:
    """Number of connected fragments in the bond graph."""
    seen: set[int] = set()
    components = 0
    for start in range(n_atoms):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for nbr in adj[current]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
    return components


def _find_carbon_ring(
    adj: dict[int, list[int]],
    atom_types: list[str],
    ring_size: int = 6,
) -> list[int]:
    """Return the indices of one all-carbon ring, or [] if there is none.

    DFS cycle search; with several matching rings the lowest-indexed one wins,
    so pass ``ring_atoms`` explicitly when the choice matters.
    """
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
            if nbr == start and len(path) == ring_size:
                return path[:]
            if nbr in visited or len(path) >= ring_size:
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

    return []


def _validate_ring_atoms(
    ring_atoms: Sequence[int],
    adj: dict[int, list[int]],
    n_atoms: int,
) -> list[int]:
    """Check explicitly configured ring indices and return them sorted."""
    indices = [int(i) for i in ring_atoms]
    if not indices:
        return []
    if len(indices) < 3:
        raise ValueError(
            f"ring_atoms needs at least 3 atoms to define a plane, got {indices}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"ring_atoms contains duplicate indices: {indices}")
    out_of_range = [i for i in indices if not 0 <= i < n_atoms]
    if out_of_range:
        raise ValueError(
            f"ring_atoms indices out of range for a {n_atoms}-atom molecule: "
            f"{out_of_range}. Indices are 0-based."
        )
    ring_set = set(indices)
    for i in indices:
        if len(ring_set.intersection(adj[i])) != 2:
            raise ValueError(
                f"ring_atoms {indices} is not a closed ring in the bond graph: "
                f"atom {i} has {len(ring_set.intersection(adj[i]))} ring "
                "neighbours, expected 2."
            )
    return sorted(indices)


# ---------------------------------------------------------------------------
# Hard-chemistry checks
# ---------------------------------------------------------------------------

@dataclass
class ChemistryFlags:
    """Per-frame hard-chemistry check results."""

    broken_bonds: list[tuple[int, int, float]]   # (i, j, distance_Å)
    close_contacts: list[tuple[int, int, float]]  # (i, j, distance_Å)
    ring_planarity_rmsd: float | None             # Å; None when there is no ring

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
    topo: MoleculeTopology,
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

    Ring planarity is None for a molecule without a ring.
    """
    broken: list[tuple[int, int, float]] = []
    for i, j in topo.bonds:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        if d > bond_break_cutoff:
            broken.append((i, j, d))

    bonded_set = set(map(frozenset, topo.bonds))
    close: list[tuple[int, int, float]] = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset({i, j}) in bonded_set:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < close_contact_cutoff:
                close.append((i, j, d))

    planarity = (
        _ring_planarity_rmsd(positions, topo.ring_atoms) if topo.has_ring else None
    )
    return ChemistryFlags(broken, close, planarity)


def check_chemistry_batch(
    positions: np.ndarray,
    topo: MoleculeTopology,
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
    ring_pos = positions[list(ring_indices)]   # (ring_size, 3)
    centroid = ring_pos.mean(axis=0)
    centered = ring_pos - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    distances = np.abs(centered @ normal)
    return float(np.sqrt(np.mean(distances**2)))
