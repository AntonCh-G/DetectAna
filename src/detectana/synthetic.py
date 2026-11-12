"""Synthetic demo frames for aspirin — not simulation output.

Two kinds of motion around one equilibrium geometry, both with the two flexible
torsions (carboxyl, ester) drawn from wrapped normals and a random orientation:

``"rattle"``
    every atom displaced independently (element-dependent σ). What
    ``data/smoke`` is built from; correlations between bonds, angles and
    dihedrals come only from shared atoms, not from physics.
``"harmonic"``
    exact classical thermal sampling (300 K) of a generic valence spring model
    built from the topology: bond, angle and torsion springs with textbook-scale
    force constants. Atoms move together along normal modes, so a C–H stretch
    comes with H–C–X angle changes, as in real molecules. Still not real MD:
    generic constants, purely harmonic, no quantum effects.

Positions in Å, energies in eV. Pipeline-side (imports ``evaluation``); the
core library does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detectana.descriptors import _dihedral_rad
from detectana.evaluation import perturb_dihedral
from detectana.topology import MoleculeTopology

# Torsions left free to move. Everything else is held by the rattle or springs.
CARBOXYL = (6, 5, 10, 7)
ESTER = (5, 6, 12, 11)

# Spread of each free torsion in the reference set, degrees.
CARBOXYL_SIGMA_DEG = 12.0
ESTER_SIGMA_DEG = 25.0

# Rattle amplitude per atom, Å. Hydrogens move further at the same temperature.
RATTLE_HEAVY_A = 0.035
RATTLE_H_A = 0.070

# Seeds of the shipped smoke set (scripts/make_demo_data.py).
SEED_TRAIN = 20260810
SEED_VALID = 20260811
SEED_TRAJ = 20260812

#: k_B T at 300 K, eV.
KT_300K_EV = 0.025852

# Generic force constants, E = ½ k (q − q0)². Order of magnitude of common
# force fields; not fitted to aspirin.
K_BOND_XH = 30.0        # eV/Å²   bonds to hydrogen
K_BOND_HEAVY = 40.0     # eV/Å²   heavy-atom bonds
K_ANGLE_XH = 3.0        # eV/rad² angles with a terminal hydrogen
K_ANGLE_HEAVY = 5.0     # eV/rad² other angles
K_TORSION_RING = 2.0    # eV/rad² torsions inside the ring (keeps it planar)
K_TORSION = 0.3         # eV/rad² other torsions


@dataclass(frozen=True)
class HarmonicModel:
    """Cartesian Hessian of the spring model and its vibrational normal modes.

    hessian : (3N, 3N) eV/Å², = Bᵀ K B at the equilibrium geometry
    modes : (3N − 6, 3N) unit normal-mode vectors (rigid motions removed)
    mode_amplitudes : (3N − 6,) Å, thermal RMS amplitude √(kT/λ) of each mode
    """

    hessian: np.ndarray
    modes: np.ndarray
    mode_amplitudes: np.ndarray

    def sample_displacements(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, N, 3) Cartesian displacements drawn from the thermal distribution."""
        g = rng.normal(size=(n, len(self.mode_amplitudes)))
        return ((g * self.mode_amplitudes) @ self.modes).reshape(n, -1, 3)


def harmonic_model(
    equilibrium: np.ndarray, topo: MoleculeTopology, *, kT: float = KT_300K_EV
) -> HarmonicModel:
    """Build the spring model's Hessian from bonds, angles and dihedrals."""
    eq = np.asarray(equilibrium, dtype=np.float64)
    sym = topo.atom_types
    ring = set(topo.ring_atoms)
    coords: list[tuple[str, tuple[int, ...], float]] = []
    for i, j in topo.bonds:
        coords.append(("bond", (i, j), K_BOND_XH if "H" in (sym[i], sym[j]) else K_BOND_HEAVY))
    for i, j, k in topo.angles:
        coords.append(("angle", (i, j, k), K_ANGLE_XH if "H" in (sym[i], sym[k]) else K_ANGLE_HEAVY))
    for quad in topo.dihedrals:
        coords.append(("dihedral", tuple(quad), K_TORSION_RING if set(quad) <= ring else K_TORSION))

    B = np.stack([_gradient(kind, atoms, eq) for kind, atoms, _ in coords])  # (n_coords, 3N)
    K = np.array([k for *_, k in coords])
    H = B.T @ (K[:, None] * B)
    lam, vec = np.linalg.eigh(H)
    keep = lam > 1e-6 * lam.max()
    if int(np.sum(~keep)) != 6:
        raise ValueError(f"spring model has {int(np.sum(~keep))} zero modes; expected the 6 rigid motions")
    return HarmonicModel(H, vec[:, keep].T, np.sqrt(kT / lam[keep]))


def reference_frames(
    equilibrium: np.ndarray,
    topo: MoleculeTopology,
    n_frames: int,
    seed: int,
    *,
    motion: str = "rattle",
) -> np.ndarray:
    """``n_frames`` synthetic reference frames, (n_frames, N, 3) Å.

    ``motion="rattle"`` reproduces ``data/smoke`` exactly for its seeds.
    ``motion="harmonic"`` vibrates the equilibrium geometry along the spring
    model's normal modes, then turns the two free torsions.
    """
    if motion not in ("rattle", "harmonic"):
        raise ValueError(f"motion must be 'rattle' or 'harmonic', got {motion!r}")
    eq = np.asarray(equilibrium, dtype=np.float64)
    rng = np.random.default_rng(seed)
    model = harmonic_model(eq, topo) if motion == "harmonic" else None
    frames = []
    for _ in range(n_frames):
        pos = eq.copy()
        if model is not None:
            pos = pos + model.sample_displacements(1, rng)[0]
        for dihedral, sigma in ((CARBOXYL, CARBOXYL_SIGMA_DEG), (ESTER, ESTER_SIGMA_DEG)):
            pos = perturb_dihedral(pos, topo, float(rng.normal(scale=sigma)), dihedral=dihedral)
        if model is None:
            pos = rattle(pos, topo.atom_types, rng)
        frames.append(random_orientation(pos, rng))
    return np.stack(frames)


def rattle(pos: np.ndarray, symbols: list[str], rng: np.random.Generator) -> np.ndarray:
    sigma = np.array(
        [RATTLE_H_A if s == "H" else RATTLE_HEAVY_A for s in symbols],
        dtype=np.float64,
    )[:, None]
    return pos + rng.normal(scale=sigma, size=pos.shape)


def random_orientation(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rotation about the centre, plus a small translation.

    The descriptors are internal coordinates and so invariant to this; it makes
    the files look like trajectory output rather than aligned copies.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    centre = pos.mean(axis=0)
    return (pos - centre) @ rot.T + centre + rng.normal(scale=0.05, size=3)


def _internal(kind: str, atoms: tuple[int, ...], pos: np.ndarray) -> float:
    p = [pos[a] for a in atoms]
    if kind == "bond":
        return float(np.linalg.norm(p[0] - p[1]))
    if kind == "angle":
        v1, v2 = p[0] - p[1], p[2] - p[1]
        return float(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    return _dihedral_rad(*p)


def _gradient(kind: str, atoms: tuple[int, ...], pos: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Row of the Wilson B matrix: d(coordinate)/d(Cartesian), central differences."""
    row = np.zeros(pos.size)
    for a in atoms:
        for d in range(3):
            plus, minus = pos.copy(), pos.copy()
            plus[a, d] += h
            minus[a, d] -= h
            diff = _internal(kind, atoms, plus) - _internal(kind, atoms, minus)
            if kind == "dihedral":
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
            row[3 * a + d] = diff / (2 * h)
    return row
