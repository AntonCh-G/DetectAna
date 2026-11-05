"""Internal-coordinate fingerprint + PCA/standardisation pipeline.

Feature vector per frame
------------------------
1. Bond lengths          (n_bonds,)         Å
2. Bond angles           (n_angles,)        radians
3. Dihedral sines        (n_dihedrals,)     periodic-safe torsion encoding …
4. Dihedral cosines      (n_dihedrals,)     … the other half of it
5. Ring planarity RMSD   (1,)               Å — omitted for a ring-less molecule

The layout comes entirely from the topology, so ``topo.n_features`` and
``topo.feature_names`` are the authority on length and column order.

Standardisation: zero-mean, unit-variance using training-set statistics only.
PCA: fit on standardised training features, retain ``pca_variance`` fraction.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from detectana.registry import DescriptorValues, descriptor
from detectana.topology import MoleculeTopology, _ring_planarity_rmsd

# ---------------------------------------------------------------------------
# Per-frame descriptor
# ---------------------------------------------------------------------------

def compute_descriptor(
    positions: np.ndarray,
    topo: MoleculeTopology,
) -> np.ndarray:
    """Compute internal-coordinate feature vector for one frame.

    Parameters
    ----------
    positions : (n_atoms, 3) Å
    topo : MoleculeTopology

    Returns
    -------
    features : (n_features,)
    """
    feats: list[float] = []

    # ── Bond lengths ────────────────────────────────────────────────────────
    for i, j in topo.bonds:
        feats.append(float(np.linalg.norm(positions[i] - positions[j])))

    # ── Bond angles (radians) ────────────────────────────────────────────────
    for i, j, k in topo.angles:
        v1 = positions[i] - positions[j]
        v2 = positions[k] - positions[j]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        feats.append(np.arccos(cos_a))

    # ── Dihedral torsions: all sines, then all cosines ───────────────────────
    # Same column order as the vectorised batch path and topo.feature_names.
    dih_rad = [
        _dihedral_rad(positions[i], positions[j], positions[k], positions[l])
        for i, j, k, l in topo.dihedrals
    ]
    feats += [np.sin(a) for a in dih_rad]
    feats += [np.cos(a) for a in dih_rad]

    # ── Ring planarity (only when the molecule has a ring) ───────────────────
    if topo.has_ring:
        feats.append(_ring_planarity_rmsd(positions, topo.ring_atoms))

    return np.array(feats, dtype=np.float64)


def compute_descriptor_batch(
    positions: np.ndarray,
    topo: MoleculeTopology,
) -> np.ndarray:
    """Compute descriptors for a batch of frames — fully vectorised.

    Uses precomputed numpy index arrays from ``topo`` (bond_idx, angle_idx,
    dihedral_idx, ring_idx).  Falls back to the per-frame loop when those
    arrays are not yet populated (e.g. during tests that build topo manually).

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3) Å

    Returns
    -------
    features : (n_frames, n_features)
    """
    if topo.bond_idx is None:
        # Fallback: per-frame loop (slow — only for incomplete topology objects)
        return np.stack([compute_descriptor(positions[i], topo) for i in range(len(positions))])

    return _compute_descriptor_batch_vectorised(positions, topo)


def _compute_descriptor_batch_vectorised(
    pos: np.ndarray,
    topo: MoleculeTopology,
) -> np.ndarray:
    """Vectorised internal-coordinate descriptor for (n, n_atoms, 3) positions."""
    bond_idx, angle_idx, dihedral_idx = topo.bond_idx, topo.angle_idx, topo.dihedral_idx
    if bond_idx is None or angle_idx is None or dihedral_idx is None:
        raise ValueError(
            "Topology is missing its index arrays — build it with build_topology()."
        )
    blocks = [
        _bond_lengths(pos, bond_idx),
        _bond_angles(pos, angle_idx),
        _dihedral_sin_cos(pos, dihedral_idx),
    ]
    # Ring planarity RMSD: (n, 1) — skipped for a ring-less molecule
    if topo.has_ring and topo.ring_idx is not None:
        blocks.append(_ring_planarity_batch(pos, topo.ring_idx))
    return np.concatenate(blocks, axis=1)


def _bond_lengths(pos: np.ndarray, bond_idx: np.ndarray) -> np.ndarray:
    """Bond lengths (n, n_bonds), Å."""
    bi, bj = bond_idx[:, 0], bond_idx[:, 1]
    bond_vecs = pos[:, bj] - pos[:, bi]                        # (n, n_bonds, 3)
    return np.linalg.norm(bond_vecs, axis=-1)                  # (n, n_bonds)


def _bond_angles(pos: np.ndarray, angle_idx: np.ndarray) -> np.ndarray:
    """Bond angles (n, n_angles), radians; the middle atom is the vertex."""
    ai, aj, ak = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
    v1 = pos[:, ai] - pos[:, aj]                               # (n, n_angles, 3)
    v2 = pos[:, ak] - pos[:, aj]
    n1 = np.linalg.norm(v1, axis=-1)                           # (n, n_angles)
    n2 = np.linalg.norm(v2, axis=-1)
    cos_a = np.einsum("nid,nid->ni", v1, v2) / (n1 * n2 + 1e-12)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))               # (n, n_angles)


def _dihedral_sin_cos(pos: np.ndarray, dihedral_idx: np.ndarray) -> np.ndarray:
    """Dihedral torsions (n, 2 × n_dihedrals): every sine, then every cosine."""
    di = dihedral_idx
    p0 = pos[:, di[:, 0]]                                      # (n, n_dih, 3)
    p1 = pos[:, di[:, 1]]
    p2 = pos[:, di[:, 2]]
    p3 = pos[:, di[:, 3]]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_len = np.linalg.norm(b1, axis=-1, keepdims=True)       # (n, n_dih, 1)
    b1_hat = b1 / (b1_len + 1e-12)

    # Project b0 and b2 onto plane perpendicular to b1
    v = b0 - np.einsum("nid,nid->ni", b0, b1_hat) [..., None] * b1_hat
    w = b2 - np.einsum("nid,nid->ni", b2, b1_hat) [..., None] * b1_hat

    x = np.einsum("nid,nid->ni", v, w)
    y = np.einsum("nid,nid->ni", np.cross(b1_hat, v), w)
    dih_angles = np.arctan2(y, x)                              # (n, n_dih)
    return np.concatenate([np.sin(dih_angles), np.cos(dih_angles)], axis=1)


def _ring_planarity_batch(pos: np.ndarray, ring_idx: np.ndarray) -> np.ndarray:
    """Ring planarity RMSD (n, 1), Å: distance of ring atoms from their best plane."""
    ring_pos = pos[:, ring_idx]                                 # (n, ring_size, 3)
    centroid = ring_pos.mean(axis=1, keepdims=True)
    centered = ring_pos - centroid                              # (n, ring_size, 3)
    # Batched SVD: last right singular vector is the plane normal
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)     # Vt: (n, 3, 3)
    normals = Vt[:, -1, :]                                      # (n, 3)
    dists = np.abs(np.einsum("nkd,nd->nk", centered, normals))  # (n, ring_size)
    return np.sqrt(np.mean(dists**2, axis=1, keepdims=True))    # (n, 1)


# ---------------------------------------------------------------------------
# Built-in registered descriptors (ADR 0006)
# ---------------------------------------------------------------------------
# Side by side in BUILTIN_DESCRIPTORS order they equal compute_descriptor_batch,
# column for column. Names come from the topology (build_topology /
# complete_topology), so a hand-built topology without names is refused.

def _named(names: list[str], n: int, what: str) -> list[str]:
    if len(names) != n:
        raise ValueError(
            f"topology has {len(names)} {what} names for {n} {what}s; "
            "build it with build_topology() or complete_topology()"
        )
    return list(names)


@descriptor("bonds", helpers=(_bond_lengths, _named))
def bonds(positions: np.ndarray, topology: MoleculeTopology) -> DescriptorValues:
    """Every bonded length, Å. One group per bond."""
    idx = np.asarray(topology.bonds, dtype=np.intp).reshape(-1, 2)
    names = _named(topology.bond_names, len(idx), "bond")
    return DescriptorValues(
        _bond_lengths(np.asarray(positions, dtype=np.float64), idx),
        tuple(names), tuple(map(tuple, idx.tolist())), tuple(names),
    )


@descriptor("angles", helpers=(_bond_angles, _named))
def angles(positions: np.ndarray, topology: MoleculeTopology) -> DescriptorValues:
    """Every bond angle, radians. One group per angle."""
    idx = np.asarray(topology.angles, dtype=np.intp).reshape(-1, 3)
    names = _named(topology.angle_names, len(idx), "angle")
    return DescriptorValues(
        _bond_angles(np.asarray(positions, dtype=np.float64), idx),
        tuple(names), tuple(map(tuple, idx.tolist())), tuple(names),
    )


@descriptor("dihedrals", helpers=(_dihedral_sin_cos, _named))
def dihedrals(positions: np.ndarray, topology: MoleculeTopology) -> DescriptorValues:
    """Every dihedral's sine, then every cosine. A dihedral's sin and cos are one group."""
    idx = np.asarray(topology.dihedrals, dtype=np.intp).reshape(-1, 4)
    names = _named(topology.dihedral_names, 2 * len(idx), "dihedral sin/cos")
    labels = [n.removeprefix("sin_") for n in names[: len(idx)]]
    quads = tuple(map(tuple, idx.tolist()))
    return DescriptorValues(
        _dihedral_sin_cos(np.asarray(positions, dtype=np.float64), idx),
        tuple(names), quads + quads, tuple(labels + labels),
    )


@descriptor("ring", helpers=(_ring_planarity_batch,))
def ring(positions: np.ndarray, topology: MoleculeTopology) -> DescriptorValues:
    """Ring planarity RMSD, Å, one number on the ring atoms; none without a ring."""
    pos = np.asarray(positions, dtype=np.float64)
    if not topology.has_ring:
        return DescriptorValues(np.zeros((len(pos), 0)), (), (), ())
    idx = np.asarray(topology.ring_atoms, dtype=np.intp)
    name = topology.planarity_name
    return DescriptorValues(
        _ring_planarity_batch(pos, idx), (name,), (tuple(idx.tolist()),), (name,),
    )


# ---------------------------------------------------------------------------
# Dihedral helper
# ---------------------------------------------------------------------------

def _dihedral_rad(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Praxitelean dihedral angle for atoms p0-p1-p2-p3 (radians, −π to π)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / (np.linalg.norm(b1) + 1e-12)

    # Project b0 and b2 onto the plane perpendicular to b1
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    return float(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# Standardiser + PCA pipeline
# ---------------------------------------------------------------------------

class DescriptorPipeline:
    """Fit StandardScaler + PCA on training data; transform any data.

    Usage
    -----
    pipe = DescriptorPipeline(pca_variance=0.95, random_seed=42)
    pipe.fit(X_train)          # fit on training descriptors
    X_train_pca = pipe.transform(X_train)
    X_val_pca   = pipe.transform(X_val)
    pipe.save("outputs/pipeline.pkl")
    pipe2 = DescriptorPipeline.load("outputs/pipeline.pkl")
    """

    def __init__(self, pca_variance: float = 0.95, random_seed: int = 42) -> None:
        self.pca_variance = pca_variance
        self.random_seed = random_seed
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> DescriptorPipeline:
        """Fit scaler and PCA on training descriptors.

        Parameters
        ----------
        X_train : (n_train, n_features)
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        self._pca = PCA(
            n_components=self.pca_variance,
            random_state=self.random_seed,
        )
        self._pca.fit(X_scaled)
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise and project to PCA space.

        Parameters
        ----------
        X : (n_frames, n_features)

        Returns
        -------
        X_pca : (n_frames, n_components)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        return self._pca.transform(X_scaled)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    @property
    def n_components(self) -> int:
        self._check_fitted()
        return self._pca.n_components_  # type: ignore[union-attr]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self._pca.explained_variance_ratio_  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"scaler": self._scaler, "pca": self._pca,
                         "pca_variance": self.pca_variance,
                         "random_seed": self.random_seed}, fh)

    @classmethod
    def load(cls, path: str | Path) -> DescriptorPipeline:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(pca_variance=state["pca_variance"], random_seed=state["random_seed"])
        obj._scaler = state["scaler"]
        obj._pca = state["pca"]
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._scaler is None or self._pca is None:
            raise RuntimeError("DescriptorPipeline has not been fitted. Call fit() first.")
