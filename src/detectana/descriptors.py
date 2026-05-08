"""Internal-coordinate fingerprint + PCA/standardisation pipeline.

Feature vector per frame
------------------------
1. Bond lengths          (n_bonds,)         Å
2. Bond angles           (n_angles,)        radians
3. Dihedral torsions     (2 × n_dihedrals,) [sin θ, cos θ] pairs — periodic-safe
4. Ring planarity RMSD   (1,)               Å

Standardisation: zero-mean, unit-variance using training-set statistics only.
PCA: fit on standardised training features, retain ``pca_variance`` fraction.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from detectana.topology import AspirinTopology, _ring_planarity_rmsd


# ---------------------------------------------------------------------------
# Per-frame descriptor
# ---------------------------------------------------------------------------

def compute_descriptor(
    positions: np.ndarray,
    topo: AspirinTopology,
) -> np.ndarray:
    """Compute internal-coordinate feature vector for one frame.

    Parameters
    ----------
    positions : (n_atoms, 3) Å
    topo : AspirinTopology

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

    # ── Dihedral torsions (sin, cos) ─────────────────────────────────────────
    for i, j, k, l in topo.dihedrals:
        angle_rad = _dihedral_rad(positions[i], positions[j], positions[k], positions[l])
        feats.append(np.sin(angle_rad))
        feats.append(np.cos(angle_rad))

    # ── Ring planarity ────────────────────────────────────────────────────────
    feats.append(_ring_planarity_rmsd(positions, topo.ring_atoms))

    return np.array(feats, dtype=np.float64)


def compute_descriptor_batch(
    positions: np.ndarray,
    topo: AspirinTopology,
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
    topo: AspirinTopology,
) -> np.ndarray:
    """Vectorised internal-coordinate descriptor for (n, n_atoms, 3) positions."""
    # ── Bond lengths: (n, n_bonds) ───────────────────────────────────────────
    bi, bj = topo.bond_idx[:, 0], topo.bond_idx[:, 1]
    bond_vecs = pos[:, bj] - pos[:, bi]                        # (n, n_bonds, 3)
    bond_lengths = np.linalg.norm(bond_vecs, axis=-1)          # (n, n_bonds)

    # ── Bond angles: (n, n_angles) ───────────────────────────────────────────
    ai, aj, ak = topo.angle_idx[:, 0], topo.angle_idx[:, 1], topo.angle_idx[:, 2]
    v1 = pos[:, ai] - pos[:, aj]                               # (n, n_angles, 3)
    v2 = pos[:, ak] - pos[:, aj]
    n1 = np.linalg.norm(v1, axis=-1)                           # (n, n_angles)
    n2 = np.linalg.norm(v2, axis=-1)
    cos_a = np.einsum("nid,nid->ni", v1, v2) / (n1 * n2 + 1e-12)
    angles = np.arccos(np.clip(cos_a, -1.0, 1.0))             # (n, n_angles)

    # ── Dihedral torsions: (n, 2 × n_dihedrals) ─────────────────────────────
    di = topo.dihedral_idx
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
    sin_cos = np.concatenate(
        [np.sin(dih_angles), np.cos(dih_angles)], axis=1
    )                                                          # (n, 2*n_dih)

    # ── Ring planarity RMSD: (n, 1) ──────────────────────────────────────────
    ring_pos = pos[:, topo.ring_idx]                           # (n, 6, 3)
    centroid = ring_pos.mean(axis=1, keepdims=True)
    centered = ring_pos - centroid                             # (n, 6, 3)
    # Batched SVD: last right singular vector is the plane normal
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)   # Vt: (n, 3, 3)
    normals = Vt[:, -1, :]                                     # (n, 3)
    dists = np.abs(np.einsum("nkd,nd->nk", centered, normals))  # (n, 6)
    planarity = np.sqrt(np.mean(dists**2, axis=1, keepdims=True))  # (n, 1)

    return np.concatenate([bond_lengths, angles, sin_cos, planarity], axis=1)


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
    def fit(self, X_train: np.ndarray) -> "DescriptorPipeline":
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
    def inverse_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return per-original-feature squared contribution to PCA score.

        Useful for identifying which internal coordinate drives an anomaly.

        Parameters
        ----------
        X : (n_frames, n_features) raw descriptors

        Returns
        -------
        contributions : (n_frames, n_features)
            Each row sums to approximately the squared Mahalanobis distance
            in PCA space (up to the variance explained).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        # Project to PCA then reconstruct back — residual carries the variance
        X_pca = self._pca.transform(X_scaled)  # type: ignore[union-attr]
        X_recon = self._pca.inverse_transform(X_pca)  # type: ignore[union-attr]
        return (X_scaled - X_recon) ** 2

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
    def load(cls, path: str | Path) -> "DescriptorPipeline":
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
