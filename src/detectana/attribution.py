"""Reconstruction-based contribution (RBC; Alcala & Qin, Automatica 45(7), 2009).

Both scores of a track are quadratic forms in the standardised descriptor z:

    T²  = zᵀ M_T2 z,   M_T2  = Pᵀ S⁻¹ P   (P kept PCA directions, S their covariance)
    SPE = zᵀ M_SPE z,  M_SPE = I − PᵀP

The RBC of a descriptor group g (columns ξ) is how much the score drops when the
group's numbers are reconstructed — moved to the values that make the score as
small as possible, everything else held fixed:

    RBC_g = (ξᵀ M z)ᵀ (ξᵀ M ξ)⁺ (ξᵀ M z)

RBC values are *attribution strengths*. They overlap and do not sum to the score
(for M = [[1,1],[1,1]], z = [1,1] the score is 4 and each single RBC is 4), so
they are never shares or causes (ADR 0007). A larger group can recover more of
the score, so compare groups of similar size.

numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def quadratic_score(z: np.ndarray, M: np.ndarray) -> np.ndarray | float:
    """zᵀ M z for one vector (m,) or each row of (n, m)."""
    Z = np.asarray(z, dtype=np.float64)
    if Z.ndim == 1:
        return float(Z @ M @ Z)
    return np.einsum("ni,ij,nj->n", Z, M, Z)


def reconstruction_contributions(
    z: np.ndarray, M: np.ndarray, groups: Sequence[Sequence[int]]
) -> np.ndarray:
    """RBC of each group, for one vector z (m,) → (G,) or rows (n, m) → (n, G).

    groups : column indices of each group; a dihedral's sin and cos go together.
    A group the score cannot see (ξᵀMξ = 0) contributes 0.
    """
    Z = np.asarray(z, dtype=np.float64)
    single = Z.ndim == 1
    Z = np.atleast_2d(Z)
    M = np.asarray(M, dtype=np.float64)
    out = np.empty((Z.shape[0], len(groups)))
    MZ = Z @ M  # (n, m); M is symmetric
    for g, cols in enumerate(groups):
        cols = list(cols)
        if not cols:
            raise ValueError(f"group {g} is empty")
        B = MZ[:, cols]                              # (n, |g|) = ξᵀ M z per row
        A_pinv = np.linalg.pinv(M[np.ix_(cols, cols)], hermitian=True)
        out[:, g] = np.maximum(np.einsum("ni,ij,nj->n", B, A_pinv, B), 0.0)
    return out[0] if single else out


def atom_strength(
    strength: np.ndarray, group_atoms: Sequence[Sequence[int]], n_atoms: int
) -> np.ndarray:
    """Per atom, the largest group strength among groups involving it (0 if none).

    strength : (n, G) or (G,). Returns (n, n_atoms) or (n_atoms,). A pointer for
    colouring atoms in score units; the explanation itself is the group.
    """
    S = np.asarray(strength, dtype=np.float64)
    single = S.ndim == 1
    S = np.atleast_2d(S)
    out = np.zeros((S.shape[0], n_atoms))
    for g, atoms in enumerate(group_atoms):
        idx = list(atoms)
        out[:, idx] = np.maximum(out[:, idx], S[:, [g]])
    return out[0] if single else out


@dataclass(frozen=True)
class GroupStrength:
    """One descriptor group's attribution strength for one frame."""

    group_id: str
    atoms: tuple[int, ...]
    size: int
    strength: float


@dataclass(frozen=True)
class Attribution:
    """RBC of every descriptor group of one track's score, for a batch of frames.

    Attribution strengths overlap and do not sum to the score; never present
    them as shares. A larger group (``group_sizes``) tends to show a larger
    strength, so compare groups of similar size.

    values : (n,) the score itself (T² or SPE)
    thresholds : the score's thresholds by role
    strength : (n, G) RBC per group
    atom_strength : (n, n_atoms) per atom, the largest strength of a group involving it
    """

    track: str
    score: str
    values: np.ndarray
    thresholds: dict[str, float]
    group_ids: tuple[str, ...]
    group_atoms: tuple[tuple[int, ...], ...]
    group_sizes: tuple[int, ...]
    strength: np.ndarray
    atom_strength: np.ndarray

    def top(self, frame: int = 0, k: int = 5) -> list[GroupStrength]:
        """The ``k`` strongest groups of one frame, strongest first."""
        row = self.strength[frame]
        order = np.argsort(row, kind="stable")[::-1][:k]
        return [
            GroupStrength(self.group_ids[g], self.group_atoms[g], self.group_sizes[g], float(row[g]))
            for g in order
        ]
