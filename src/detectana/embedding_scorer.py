"""Per-atom Mahalanobis OOD scorer for MLFF invariant per-atom embeddings.

One MahalanobisScorer is fit per atom index, preserving the stable atom
ordering validated against initial.xyz. Per-frame scores are reduced to a
scalar via max over atoms — consistent with the project rule of never
averaging away signals. A single threshold is calibrated on the
max-over-atoms scores from the validation set.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from detectana.scorer import MahalanobisScorer


class EmbeddingPipeline:
    """Per-atom embedding OOD scorer built on MLFF invariant per-atom features.

    Usage
    -----
    pipe = EmbeddingPipeline()
    pipe.fit(ref_train_embeddings)      # (n_frames, n_atoms, n_features)
    pipe.calibrate(ref_val_embeddings)  # sets self.threshold
    scores = pipe.score(embeddings)     # (n_frames,) — max over atoms
    flags  = pipe.is_ood(scores)
    pipe.save("outputs/models/embedding_pipeline.pkl")
    """

    def __init__(self) -> None:
        self._atom_scorers: list[MahalanobisScorer] | None = None
        self.threshold: float | None = None
        self.n_atoms: int | None = None
        self.n_features: int | None = None

    # ------------------------------------------------------------------
    def fit(self, embeddings: np.ndarray) -> EmbeddingPipeline:
        """Fit one Mahalanobis scorer per atom from training embeddings.

        Parameters
        ----------
        embeddings : (n_frames, n_atoms, n_features)
        """
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected (n_frames, n_atoms, n_features), got shape {embeddings.shape}"
            )
        _, n_atoms, n_features = embeddings.shape
        self.n_atoms = n_atoms
        self.n_features = n_features
        self._atom_scorers = [
            MahalanobisScorer().fit(embeddings[:, i, :])
            for i in range(n_atoms)
        ]
        return self

    # ------------------------------------------------------------------
    def score_per_atom(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-frame per-atom Mahalanobis distances.

        Parameters
        ----------
        embeddings : (n_frames, n_atoms, n_features)

        Returns
        -------
        scores : (n_frames, n_atoms)
        """
        atom_scorers = self._fitted()
        n_frames, n_atoms, _ = embeddings.shape
        if n_atoms != self.n_atoms:
            raise ValueError(f"Expected {self.n_atoms} atoms, got {n_atoms}")
        out = np.empty((n_frames, n_atoms), dtype=np.float64)
        for i, scorer in enumerate(atom_scorers):
            out[:, i] = scorer.score(embeddings[:, i, :])
        return out

    # ------------------------------------------------------------------
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-frame OOD score: max over atoms.

        Parameters
        ----------
        embeddings : (n_frames, n_atoms, n_features)

        Returns
        -------
        scores : (n_frames,)
        """
        return self.score_per_atom(embeddings).max(axis=1)

    # ------------------------------------------------------------------
    def calibrate(
        self,
        val_embeddings: np.ndarray,
        percentile: float = 99.0,
    ) -> float:
        """Set threshold as percentile of max-over-atoms scores on validation set.

        Parameters
        ----------
        val_embeddings : (n_val_frames, n_atoms, n_features)
        percentile : float in (0, 100]

        Returns
        -------
        threshold : float
        """
        val_scores = self.score(val_embeddings)
        self.threshold = float(np.percentile(val_scores, percentile))
        return self.threshold

    # ------------------------------------------------------------------
    def is_ood(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean array: True where score > threshold."""
        if self.threshold is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate() first.")
        return scores > self.threshold

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "atom_scorers": self._atom_scorers,
                    "threshold": self.threshold,
                    "n_atoms": self.n_atoms,
                    "n_features": self.n_features,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingPipeline:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj._atom_scorers = state["atom_scorers"]
        obj.threshold = state["threshold"]
        obj.n_atoms = state["n_atoms"]
        obj.n_features = state["n_features"]
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        self._fitted()

    def _fitted(self) -> list[MahalanobisScorer]:
        """Return the per-atom scorers, raising if ``fit()`` was never called."""
        if self._atom_scorers is None:
            raise RuntimeError("EmbeddingPipeline not fitted. Call fit() first.")
        return self._atom_scorers
