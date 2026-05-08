"""Mahalanobis OOD scorer + threshold calibration.

The scorer is fit on PCA projections of *training* frames only.
Threshold is calibrated on *validation* frames (99th percentile by default).
No PIMD trajectory frames leak into calibration unless explicitly configured.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import mahalanobis


class MahalanobisScorer:
    """Mahalanobis-distance OOD scorer in PCA-reduced space.

    Usage
    -----
    scorer = MahalanobisScorer()
    scorer.fit(X_train_pca)
    scorer.calibrate(X_val_pca, percentile=99.0)
    scores = scorer.score(X_bead_pca)
    flags  = scores > scorer.threshold
    scorer.save("outputs/scorer.pkl")
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self.threshold: float | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train_pca: np.ndarray) -> "MahalanobisScorer":
        """Compute mean and inverse covariance from training PCA projections.

        Parameters
        ----------
        X_train_pca : (n_train, n_components)
        """
        self._mean = X_train_pca.mean(axis=0)
        cov = np.cov(X_train_pca, rowvar=False)
        # Regularise slightly for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-8
        self._inv_cov = np.linalg.inv(cov)
        return self

    # ------------------------------------------------------------------
    def score(self, X_pca: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each row of X_pca.

        Parameters
        ----------
        X_pca : (n_frames, n_components)

        Returns
        -------
        distances : (n_frames,)
        """
        self._check_fitted()
        diff = X_pca - self._mean  # (n_frames, n_components)
        # Vectorised: d² = diff @ inv_cov @ diff.T  →  diag only
        d2 = np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff)
        return np.sqrt(np.maximum(d2, 0.0))

    # ------------------------------------------------------------------
    def calibrate(
        self,
        X_val_pca: np.ndarray,
        percentile: float = 99.0,
    ) -> float:
        """Set threshold as ``percentile``-th percentile of validation scores.

        Parameters
        ----------
        X_val_pca : (n_val, n_components)
        percentile : float in (0, 100]

        Returns
        -------
        threshold : float
        """
        val_scores = self.score(X_val_pca)
        self.threshold = float(np.percentile(val_scores, percentile))
        return self.threshold

    # ------------------------------------------------------------------
    def is_ood(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean array: True where score > threshold.

        Raises if threshold has not been calibrated.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate() first.")
        return scores > self.threshold

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "mean": self._mean,
                "inv_cov": self._inv_cov,
                "threshold": self.threshold,
            }, fh)

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisScorer":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj._mean = state["mean"]
        obj._inv_cov = state["inv_cov"]
        obj.threshold = state["threshold"]
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._mean is None or self._inv_cov is None:
            raise RuntimeError("MahalanobisScorer not fitted. Call fit() first.")
