"""Mahalanobis OOD scorer + threshold calibration.

The scorer is fit on PCA projections of *training* frames only.
Thresholds are calibrated on *validation* frames (99th percentile by default).
No PIMD trajectory frames leak into calibration unless explicitly configured.

A run may need more than one threshold — a sensitive one for the individual beads
and a strict one for the centroid — so thresholds are kept in a dict keyed by
track name. Calibrating one track never overwrites another, which is what a
single mutable ``self.threshold`` attribute used to do.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DEFAULT_TRACK = "default"


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

    Two tracks, calibrated independently and both kept:

    bead_thr = scorer.calibrate(X_val_pca, percentile=99.0, track="bead")
    cent_thr = scorer.calibrate(X_val_pca, percentile=85.0, track="centroid")
    flags    = scorer.is_ood(scores, track="bead")
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self.thresholds: dict[str, float] = {}

    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float | None:
        """Threshold of the ``"default"`` track, or None if never calibrated.

        Kept so ``scorer.threshold`` still reads as it did before named tracks.
        """
        return self.thresholds.get(DEFAULT_TRACK)

    @threshold.setter
    def threshold(self, value: float | None) -> None:
        if value is None:
            self.thresholds.pop(DEFAULT_TRACK, None)
        else:
            self.thresholds[DEFAULT_TRACK] = float(value)

    # ------------------------------------------------------------------
    def fit(self, X_train_pca: np.ndarray) -> MahalanobisScorer:
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
        mean, inv_cov = self._fitted()
        diff = X_pca - mean  # (n_frames, n_components)
        # Vectorised: d² = diff @ inv_cov @ diff.T  →  diag only
        d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        return np.sqrt(np.maximum(d2, 0.0))

    # ------------------------------------------------------------------
    def calibrate(
        self,
        X_val_pca: np.ndarray,
        percentile: float = 99.0,
        track: str = DEFAULT_TRACK,
    ) -> float:
        """Store the ``percentile``-th percentile of validation scores for ``track``.

        Calibrating a track leaves every other track alone, so a bead and a
        centroid threshold can coexist on one scorer and a saved pickle carries
        both.

        Parameters
        ----------
        X_val_pca : (n_val, n_components)
        percentile : float in (0, 100]
        track : name this threshold is stored under, e.g. ``"bead"``.

        Returns
        -------
        threshold : float
        """
        val_scores = self.score(X_val_pca)
        threshold = float(np.percentile(val_scores, percentile))
        self.thresholds[track] = threshold
        return threshold

    # ------------------------------------------------------------------
    def get_threshold(self, track: str = DEFAULT_TRACK) -> float:
        """Return the threshold of ``track``, raising if it was never calibrated."""
        if track not in self.thresholds:
            known = sorted(self.thresholds) or ["none"]
            raise RuntimeError(
                f"Threshold for track {track!r} not calibrated "
                f"(calibrated tracks: {', '.join(known)}). Call calibrate() first."
            )
        return self.thresholds[track]

    # ------------------------------------------------------------------
    def is_ood(self, scores: np.ndarray, track: str = DEFAULT_TRACK) -> np.ndarray:
        """Return boolean array: True where score > the ``track`` threshold.

        Raises if that track has not been calibrated.
        """
        return scores > self.get_threshold(track)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "mean": self._mean,
                "inv_cov": self._inv_cov,
                # Both keys are written: "thresholds" is the real state, and
                # "threshold" keeps older readers of the pickle working.
                "thresholds": dict(self.thresholds),
                "threshold": self.threshold,
            }, fh)

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisScorer:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj._mean = state["mean"]
        obj._inv_cov = state["inv_cov"]
        # Pickles written before named tracks only carry a single "threshold".
        thresholds = state.get("thresholds")
        if thresholds:
            obj.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        elif state.get("threshold") is not None:
            obj.thresholds = {DEFAULT_TRACK: float(state["threshold"])}
        return obj

    # ------------------------------------------------------------------
    # Pickle hooks. ``threshold`` used to be a plain attribute, so instances
    # pickled before named tracks (directly, or nested inside an
    # EmbeddingPipeline) carry it in their state dict and no ``thresholds``.
    def __setstate__(self, state: dict) -> None:
        self._mean = state.get("_mean")
        self._inv_cov = state.get("_inv_cov")
        thresholds = state.get("thresholds")
        if thresholds:
            self.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        else:
            self.thresholds = {}
            legacy = state.get("threshold")
            if legacy is not None:
                self.thresholds[DEFAULT_TRACK] = float(legacy)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        self._fitted()

    def _fitted(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, inv_cov)``, raising if ``fit()`` was never called."""
        if self._mean is None or self._inv_cov is None:
            raise RuntimeError("MahalanobisScorer not fitted. Call fit() first.")
        return self._mean, self._inv_cov
