"""Aggregate per-bead OOD scores into per-timestep summaries.

Per-timestep aggregates (AGENTS.md requirement):
- max bead score
- 95th-percentile bead score
- fraction of beads above OOD threshold
- centroid score (computed separately by the pipeline)

Inputs
------
bead_scores : (n_beads, n_frames) — Mahalanobis scores per bead per timestep
centroid_scores : (n_frames,) — Mahalanobis scores of the centroid trajectory
threshold : float — OOD threshold calibrated on validation set
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_bead_scores(
    bead_scores: np.ndarray,
    centroid_scores: np.ndarray,
    steps: np.ndarray,
    threshold: float,
    frame_time_fs: float = 10.0,
) -> pd.DataFrame:
    """Build per-timestep aggregate table.

    Parameters
    ----------
    bead_scores : (n_beads, n_frames)
    centroid_scores : (n_frames,)
    steps : (n_frames,) simulation step indices
    threshold : OOD threshold
    frame_time_fs : time per frame in femtoseconds (timestep × stride)

    Returns
    -------
    DataFrame with columns:
        step, time_ps, bead_max, bead_p95, bead_frac_ood,
        centroid_score, centroid_ood
    """
    if bead_scores.ndim != 2:
        raise ValueError(f"bead_scores must be 2-D (n_beads, n_frames), got shape {bead_scores.shape}")
    if centroid_scores.shape[0] != bead_scores.shape[1]:
        raise ValueError(
            f"centroid_scores length {centroid_scores.shape[0]} != "
            f"n_frames {bead_scores.shape[1]}"
        )

    bead_max = bead_scores.max(axis=0)
    bead_p95 = np.percentile(bead_scores, 95, axis=0)
    bead_frac_ood = (bead_scores > threshold).mean(axis=0)

    return pd.DataFrame({
        "step": steps.astype(np.int64),
        "time_ps": steps * frame_time_fs / 1000.0,
        "bead_max": bead_max,
        "bead_p95": bead_p95,
        "bead_frac_ood": bead_frac_ood,
        "centroid_score": centroid_scores,
        "centroid_ood": centroid_scores > threshold,
    })


def bead_score_summary(bead_scores: np.ndarray, threshold: float) -> dict:
    """Return scalar summary statistics for a full bead score matrix.

    Parameters
    ----------
    bead_scores : (n_beads, n_frames)
    threshold : OOD threshold

    Returns
    -------
    dict with keys: mean, median, p95, p99, max, frac_ood
    """
    flat = bead_scores.ravel()
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
        "frac_ood": float((flat > threshold).mean()),
    }
