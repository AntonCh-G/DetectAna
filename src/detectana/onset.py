"""Windowed anomaly-onset detector.

Onset definition (AGENTS.md step 9):
    First window W such that fraction of frames in W with score > threshold
    exceeds ``fraction_threshold``.

Onset is reported separately for:
- first_bead_anomaly : first frame where any bead exceeds threshold
- persistent_bead_anomaly : first window where bead_frac_ood > fraction_threshold
- centroid_anomaly : first window where centroid_ood fraction > fraction_threshold
- collective_anomaly : first window where *both* bead and centroid criteria hold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OnsetResult:
    """Onset step indices (simulation steps, not frame indices).

    None means the criterion was never met in the trajectory.
    """

    first_bead_anomaly_step: int | None
    persistent_bead_anomaly_step: int | None
    centroid_anomaly_step: int | None
    collective_anomaly_step: int | None

    # Frame indices (0-based) for the same events — useful for plotting and extraction
    first_bead_anomaly_frame: int | None
    persistent_bead_anomaly_frame: int | None
    centroid_anomaly_frame: int | None
    collective_anomaly_frame: int | None

    # Which bead (0-based index) first exceeded the OOD threshold
    first_anomaly_bead_idx: int | None = None

    def to_dict(self) -> dict:
        return {
            "first_bead_anomaly_step": self.first_bead_anomaly_step,
            "persistent_bead_anomaly_step": self.persistent_bead_anomaly_step,
            "centroid_anomaly_step": self.centroid_anomaly_step,
            "collective_anomaly_step": self.collective_anomaly_step,
            "first_bead_anomaly_frame": self.first_bead_anomaly_frame,
            "persistent_bead_anomaly_frame": self.persistent_bead_anomaly_frame,
            "centroid_anomaly_frame": self.centroid_anomaly_frame,
            "collective_anomaly_frame": self.collective_anomaly_frame,
            "first_anomaly_bead_idx": self.first_anomaly_bead_idx,
        }


def detect_onset(
    aggregate_df: pd.DataFrame,
    threshold: float,
    window_frames: int = 500,
    step_frames: int = 50,
    fraction_threshold: float = 0.20,
) -> OnsetResult:
    """Detect anomaly onset from per-timestep aggregate table.

    Parameters
    ----------
    aggregate_df : output of ``aggregator.aggregate_bead_scores``
        Must contain columns: step, bead_frac_ood, centroid_ood.
    threshold : OOD threshold (used only for display; flags already in df).
    window_frames : number of frames per sliding window.
    step_frames : window step size in frames.
    fraction_threshold : fraction of window that must be OOD to declare onset.

    Returns
    -------
    OnsetResult
    """
    steps = aggregate_df["step"].to_numpy(dtype=np.int64)
    bead_frac = aggregate_df["bead_frac_ood"].to_numpy(dtype=np.float64)
    centroid_ood = aggregate_df["centroid_ood"].to_numpy(dtype=bool)
    n = len(steps)

    # ── First single-frame bead anomaly ────────────────────────────────────
    first_bead_idx = _first_nonzero(bead_frac)
    first_bead_step = int(steps[first_bead_idx]) if first_bead_idx is not None else None

    # ── Windowed onset ─────────────────────────────────────────────────────
    persistent_bead_idx = _windowed_onset(bead_frac, window_frames, step_frames, fraction_threshold)
    persistent_bead_step = int(steps[persistent_bead_idx]) if persistent_bead_idx is not None else None

    centroid_frac = centroid_ood.astype(np.float64)
    centroid_idx = _windowed_onset(centroid_frac, window_frames, step_frames, fraction_threshold)
    centroid_step = int(steps[centroid_idx]) if centroid_idx is not None else None

    # ── Collective onset (both criteria hold in same window) ───────────────
    collective_idx: int | None = None
    for start in range(0, n - window_frames + 1, step_frames):
        end = start + window_frames
        bw = bead_frac[start:end].mean()
        cw = centroid_frac[start:end].mean()
        if bw >= fraction_threshold and cw >= fraction_threshold:
            collective_idx = start
            break
    collective_step = int(steps[collective_idx]) if collective_idx is not None else None

    return OnsetResult(
        first_bead_anomaly_step=first_bead_step,
        persistent_bead_anomaly_step=persistent_bead_step,
        centroid_anomaly_step=centroid_step,
        collective_anomaly_step=collective_step,
        first_bead_anomaly_frame=first_bead_idx,
        persistent_bead_anomaly_frame=persistent_bead_idx,
        centroid_anomaly_frame=centroid_idx,
        collective_anomaly_frame=collective_idx,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_nonzero(arr: np.ndarray) -> int | None:
    """Index of first element > 0, or None."""
    idx = np.flatnonzero(arr > 0)
    return int(idx[0]) if len(idx) > 0 else None


def _windowed_onset(
    signal: np.ndarray,
    window: int,
    step: int,
    frac: float,
) -> int | None:
    """Return the start frame-index of the first window where mean(signal) >= frac."""
    n = len(signal)
    for start in range(0, n - window + 1, step):
        if signal[start : start + window].mean() >= frac:
            return start
    return None
