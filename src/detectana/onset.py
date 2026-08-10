"""Windowed anomaly-onset detector.

Onset definition (workflow step 9):
    First window W such that fraction of frames in W with score > threshold
    exceeds ``fraction_threshold``.

Onset is reported separately for:
- first_bead_anomaly : first frame where any bead exceeds threshold
- persistent_bead_anomaly : first window where bead_frac_ood > fraction_threshold
- centroid_anomaly : first window where centroid_ood fraction > fraction_threshold
- collective_anomaly : first window where *both* bead and centroid criteria hold

Choosing the window rule
------------------------
``fraction_threshold`` is what actually controls false alarms, not the OOD
threshold. A threshold calibrated to flag a fraction α of in-distribution frames
flags, by construction, about α of the frames in any long run: at α = 1 % over
200 000 frames that is ~2000 flagged frames per bead, and the first of them
arrives after ~100 frames. ``first_bead_anomaly`` is therefore a property of the
threshold, not of the trajectory, and only the windowed criteria carry evidence.

The helpers below turn that into a design choice: given α, a window length and a
run-level false-alarm budget, ``choose_fraction_threshold`` returns the *loosest*
fraction that stays inside the budget, which is the most sensitive rule that
still controls false alarms. ``onset_design_report`` reports the numbers for a
rule you have already fixed. Both work on the null hypothesis that flags inside a
window are Bernoulli(α); the frame-to-frame correlation of a trajectory is
handled by an effective sample size, and window overlap by a union bound, so the
reported probability is an upper bound rather than an estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

log = logging.getLogger(__name__)


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

    # ── Embedding OOD track ────────────────────────────────────────────────
    # None when embedding track is disabled or the criterion was never met.
    # Frame indices are relative to the embedding-scored subset (not the full
    # trajectory), because embedding inference may cover only a strided
    # frame range.
    embedding_persistent_bead_onset_step: int | None = None
    embedding_persistent_bead_onset_frame: int | None = None
    embedding_centroid_onset_step: int | None = None
    embedding_centroid_onset_frame: int | None = None

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
            "embedding_persistent_bead_onset_step": self.embedding_persistent_bead_onset_step,
            "embedding_persistent_bead_onset_frame": self.embedding_persistent_bead_onset_frame,
            "embedding_centroid_onset_step": self.embedding_centroid_onset_step,
            "embedding_centroid_onset_frame": self.embedding_centroid_onset_frame,
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

    # ── Embedding onset (optional — only when columns are present) ────────────
    emb_pb_step = emb_pb_frame = emb_c_step = emb_c_frame = None
    if "emb_bead_frac_ood" in aggregate_df.columns:
        valid = aggregate_df["emb_bead_frac_ood"].notna()
        emb_sub = aggregate_df.loc[valid].reset_index(drop=True)
        orig_indices = np.where(valid.to_numpy())[0]

        emb_steps_sub = emb_sub["step"].to_numpy(dtype=np.int64)
        emb_bead_frac = emb_sub["emb_bead_frac_ood"].to_numpy(dtype=np.float64)
        emb_cent_ood = emb_sub["emb_centroid_ood"].to_numpy(dtype=np.float64)

        emb_pb_idx = _windowed_onset(emb_bead_frac, window_frames, step_frames, fraction_threshold)
        if emb_pb_idx is not None:
            emb_pb_step = int(emb_steps_sub[emb_pb_idx])
            emb_pb_frame = int(orig_indices[emb_pb_idx])

        emb_c_idx = _windowed_onset(emb_cent_ood, window_frames, step_frames, fraction_threshold)
        if emb_c_idx is not None:
            emb_c_step = int(emb_steps_sub[emb_c_idx])
            emb_c_frame = int(orig_indices[emb_c_idx])

    return OnsetResult(
        first_bead_anomaly_step=first_bead_step,
        persistent_bead_anomaly_step=persistent_bead_step,
        centroid_anomaly_step=centroid_step,
        collective_anomaly_step=collective_step,
        first_bead_anomaly_frame=first_bead_idx,
        persistent_bead_anomaly_frame=persistent_bead_idx,
        centroid_anomaly_frame=centroid_idx,
        collective_anomaly_frame=collective_idx,
        embedding_persistent_bead_onset_step=emb_pb_step,
        embedding_persistent_bead_onset_frame=emb_pb_frame,
        embedding_centroid_onset_step=emb_c_step,
        embedding_centroid_onset_frame=emb_c_frame,
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


# ---------------------------------------------------------------------------
# Window-rule design: false-alarm arithmetic
# ---------------------------------------------------------------------------

def n_windows(n_frames: int, window_frames: int, step_frames: int) -> int:
    """Number of windows ``_windowed_onset`` will test over a run."""
    if n_frames < window_frames:
        return 0
    return (n_frames - window_frames) // step_frames + 1


def estimate_lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a score series, clipped to [0, 0.99].

    Negative values are clipped to 0 because they would *raise* the effective
    sample size; assuming independence is the conservative choice there.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    var = float(x.var())
    if var <= 0.0:
        return 0.0
    rho = float((x[:-1] * x[1:]).mean() / var)
    return float(np.clip(rho, 0.0, 0.99))


def effective_window_trials(
    window_frames: int,
    frame_autocorrelation: float = 0.0,
    n_effective_beads: int = 1,
) -> int:
    """Number of independent flag opportunities inside one window.

    Consecutive frames of a trajectory are correlated, so a 500-frame window does
    not carry 500 independent chances to flag. Under an AR(1) approximation the
    effective count is ``window * (1 - ρ) / (1 + ρ)``.

    Parameters
    ----------
    window_frames : window length in frames.
    frame_autocorrelation : lag-1 autocorrelation ρ of the score series. 0 means
        independent frames, which is optimistic for MD data — measure it with
        ``estimate_lag1_autocorrelation`` on a stable segment.
    n_effective_beads : independent beads contributing per frame. The bead
        criterion averages over beads as well as frames, but beads are
        path-integral images of one molecule and are strongly correlated, so the
        default of 1 (treat a timestep as a single observation) is the
        conservative choice. Raise it only with evidence.
    """
    if window_frames < 1:
        raise ValueError(f"window_frames must be >= 1, got {window_frames}")
    if n_effective_beads < 1:
        raise ValueError(f"n_effective_beads must be >= 1, got {n_effective_beads}")
    rho = float(np.clip(frame_autocorrelation, 0.0, 0.99))
    n_eff = window_frames * (1.0 - rho) / (1.0 + rho)
    return max(1, int(round(n_eff * n_effective_beads)))


def window_false_alarm_probability(
    false_flag_rate: float,
    window_frames: int,
    fraction_threshold: float,
    n_windows_tested: int = 1,
    frame_autocorrelation: float = 0.0,
    n_effective_beads: int = 1,
) -> float:
    """Upper bound on P(the window rule fires somewhere in a clean run).

    Null model: each of the ``effective_window_trials`` opportunities in a window
    flags independently with probability ``false_flag_rate``, so the flag count is
    Binomial and the rule fires when it reaches ``fraction_threshold``. Windows
    overlap and are positively dependent, so the union bound over windows makes
    the result an upper bound, not an estimate.

    Parameters
    ----------
    false_flag_rate : α, the fraction of in-distribution frames the OOD threshold
        flags by construction (1 − percentile/100).
    fraction_threshold : the rule's trigger fraction.
    n_windows_tested : windows examined over the run (see ``n_windows``).

    Returns
    -------
    probability : in [0, 1].
    """
    if not 0.0 < false_flag_rate < 1.0:
        raise ValueError(f"false_flag_rate must be in (0, 1), got {false_flag_rate}")
    if not 0.0 < fraction_threshold <= 1.0:
        raise ValueError(
            f"fraction_threshold must be in (0, 1], got {fraction_threshold}"
        )
    trials = effective_window_trials(
        window_frames, frame_autocorrelation, n_effective_beads
    )
    needed = int(np.ceil(fraction_threshold * trials))
    needed = min(max(needed, 1), trials)
    p_window = float(binom.sf(needed - 1, trials, false_flag_rate))
    return float(min(1.0, p_window * max(n_windows_tested, 1)))


def choose_fraction_threshold(
    false_alarm_budget: float,
    false_flag_rate: float,
    window_frames: int,
    n_windows_tested: int = 1,
    frame_autocorrelation: float = 0.0,
    n_effective_beads: int = 1,
) -> tuple[float, float]:
    """Loosest ``fraction_threshold`` whose run-level false-alarm bound fits the budget.

    Sensitivity falls as the fraction rises, so the smallest fraction that meets
    the budget is the best rule available: it declares an onset on the weakest
    evidence the budget allows.

    Returns
    -------
    fraction_threshold : achievable fraction (a multiple of 1/effective trials).
    false_alarm_probability : the bound at that fraction.

    Raises
    ------
    ValueError : when even "every opportunity in the window flags" exceeds the
        budget — tighten the OOD threshold or lengthen the window instead.
    """
    if not 0.0 < false_alarm_budget < 1.0:
        raise ValueError(
            f"false_alarm_budget must be in (0, 1), got {false_alarm_budget}"
        )
    trials = effective_window_trials(
        window_frames, frame_autocorrelation, n_effective_beads
    )
    for needed in range(1, trials + 1):
        p_window = float(binom.sf(needed - 1, trials, false_flag_rate))
        p_run = min(1.0, p_window * max(n_windows_tested, 1))
        if p_run <= false_alarm_budget:
            return needed / trials, p_run
    raise ValueError(
        f"No fraction_threshold meets a false-alarm budget of {false_alarm_budget:.2%} "
        f"with false_flag_rate={false_flag_rate:.2%}, window_frames={window_frames} "
        f"({trials} effective trials) and {n_windows_tested} windows. "
        "Lower the OOD false-flag rate, lengthen the window, or raise the budget."
    )


def onset_design_report(
    false_flag_rate: float,
    window_frames: int,
    step_frames: int,
    n_frames: int,
    fraction_threshold: float,
    frame_autocorrelation: float = 0.0,
    n_effective_beads: int = 1,
) -> dict:
    """Describe what a window rule costs and buys, for logging and the manifest.

    Two run-level numbers are reported, because the union bound over *overlapping*
    windows is loose: windows that share most of their frames are nearly the same
    test, so counting each as independent overstates the false-alarm rate. The
    ``..._per_run`` value counts every window start (conservative, and what
    ``choose_fraction_threshold`` works from); ``..._per_run_nonoverlapping``
    counts only disjoint windows (optimistic). The truth lies between them.
    """
    windows = n_windows(n_frames, window_frames, step_frames)
    disjoint_windows = max(1, n_frames // window_frames)
    trials = effective_window_trials(
        window_frames, frame_autocorrelation, n_effective_beads
    )
    return {
        "false_flag_rate": float(false_flag_rate),
        "window_frames": int(window_frames),
        "step_frames": int(step_frames),
        "fraction_threshold": float(fraction_threshold),
        "frame_autocorrelation": float(frame_autocorrelation),
        "n_effective_beads": int(n_effective_beads),
        "n_windows_tested": int(windows),
        "effective_trials_per_window": int(trials),
        "flags_needed_per_window": int(
            min(max(int(np.ceil(fraction_threshold * trials)), 1), trials)
        ),
        "expected_false_flags_per_frame_series": float(false_flag_rate * n_frames),
        "frames_to_first_false_flag": float(1.0 / false_flag_rate),
        "false_alarm_probability_per_window": window_false_alarm_probability(
            false_flag_rate, window_frames, fraction_threshold, 1,
            frame_autocorrelation, n_effective_beads,
        ),
        "false_alarm_probability_per_run": window_false_alarm_probability(
            false_flag_rate, window_frames, fraction_threshold, windows,
            frame_autocorrelation, n_effective_beads,
        ),
        "false_alarm_probability_per_run_nonoverlapping": window_false_alarm_probability(
            false_flag_rate, window_frames, fraction_threshold, disjoint_windows,
            frame_autocorrelation, n_effective_beads,
        ),
        "n_disjoint_windows": int(disjoint_windows),
    }


def resolve_onset_rule(
    onset_cfg: dict,
    false_flag_rate: float,
    n_frames: int,
    stable_series: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Return the ``fraction_threshold`` to use, plus its design report.

    Two modes, chosen by the config:

    - ``fraction_threshold`` given (default): used as-is; the report says what its
      false-alarm bound is. Nothing changes silently.
    - ``false_alarm_budget`` given: the fraction is derived from the budget, which
      is the recommended way round — you pick the false-alarm rate you can accept
      and get the most sensitive rule that respects it.

    ``frame_autocorrelation`` may be a number or ``"auto"``. ``"auto"`` measures
    the lag-1 autocorrelation of ``stable_series`` — pass the early, believed-quiet
    part of the run, since a series containing the anomaly would overestimate it.
    """
    window_frames = int(onset_cfg["window_frames"])
    step_frames = int(onset_cfg["step_frames"])
    n_effective_beads = int(onset_cfg.get("n_effective_beads", 1))

    rho_cfg = onset_cfg.get("frame_autocorrelation", 0.0)
    if isinstance(rho_cfg, str):
        if rho_cfg != "auto":
            raise ValueError(
                f"onset.frame_autocorrelation must be a number or 'auto', got {rho_cfg!r}"
            )
        rho = (
            estimate_lag1_autocorrelation(stable_series)
            if stable_series is not None and len(stable_series) >= 3
            else 0.0
        )
        log.info("Onset design: measured frame autocorrelation ρ=%.3f", rho)
    else:
        rho = float(rho_cfg)
        if rho == 0.0:
            log.warning(
                "Onset design: frame_autocorrelation=0 assumes independent frames, "
                "which is optimistic for MD. Set it to 'auto' or a measured value "
                "to make the false-alarm bound honest."
            )

    windows = n_windows(n_frames, window_frames, step_frames)
    budget = onset_cfg.get("false_alarm_budget")

    if budget is not None:
        fraction, p_run = choose_fraction_threshold(
            float(budget), false_flag_rate, window_frames, windows, rho,
            n_effective_beads,
        )
        configured = onset_cfg.get("fraction_threshold")
        log.info(
            "Onset design: false_alarm_budget=%.3g → fraction_threshold=%.4f "
            "(run-level false-alarm bound %.3g)", float(budget), fraction, p_run,
        )
        if configured is not None and not np.isclose(float(configured), fraction):
            log.warning(
                "Onset design: false_alarm_budget overrides the configured "
                "fraction_threshold=%.4f (using %.4f).", float(configured), fraction,
            )
    else:
        fraction = float(onset_cfg["fraction_threshold"])

    report = onset_design_report(
        false_flag_rate, window_frames, step_frames, n_frames, fraction, rho,
        n_effective_beads,
    )
    report["derived_from_false_alarm_budget"] = (
        None if budget is None else float(budget)
    )
    log.info(
        "Onset rule: %d/%d effective flags per %d-frame window (fraction %.4f); "
        "false-alarm bound %.3g per run (%.3g counting disjoint windows only)",
        report["flags_needed_per_window"], report["effective_trials_per_window"],
        window_frames, fraction, report["false_alarm_probability_per_run"],
        report["false_alarm_probability_per_run_nonoverlapping"],
    )
    if report["false_alarm_probability_per_run"] < 1e-12:
        log.warning(
            "Onset rule: false-alarm bound is %.1e — far below any useful budget, "
            "so the rule is likely insensitive. Consider setting "
            "onset.false_alarm_budget instead of fraction_threshold.",
            report["false_alarm_probability_per_run"],
        )
    return fraction, report
