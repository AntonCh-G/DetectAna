"""Detector evaluation: does the OOD score do anything useful?

Two questions this module answers, both of which the scoring code cannot.

**Can the detector separate normal from abnormal?**
``detection_metrics`` compares scores on frames known to be in-distribution
against scores on frames known not to be, and reports AUROC, average precision
and — the operationally meaningful one — the detection rate at a fixed false-flag
rate. The threshold for that last number is the conformal order statistic of the
in-distribution scores, the same rule the pipeline calibrates with, so the number
answers "of the abnormal frames, how many would this run have flagged?".

**Does a high score mean the force field is actually wrong there?**
``score_error_correlation`` and ``error_by_score_decile`` relate the score to a
per-frame model error. This is the question that decides whether the detector is
useful for anything: an OOD score that does not track model error is a curiosity,
and one that does is a reliability estimate. It needs forces from the reference
method, so it is driven by ``scripts/score_vs_error.py`` rather than by the
pipeline.

Labelled abnormal frames are usually unavailable, so the perturbation helpers at
the bottom manufacture them: Gaussian rattle, single-bond stretch, and dihedral
rotation. Rotation is the one worth watching — it is a conformational change with
every bond length and angle left intact, which is what a real anomaly in a
flexible molecule tends to look like, and it is the case a bond-length-based check
misses.

Nothing here is used while scoring a trajectory. Keeping it separate is
deliberate: evaluation data must never touch the fit or the threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection performance
# ---------------------------------------------------------------------------

def conformal_threshold(scores_in_distribution: np.ndarray, false_flag_rate: float) -> float:
    """Threshold flagging at most ``false_flag_rate`` of in-distribution frames.

    The k-th largest score with k = ceil((n+1)·α), which under exchangeability
    bounds the false-flag probability by α for a new in-distribution frame. Same
    rule the pipeline calibrates with, so detection rates computed against it
    describe the deployed detector rather than an idealised one.
    """
    scores = np.asarray(scores_in_distribution, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Need at least one in-distribution score")
    if not 0.0 < false_flag_rate < 1.0:
        raise ValueError(f"false_flag_rate must be in (0, 1), got {false_flag_rate}")
    k = int(np.ceil((len(scores) + 1) * false_flag_rate))
    k = min(max(k, 1), len(scores))
    return float(np.sort(scores)[-k])


def detection_metrics(
    scores_in_distribution: np.ndarray,
    scores_abnormal: np.ndarray,
    false_flag_rate: float = 0.01,
) -> dict:
    """Separation between in-distribution and abnormal scores.

    Parameters
    ----------
    scores_in_distribution : scores on frames that should not be flagged. These
        must be held out from the fit and from threshold calibration.
    scores_abnormal : scores on frames that should be flagged.
    false_flag_rate : α at which the detection rate is reported.

    Returns
    -------
    dict with auroc, average_precision, detection_rate (recall at the α
    threshold), threshold, the median of each score group, and group sizes.
    AUROC and average precision are None when either group is empty.
    """
    ok = np.asarray(scores_in_distribution, dtype=np.float64)
    bad = np.asarray(scores_abnormal, dtype=np.float64)

    result = {
        "n_in_distribution": int(ok.size),
        "n_abnormal": int(bad.size),
        "median_score_in_distribution": float(np.median(ok)) if ok.size else None,
        "median_score_abnormal": float(np.median(bad)) if bad.size else None,
        "false_flag_rate": float(false_flag_rate),
        "auroc": None,
        "average_precision": None,
        "detection_rate": None,
        "threshold": None,
    }
    if ok.size == 0 or bad.size == 0:
        return result

    labels = np.concatenate([np.zeros(ok.size, dtype=int), np.ones(bad.size, dtype=int)])
    scores = np.concatenate([ok, bad])
    result["auroc"] = float(roc_auc_score(labels, scores))
    result["average_precision"] = float(average_precision_score(labels, scores))

    threshold = conformal_threshold(ok, false_flag_rate)
    result["threshold"] = threshold
    result["detection_rate"] = float((bad > threshold).mean())
    return result


# ---------------------------------------------------------------------------
# Score against model error
# ---------------------------------------------------------------------------

def per_frame_force_error(
    forces_reference: np.ndarray,
    forces_predicted: np.ndarray,
    metric: str = "mae",
) -> np.ndarray:
    """Per-frame force error between reference and predicted forces.

    Parameters
    ----------
    forces_reference, forces_predicted : (n_frames, n_atoms, 3) in the same units,
        conventionally eV/Å. i-PI writes Hartree/Bohr; convert before calling
        (1 Hartree/Bohr = 51.4221 eV/Å).
    metric : "mae" (mean absolute component error), "rmse", or "max" (largest
        per-atom force-vector norm error, the one that catches a single bad atom).

    Returns
    -------
    errors : (n_frames,) in the input force units.
    """
    ref = np.asarray(forces_reference, dtype=np.float64)
    pred = np.asarray(forces_predicted, dtype=np.float64)
    if ref.shape != pred.shape:
        raise ValueError(f"Force shape mismatch: {ref.shape} vs {pred.shape}")
    if ref.ndim != 3 or ref.shape[-1] != 3:
        raise ValueError(f"Expected (n_frames, n_atoms, 3), got {ref.shape}")

    diff = pred - ref
    if metric == "mae":
        return np.abs(diff).mean(axis=(1, 2))
    if metric == "rmse":
        return np.sqrt((diff**2).mean(axis=(1, 2)))
    if metric == "max":
        return np.linalg.norm(diff, axis=-1).max(axis=1)
    raise ValueError(f"Unknown metric {metric!r}; use 'mae', 'rmse' or 'max'")


def score_error_correlation(scores: np.ndarray, errors: np.ndarray) -> dict:
    """Relate OOD score to model error.

    Spearman is the headline: the claim worth making is "higher score ranks
    higher error", which is monotone but not linear. Pearson is reported too,
    because a large gap between the two says the relationship is driven by a few
    extreme frames.
    """
    s = np.asarray(scores, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if s.shape != e.shape:
        raise ValueError(f"Length mismatch: {s.shape} vs {e.shape}")
    finite = np.isfinite(s) & np.isfinite(e)
    s, e = s[finite], e[finite]
    if len(s) < 3:
        raise ValueError(f"Need at least 3 paired finite values, got {len(s)}")

    rho, rho_p = spearmanr(s, e)
    r, r_p = pearsonr(s, e)
    return {
        "n": int(len(s)),
        "spearman": float(rho),
        "spearman_p": float(rho_p),
        "pearson": float(r),
        "pearson_p": float(r_p),
    }


def error_by_score_decile(
    scores: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Mean model error per score quantile bin — the table to put in a report.

    Reads directly: if the top bin's error is many times the bottom bin's, the
    score is informative about model reliability; if the column is flat, it is not.
    Bins are equal-count quantiles, so ties can make them uneven and empty bins
    are dropped.
    """
    s = np.asarray(scores, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if s.shape != e.shape:
        raise ValueError(f"Length mismatch: {s.shape} vs {e.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    order = np.argsort(s, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append({
            "bin": i,
            "n": len(idx),
            "score_min": float(s[idx].min()),
            "score_max": float(s[idx].max()),
            "error_mean": float(e[idx].mean()),
            "error_median": float(np.median(e[idx])),
            "error_p95": float(np.percentile(e[idx], 95)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic abnormal frames
# ---------------------------------------------------------------------------

def perturb_gaussian(
    positions: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent Gaussian displacement of every atom, standard deviation σ (Å).

    The easy case: it distorts bonds, angles and torsions at once, so any
    geometric detector should catch it. Useful as a floor, not as evidence.
    """
    pos = np.asarray(positions, dtype=np.float64)
    return pos + rng.normal(0.0, sigma, pos.shape)


def perturb_bond_stretch(
    positions: np.ndarray,
    topo,
    delta: float,
    bond_index: int = 0,
) -> np.ndarray:
    """Lengthen one bond by ``delta`` Å, moving the whole fragment beyond it.

    Moving the fragment rather than the single atom keeps the rest of the geometry
    intact, so the signal comes from the stretched bond instead of from a shower
    of secondary distortions.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    i, j = topo.bonds[bond_index]
    direction = pos[j] - pos[i]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    moving = _fragment_beyond_bond(topo, i, j)
    pos[list(moving)] += direction * delta
    return pos


def perturb_dihedral(
    positions: np.ndarray,
    topo,
    angle_deg: float,
    dihedral: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Rotate about a bond by ``angle_deg``, leaving bonds and angles untouched.

    The interesting case: a pure conformational change. Bond lengths and angles
    are unchanged by construction, so the hard-chemistry checks stay silent and
    only the torsional part of the descriptor can see it. A detector that fails
    here fails on exactly the anomalies that matter in a flexible molecule.

    ``dihedral`` selects the (i, j, k, l) quadruple; the default picks the
    rotatable bond whose two sides are most evenly balanced, which is the bond a
    conformational change is most likely to involve.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    quad = dihedral if dihedral is not None else _most_balanced_dihedral(topo)
    _, j, k, _ = quad

    axis = pos[k] - pos[j]
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError(f"Degenerate rotation axis for dihedral {quad}")
    axis = axis / norm

    moving = _fragment_beyond_bond(topo, j, k)
    theta = np.deg2rad(angle_deg)
    # Rodrigues rotation about `axis` through the atom at j
    rel = pos[list(moving)] - pos[j]
    rotated = (
        rel * np.cos(theta)
        + np.cross(axis, rel) * np.sin(theta)
        + np.outer(rel @ axis, axis) * (1.0 - np.cos(theta))
    )
    pos[list(moving)] = rotated + pos[j]
    return pos


def dihedral_angles_deg(
    positions: np.ndarray,
    dihedral: tuple[int, int, int, int],
) -> np.ndarray:
    """Torsion angle in degrees, −180 to 180, for every frame."""
    from detectana.descriptors import _dihedral_rad

    i, j, k, l = dihedral
    pos = np.asarray(positions, dtype=np.float64)
    return np.degrees(
        np.array([_dihedral_rad(p[i], p[j], p[k], p[l]) for p in pos])
    )


def set_dihedral(
    positions: np.ndarray,
    topo,
    dihedral: tuple[int, int, int, int],
    target_deg: float,
) -> np.ndarray:
    """Rotate each frame so the given torsion equals ``target_deg``.

    Absolute rather than relative, which is what makes a controlled experiment
    possible: every distorted frame ends up at the same known torsion, so the
    result can be compared against how often the training set visits that angle.
    """
    pos = np.asarray(positions, dtype=np.float64)
    current = dihedral_angles_deg(pos, dihedral)
    return np.stack([
        perturb_dihedral(frame, topo, float(target_deg - angle), dihedral=dihedral)
        for frame, angle in zip(pos, current)
    ])


def torsion_coverage(
    train_positions: np.ndarray,
    topo,
    dihedral: tuple[int, int, int, int],
    n_bins: int = 12,
) -> pd.DataFrame:
    """How often the training set visits each slice of a torsion's full circle.

    This is what turns a distortion into a *labelled* anomaly. Rotating a torsion
    the training set already explores produces a perfectly in-distribution frame,
    and a detector that flags it would be wrong. Rotating into a slice with zero
    training frames produces a conformer the model has genuinely never seen. The
    two cases measure opposite things — specificity and sensitivity — and must not
    be pooled.
    """
    angles = dihedral_angles_deg(train_positions, dihedral)
    counts, edges = np.histogram(angles, bins=n_bins, range=(-180.0, 180.0))
    return pd.DataFrame({
        "bin": np.arange(n_bins),
        "angle_low": edges[:-1],
        "angle_high": edges[1:],
        "angle_centre": (edges[:-1] + edges[1:]) / 2.0,
        "train_count": counts,
        "visited": counts > 0,
    })


def most_gap_rich_dihedral(
    train_positions: np.ndarray,
    topo,
    n_bins: int = 12,
    min_fragment: int = 3,
) -> tuple[tuple[int, int, int, int], pd.DataFrame]:
    """Rotatable torsion with the most unvisited slices, and its coverage table.

    Preferred target for a sensitivity test: a torsion with large unexplored
    regions offers somewhere to rotate *to* that is genuinely novel. Terminal
    bonds (fragment smaller than ``min_fragment``) and ring bonds are skipped,
    since rotating those changes almost nothing or nothing at all.
    """
    ring = set(topo.ring_atoms)
    best: tuple[int, int, int, int] | None = None
    best_table: pd.DataFrame | None = None
    best_gaps = -1
    for quad in topo.dihedrals:
        _, j, k, _ = quad
        if j in ring and k in ring:
            continue
        moving = _fragment_beyond_bond(topo, j, k)
        if min(len(moving), topo.n_atoms - len(moving)) < min_fragment:
            continue
        table = torsion_coverage(train_positions, topo, quad, n_bins)
        gaps = int((~table["visited"]).sum())
        if gaps > best_gaps:
            best, best_table, best_gaps = quad, table, gaps
    if best is None or best_table is None:
        raise ValueError("No rotatable non-ring, non-terminal dihedral found")
    return best, best_table


def _adjacency(topo) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(topo.n_atoms)}
    for i, j in topo.bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _fragment_beyond_bond(topo, i: int, j: int) -> set[int]:
    """Atoms reachable from j without crossing the i–j bond (j included).

    In a ring both sides are connected, so this returns almost the whole molecule
    and the "perturbation" becomes a rigid translation of nearly everything. The
    callers avoid ring bonds for that reason.
    """
    adj = _adjacency(topo)
    seen = {i, j}
    stack = [j]
    while stack:
        current = stack.pop()
        for nbr in adj[current]:
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen - {i}


def _most_balanced_dihedral(topo) -> tuple[int, int, int, int]:
    """Pick the dihedral whose central bond splits the molecule most evenly.

    Rotating about a terminal bond moves one hydrogen and barely changes the
    geometry; rotating about a central one is a genuine conformational move. Ring
    bonds are skipped because neither side can move independently.
    """
    if not topo.dihedrals:
        raise ValueError("Topology has no dihedrals to rotate")
    ring = set(topo.ring_atoms)
    best, best_balance = None, -1
    for quad in topo.dihedrals:
        _, j, k, _ = quad
        if j in ring and k in ring:
            continue
        moving = _fragment_beyond_bond(topo, j, k)
        # Balance is worst at 0 (one side empty) and best at n_atoms/2
        balance = min(len(moving), topo.n_atoms - len(moving))
        if balance > best_balance:
            best, best_balance = quad, balance
    if best is None:
        raise ValueError("No rotatable non-ring dihedral found")
    return best
