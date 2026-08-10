#!/usr/bin/env python3
"""Benchmark the OOD detector against controlled distortions.

The pipeline reports which frames are out of distribution but never how good it is
at that. This script answers it with data already in the repository: held-out
reference frames are the negatives, and copies of those frames distorted by a
known amount are the positives.

Two magnitude ladders and one torsion scan:

- **rattle** — Gaussian displacement of every atom (σ in Å). Distorts bonds,
  angles and torsions at once. Any geometric detector should win here, so it is a
  floor rather than evidence.
- **stretch** — one bond lengthened by δ Å, the fragment beyond it moved with it.
  The case the hard-chemistry bond check also catches.
- **torsion scan** — the interesting one. A rotatable torsion is driven right
  around its circle and every frame is scored, with each target angle labelled by
  how often the *training set* visits it. Bond lengths and angles are untouched, so
  this isolates conformational novelty.

The torsion scan is what makes the benchmark meaningful, because "distorted" and
"out of distribution" are not the same thing. Rotating a torsion the training set
already explores yields a perfectly normal frame, and flagging it would be an
error; rotating into a slice with zero training frames yields a conformer the model
has never seen, and missing it would be an error. Pooling the two would report a
mediocre average for a detector that is in fact doing the right thing in both
cases, so they are reported separately: detection rate in visited slices is a
false-alarm measurement, in unvisited slices a sensitivity measurement.

Reported per magnitude: AUROC, average precision, and the detection rate at the
configured false-flag rate, where the threshold is the conformal order statistic
of the calibration scores — the same rule the pipeline deploys. So the detection
rate answers "what fraction of these would a real run have flagged?".

The fit and the threshold see only the training and calibration frames. Evaluation
frames and their distorted copies are held out from both.

Usage
-----
    python scripts/benchmark_detector.py --config config/demo.yaml
    python scripts/benchmark_detector.py --config config/local.yaml \\
        --output-dir outputs/benchmark --false-flag-rate 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.evaluation import (
    conformal_threshold,
    detection_metrics,
    most_gap_rich_dihedral,
    perturb_bond_stretch,
    perturb_gaussian,
    set_dihedral,
)
from detectana.io import MoleculeSpec, load_reference_frames
from detectana.scorer import MahalanobisScorer
from detectana.topology import build_topology, check_chemistry_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("detectana.benchmark")

# Magnitude ladders. Small enough at the bottom to be indistinguishable from
# thermal motion, large enough at the top that any detector must catch it.
RATTLE_SIGMAS_A = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
STRETCH_DELTAS_A = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
TORSION_BINS = 12   # 30° slices around the circle


def _build_positives(
    positions: np.ndarray,
    topo,
    kind: str,
    magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one distortion at one magnitude to every evaluation frame."""
    if kind == "rattle":
        return perturb_gaussian(positions, magnitude, rng)
    if kind == "stretch":
        return np.stack([
            perturb_bond_stretch(frame, topo, magnitude) for frame in positions
        ])
    raise ValueError(f"Unknown distortion {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Pipeline YAML config.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write results (default: <io.output_dir>/benchmark).",
    )
    parser.add_argument(
        "--false-flag-rate", type=float, default=None,
        help="α for the detection rate (default: 1 − threshold.percentile/100).",
    )
    parser.add_argument(
        "--calibration-split", type=float, default=0.5,
        help=(
            "Fraction of the held-out set used to calibrate the threshold; the "
            "rest becomes the evaluation negatives (default: 0.5)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(args.output_dir or Path(cfg["io"]["output_dir"]) / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = args.false_flag_rate
    if alpha is None:
        alpha = 1.0 - cfg["threshold"]["percentile"] / 100.0
    rng = np.random.default_rng(args.seed)

    # ── Fit exactly as the pipeline does ─────────────────────────────────────
    chem_cfg = cfg["chemistry"]
    topo = build_topology(
        cfg["runs"][0]["initial_xyz"],
        nl_mult=chem_cfg["nl_mult"],
        ring_atoms=chem_cfg.get("ring_atoms"),
    )
    spec = MoleculeSpec(topo.n_atoms, tuple(topo.atom_types))
    log.info("Molecule: %d atoms, %d features", topo.n_atoms, topo.n_features)

    ref_cfg = cfg["reference"]
    train_pos, _, _ = load_reference_frames(ref_cfg["train"], spec=spec)
    held_out_pos, _, _ = load_reference_frames(ref_cfg["valid"], spec=spec)

    pipe = DescriptorPipeline(
        pca_variance=cfg["descriptor"]["pca_variance"],
        random_seed=cfg["descriptor"]["random_seed"],
    ).fit(compute_descriptor_batch(train_pos, topo))
    scorer = MahalanobisScorer().fit(pipe.transform(compute_descriptor_batch(train_pos, topo)))

    def score(positions: np.ndarray) -> np.ndarray:
        return scorer.score(pipe.transform(compute_descriptor_batch(positions, topo)))

    # The Mahalanobis covariance is estimated in PCA space, so it needs many more
    # training frames than components. Below ~10 per component the inverse is
    # dominated by estimation noise and every number downstream is unreliable —
    # which is exactly the regime config/demo.yaml runs in.
    if len(train_pos) < 10 * pipe.n_components:
        log.warning(
            "Only %d training frames for %d PCA components (<10 per component). "
            "The covariance is poorly conditioned, so treat these results as a "
            "smoke test of the machinery, not as a measurement of the detector.",
            len(train_pos), pipe.n_components,
        )

    # ── Split the held-out set: calibrate on one half, evaluate on the other ──
    # Strided rather than sliced, because reference frames are consecutive in time
    # and a contiguous split would compare two different stretches of trajectory.
    n_held = len(held_out_pos)
    if n_held < 8:
        raise SystemExit(f"Need at least 8 held-out frames to split, got {n_held}")
    stride = max(2, int(round(1.0 / max(args.calibration_split, 1e-6))))
    calib_idx = np.arange(0, n_held, stride)
    eval_idx = np.setdiff1d(np.arange(n_held), calib_idx)
    calib_scores = score(held_out_pos[calib_idx])
    eval_pos = held_out_pos[eval_idx]
    eval_scores = score(eval_pos)
    log.info(
        "Held-out %d frames → %d calibration, %d evaluation negatives",
        n_held, len(calib_idx), len(eval_idx),
    )
    log.info(
        "Threshold at α=%.3g: %.3f (flags %.1f%% of the evaluation negatives)",
        alpha, conformal_threshold(calib_scores, alpha),
        100.0 * (eval_scores > conformal_threshold(calib_scores, alpha)).mean(),
    )

    def chemistry_flag_rate(positions: np.ndarray) -> float:
        """Would the hard-chemistry checks catch this without the OOD score?"""
        flags = check_chemistry_batch(
            positions, topo,
            chem_cfg["bond_break_cutoff"], chem_cfg["close_contact_cutoff"],
        )
        return float(np.mean([f.has_broken_bond or f.has_close_contact for f in flags]))

    # ── Distortion ladders ───────────────────────────────────────────────────
    rows: list[dict] = []
    for kind, magnitudes, unit in (
        ("rattle", RATTLE_SIGMAS_A, "sigma_A"),
        ("stretch", STRETCH_DELTAS_A, "delta_A"),
    ):
        for magnitude in magnitudes:
            positives = _build_positives(eval_pos, topo, kind, magnitude, rng)
            metrics = detection_metrics(calib_scores, score(positives), false_flag_rate=alpha)
            chem_rate = chemistry_flag_rate(positives)
            rows.append({
                "distortion": kind,
                "magnitude": magnitude,
                "magnitude_unit": unit,
                "train_count_at_target": None,
                "target_visited_in_training": None,
                "auroc": metrics["auroc"],
                "average_precision": metrics["average_precision"],
                "detection_rate": metrics["detection_rate"],
                "median_score_normal": metrics["median_score_in_distribution"],
                "median_score_distorted": metrics["median_score_abnormal"],
                "chemistry_flag_rate": chem_rate,
                "n_negatives": metrics["n_in_distribution"],
                "n_positives": metrics["n_abnormal"],
            })
            log.info(
                "%-8s %-7s AUROC=%.3f  detected=%5.1f%%  (chemistry flags %5.1f%%)",
                kind, f"{magnitude:g}", metrics["auroc"],
                100 * metrics["detection_rate"], 100 * chem_rate,
            )

    # ── Torsion scan, labelled by training coverage ──────────────────────────
    quad, coverage = most_gap_rich_dihedral(train_pos, topo, n_bins=TORSION_BINS)
    label = "-".join(f"{topo.atom_types[i]}{i}" for i in quad)
    log.info(
        "Torsion scan on %s: %d of %d slices unvisited in training",
        label, int((~coverage["visited"]).sum()), TORSION_BINS,
    )
    coverage.to_csv(out_dir / "torsion_coverage.csv", index=False)

    for _, slice_row in coverage.iterrows():
        target = float(slice_row["angle_centre"])
        positives = set_dihedral(eval_pos, topo, quad, target)
        metrics = detection_metrics(calib_scores, score(positives), false_flag_rate=alpha)
        rows.append({
            "distortion": "torsion_scan",
            "magnitude": target,
            "magnitude_unit": "target_angle_deg",
            "train_count_at_target": int(slice_row["train_count"]),
            "target_visited_in_training": bool(slice_row["visited"]),
            "auroc": metrics["auroc"],
            "average_precision": metrics["average_precision"],
            "detection_rate": metrics["detection_rate"],
            "median_score_normal": metrics["median_score_in_distribution"],
            "median_score_distorted": metrics["median_score_abnormal"],
            "chemistry_flag_rate": chemistry_flag_rate(positives),
            "n_negatives": metrics["n_in_distribution"],
            "n_positives": metrics["n_abnormal"],
        })
        log.info(
            "torsion  %+7.0f° train_n=%5d %-11s detected=%5.1f%%  AUROC=%.3f",
            target, int(slice_row["train_count"]),
            "(visited)" if slice_row["visited"] else "(UNVISITED)",
            100 * metrics["detection_rate"], metrics["auroc"],
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "detection_benchmark.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    # Summarising the scan needs care. Averaging over every visited slice mixes
    # well-sampled regions with ones holding a handful of frames, and the sparse
    # ones are legitimately flagged — 5 frames in 2500 is not coverage. So report
    # the two ends separately, plus the rank correlation, which is the real claim:
    # the flag rate falls monotonically as training density rises.
    scan = df[df["distortion"] == "torsion_scan"]
    unvisited = scan[~scan["target_visited_in_training"].astype(bool)]
    dense_cut = max(1, int(0.01 * len(train_pos)))   # ≥1% of training frames
    dense = scan[scan["train_count_at_target"] >= dense_cut]
    rho = None
    if len(scan) >= 3:
        from scipy.stats import spearmanr
        rho = float(spearmanr(scan["train_count_at_target"], scan["detection_rate"]).statistic)

    headline = {
        "torsion": [int(i) for i in quad],
        "torsion_label": label,
        "n_slices_unvisited": int(len(unvisited)),
        "n_slices_dense": int(len(dense)),
        "dense_slice_min_train_count": dense_cut,
        "detection_rate_unvisited_slices": (
            float(unvisited["detection_rate"].mean()) if len(unvisited) else None
        ),
        "detection_rate_dense_slices": (
            float(dense["detection_rate"].mean()) if len(dense) else None
        ),
        "spearman_train_count_vs_detection_rate": rho,
    }

    summary = {
        "config": args.config,
        "false_flag_rate": alpha,
        "threshold": conformal_threshold(calib_scores, alpha),
        "n_train": int(len(train_pos)),
        "n_calibration": int(len(calib_idx)),
        "n_evaluation_negatives": int(len(eval_idx)),
        "n_features": int(topo.n_features),
        "pca_n_components": int(pipe.n_components),
        "seed": args.seed,
        "torsion_scan": headline,
    }
    with open(out_dir / "detection_benchmark.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        _plot(df, coverage, label, out_dir, alpha)
    except Exception as exc:  # plotting must never fail the benchmark
        log.warning("Plotting failed (non-fatal): %s", exc)

    # ── Readable summary ─────────────────────────────────────────────────────
    ladder = df[df["distortion"] != "torsion_scan"]
    print(f"\nDistortion ladders — detection rate at α = {alpha:.3g}")
    print(
        ladder.pivot_table(index="magnitude", columns="distortion", values="detection_rate")
        .to_string(float_format=lambda v: f"{v:6.1%}", na_rep="     -")
    )
    print(f"\nTorsion scan on {label} — sensitivity to conformational novelty")
    if headline["detection_rate_dense_slices"] is not None:
        print(
            f"  well-sampled slices  (≥{dense_cut} training frames, {headline['n_slices_dense']:2d} of "
            f"{TORSION_BINS}): flagged {headline['detection_rate_dense_slices']:.1%}"
            f"  ← should sit near α = {alpha:.1%}"
        )
    if headline["detection_rate_unvisited_slices"] is not None:
        print(
            f"  never-visited slices (0 training frames,  {headline['n_slices_unvisited']:2d} of "
            f"{TORSION_BINS}): flagged {headline['detection_rate_unvisited_slices']:.1%}"
            "  ← should be high"
        )
    else:
        print("  no unvisited slices: this torsion is fully sampled in training.")
    if rho is not None:
        print(
            f"  Spearman(training frames in slice, flag rate) = {rho:+.2f} "
            "← negative means the flag rate tracks how little training data is there"
        )


def _plot(
    df: pd.DataFrame,
    coverage: pd.DataFrame,
    torsion_label: str,
    out_dir: Path,
    alpha: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"rattle": "steelblue", "stretch": "darkorange"}

    # Left: detection rate against distortion magnitude
    ladder = df[df["distortion"] != "torsion_scan"]
    for kind, sub in ladder.groupby("distortion"):
        sub = sub.sort_values("magnitude")
        axes[0].plot(sub["magnitude"], sub["detection_rate"], "o-",
                     color=colors.get(kind), label=kind)
    axes[0].axhline(alpha, color="red", ls="--", lw=1, label=f"α={alpha:g}")
    axes[0].set_xlabel("Distortion magnitude (Å)")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Bond and Cartesian distortions")

    # Right: torsion scan against training coverage — the informative panel
    scan = df[df["distortion"] == "torsion_scan"].sort_values("magnitude")
    ax = axes[1]
    bars = ax.twinx()
    bars.bar(coverage["angle_centre"], coverage["train_count"], width=26,
             color="lightgrey", label="training frames")
    bars.set_ylabel("Training frames per 30° slice", color="grey")
    bars.tick_params(axis="y", labelcolor="grey")
    bars.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    ax.plot(scan["magnitude"], scan["detection_rate"], "o-",
            color="forestgreen", label="detection rate")
    ax.axhline(alpha, color="red", ls="--", lw=1, label=f"α={alpha:g}")
    ax.set_xlabel(f"Target torsion angle for {torsion_label} (degrees)")
    ax.set_ylabel("Detection rate", color="forestgreen")
    ax.tick_params(axis="y", labelcolor="forestgreen")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-190, 190)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("Conformational novelty: flagged where training is empty")

    fig.tight_layout()
    fig.savefig(out_dir / "detection_benchmark.png", dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_dir / "detection_benchmark.png")


if __name__ == "__main__":
    main()
