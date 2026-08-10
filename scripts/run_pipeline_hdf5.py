#!/usr/bin/env python3
"""Run detectana OOD analysis on PIMD HDF5 trajectories.

Mirrors run_pipeline.py but reads beads and centroid directly from
nvt_trajectory.hdf5 via load_pimd_trajectory_hdf5 — no XYZ conversion needed.

Usage
-----
    python scripts/run_pipeline_hdf5.py --config config/pimd6_s4_hdf5.yaml

Config additions vs default.yaml
---------------------------------
runs:
  - name: s1
    hdf5: /path/to/nvt_trajectory.hdf5
    initial_xyz: /path/to/input.xyz
    timestep_fs: 0.2
    stride: 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from detectana import __version__
from detectana.aggregator import aggregate_bead_scores
from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.io import load_pimd_trajectory_hdf5, load_reference_frames, load_single_frame
from detectana.onset import detect_onset
from detectana.scorer import MahalanobisScorer
from detectana.topology import build_topology, check_chemistry_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Per-bead worker
# ---------------------------------------------------------------------------

def _score_bead(
    bead_idx: int,
    bead_positions: np.ndarray,          # (n_frames, n_atoms, 3)
    topo,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    bead_threshold: float,
    cache_path: Path,
    force_recompute: bool,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute descriptors and OOD scores for one bead. Cache to disk."""
    if cache_path.exists() and not force_recompute:
        log.info("  bead %02d: loading descriptor cache", bead_idx)
        data = np.load(cache_path)
        descs = data["descriptors"]
        steps = data["steps"]
    else:
        log.info("  bead %02d: computing descriptors (%d frames) …", bead_idx, len(bead_positions))
        descs = compute_descriptor_batch(bead_positions, topo)
        steps = np.arange(len(bead_positions), dtype=np.int64)
        np.savez_compressed(cache_path, descriptors=descs, steps=steps)

    X_pca = pipe.transform(descs)
    scores = scorer.score(X_pca)
    log.info(
        "  bead %02d: max=%.3f  frac_ood=%.4f",
        bead_idx, scores.max(), (scores > bead_threshold).mean(),
    )
    return bead_idx, scores, steps


# ---------------------------------------------------------------------------
# Per-run
# ---------------------------------------------------------------------------

def _process_run_hdf5(
    run_cfg: dict,
    topo,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    bead_threshold: float,
    centroid_threshold: float,
    cfg: dict,
    out_root: Path,
) -> dict:
    run_name = run_cfg["name"]
    run_out = out_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    cache_dir = run_out / "descriptor_cache"
    cache_dir.mkdir(exist_ok=True)

    io_cfg = cfg["io"]
    chem_cfg = cfg["chemistry"]
    onset_cfg = cfg["onset"]
    n_jobs = cfg.get("pipeline", {}).get("n_jobs", -1)
    force_recompute = io_cfg.get("force_recompute", False)
    timestep_fs = run_cfg.get("timestep_fs", 0.2)
    stride = run_cfg.get("stride", 50)
    frame_time_fs = timestep_fs * stride

    log.info("=== Run: %s — loading HDF5 …", run_name)
    traj = load_pimd_trajectory_hdf5(run_cfg["hdf5"])
    n_frames, n_beads, _, _ = traj.bead_positions.shape
    log.info("  %d frames, %d beads", n_frames, n_beads)

    # --- Per-bead scoring (parallel) ---
    results = Parallel(n_jobs=n_jobs)(
        delayed(_score_bead)(
            bead_idx=bi,
            bead_positions=traj.bead_positions[:, bi, :, :],
            topo=topo,
            pipe=pipe,
            scorer=scorer,
            bead_threshold=bead_threshold,
            cache_path=cache_dir / f"bead_{bi:02d}_descriptors.npz",
            force_recompute=force_recompute,
        )
        for bi in range(n_beads)
    )
    results.sort(key=lambda r: r[0])
    bead_scores = np.stack([s for _, s, _ in results])   # (n_beads, n_frames)
    steps = results[0][2]
    np.save(run_out / "bead_scores.npy", bead_scores)

    # --- Centroid scoring ---
    centroid_cache = cache_dir / "centroid_descriptors.npz"
    if centroid_cache.exists() and not force_recompute:
        data = np.load(centroid_cache)
        centroid_descs = data["descriptors"]
    else:
        log.info("  centroid: computing descriptors …")
        centroid_descs = compute_descriptor_batch(traj.centroid_positions, topo)
        np.savez_compressed(centroid_cache, descriptors=centroid_descs, steps=steps)
    centroid_scores = scorer.score(pipe.transform(centroid_descs))
    np.save(run_out / "centroid_scores.npy", centroid_scores)

    # --- Hard-chemistry check on bead 0 ---
    log.info("  chemistry check on bead 00 …")
    chem_rows = []
    pos_b0 = traj.bead_positions[:, 0, :, :]
    flags_batch = check_chemistry_batch(
        pos_b0, topo,
        chem_cfg["bond_break_cutoff"],
        chem_cfg["close_contact_cutoff"],
    )
    for step, flags in zip(steps, flags_batch):
        row = {"step": int(step)}
        row.update(flags.to_dict())
        chem_rows.append(row)
    pd.DataFrame(chem_rows).to_csv(run_out / "chemistry_flags_bead00.csv", index=False)

    del traj  # release HDF5 RAM

    # --- Aggregate ---
    agg_df = aggregate_bead_scores(
        bead_scores=bead_scores,
        centroid_scores=centroid_scores,
        steps=steps,
        threshold=bead_threshold,
        frame_time_fs=frame_time_fs,
    )
    # Override centroid_ood with its own threshold
    agg_df["centroid_ood"] = centroid_scores > centroid_threshold
    agg_df.to_csv(run_out / "frame_aggregate.csv", index=False)
    log.info("  frame_aggregate.csv saved (%d rows)", len(agg_df))

    # --- Onset ---
    onset = detect_onset(
        aggregate_df=agg_df,
        threshold=bead_threshold,
        window_frames=onset_cfg["window_frames"],
        step_frames=onset_cfg["step_frames"],
        fraction_threshold=onset_cfg["fraction_threshold"],
    )
    ood_mask = bead_scores > bead_threshold
    if ood_mask.any():
        first_ood = np.where(ood_mask.any(axis=1), np.argmax(ood_mask, axis=1), np.iinfo(np.int64).max)
        onset.first_anomaly_bead_idx = int(np.argmin(first_ood))

    onset_dict = onset.to_dict()
    pd.DataFrame([{"run": run_name, **onset_dict}]).to_csv(run_out / "onset_table.csv", index=False)
    log.info("  onset_table.csv: %s", onset_dict)

    # --- Plots ---
    try:
        _make_plots(agg_df, onset, bead_threshold, centroid_threshold, run_name, run_out)
    except Exception as exc:
        log.warning("  plotting failed (non-fatal): %s", exc)

    return {"run": run_name, **onset_dict}


# ---------------------------------------------------------------------------
# Plots (same as pipeline.py)
# ---------------------------------------------------------------------------

def _make_plots(agg_df, onset, bead_threshold, centroid_threshold, run_name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    time_ps = agg_df["time_ps"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(time_ps, agg_df["bead_max"], lw=0.5, label="bead max", color="steelblue")
    ax.plot(time_ps, agg_df["bead_p95"], lw=0.5, label="bead p95", color="cornflowerblue")
    ax.axhline(bead_threshold, color="red", ls="--", lw=1, label=f"bead threshold={bead_threshold:.2f} (99th pct)")
    if onset.persistent_bead_anomaly_frame is not None and onset.persistent_bead_anomaly_frame < len(time_ps):
        ax.axvline(time_ps[onset.persistent_bead_anomaly_frame], color="orange", ls=":", lw=1.2, label="bead onset")
    ax.set_ylabel("Mahalanobis distance")
    ax.legend(fontsize=7)
    ax.set_title(f"{run_name} — HDF5 trajectory")

    ax = axes[1]
    ax.plot(time_ps, agg_df["bead_frac_ood"], lw=0.5, color="darkorange", label="bead frac OOD (99th pct)")
    ax.set_ylabel("Fraction OOD")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(time_ps, agg_df["centroid_score"], lw=0.5, color="forestgreen", label="centroid score")
    ax.axhline(centroid_threshold, color="purple", ls="--", lw=1, label=f"centroid threshold={centroid_threshold:.2f} (85th pct)")
    if onset.centroid_anomaly_frame is not None and onset.centroid_anomaly_frame < len(time_ps):
        ax.axvline(time_ps[onset.centroid_anomaly_frame], color="purple", ls=":", lw=1.2, label="centroid onset")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_xlabel("Time (ps)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(plots_dir / "score_vs_time.png", dpi=150)
    plt.close(fig)
    log.info("  saved score_vs_time.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline_hdf5(cfg: dict) -> None:
    out_root = Path(cfg["io"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # Reference data
    log.info("Loading reference data …")
    ref_cfg = cfg["reference"]
    train_pos, _, _ = load_reference_frames(ref_cfg["train"])
    valid_pos, _, _ = load_reference_frames(ref_cfg["valid"])

    # Topology from the first run's initial.xyz
    first_run = cfg["runs"][0]
    topo = build_topology(first_run["initial_xyz"], nl_mult=cfg["chemistry"]["nl_mult"])
    log.info(
        "Topology: %d bonds, %d angles, %d dihedrals → %d features",
        len(topo.bonds), len(topo.angles), len(topo.dihedrals), topo.n_features,
    )

    # Fit PCA + scorer
    log.info("Fitting descriptors on training data …")
    X_train = compute_descriptor_batch(train_pos, topo)
    pipe = DescriptorPipeline(
        pca_variance=cfg["descriptor"]["pca_variance"],
        random_seed=cfg["descriptor"]["random_seed"],
    )
    pipe.fit(X_train)
    log.info(
        "PCA: %d components, %.1f%% variance",
        pipe.n_components, pipe.explained_variance_ratio.sum() * 100,
    )

    X_val = compute_descriptor_batch(valid_pos, topo)
    X_val_pca = pipe.transform(X_val)
    scorer = MahalanobisScorer()
    scorer.fit(pipe.transform(X_train))
    thr_cfg = cfg["threshold"]
    bead_threshold = scorer.calibrate(X_val_pca, percentile=thr_cfg["bead_percentile"])
    centroid_threshold = scorer.calibrate(X_val_pca, percentile=thr_cfg["centroid_percentile"])
    log.info(
        "OOD thresholds — bead (%.1f-th pct): %.4f  centroid (%.1f-th pct): %.4f",
        thr_cfg["bead_percentile"], bead_threshold,
        thr_cfg["centroid_percentile"], centroid_threshold,
    )

    models_dir = out_root / "models"
    models_dir.mkdir(exist_ok=True)
    pipe.save(models_dir / "descriptor_pipeline.pkl")
    scorer.save(models_dir / "scorer.pkl")

    # Per-run
    all_onset_rows = []
    for run_cfg in cfg["runs"]:
        row = _process_run_hdf5(
            run_cfg=run_cfg,
            topo=topo,
            pipe=pipe,
            scorer=scorer,
            bead_threshold=bead_threshold,
            centroid_threshold=centroid_threshold,
            cfg=cfg,
            out_root=out_root,
        )
        all_onset_rows.append(row)

    pd.DataFrame(all_onset_rows).to_csv(out_root / "onset_summary.csv", index=False)
    log.info("onset_summary.csv saved to %s", out_root)

    manifest = {
        "detectana_version": __version__,
        "config": cfg,
        "bead_ood_threshold": bead_threshold,
        "centroid_ood_threshold": centroid_threshold,
        "pca_n_components": pipe.n_components,
        "pca_variance_explained": float(pipe.explained_variance_ratio.sum()),
        "n_features": topo.n_features,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, cls=_NumpyEncoder)
    log.info("manifest.json saved")


def _parse_args():
    p = argparse.ArgumentParser(description="DetectAna — HDF5 pipeline")
    p.add_argument(
        "--config",
        default="config/pimd6_s4_hdf5.yaml",
        help="YAML config file",
    )
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.force_recompute:
        cfg.setdefault("io", {})["force_recompute"] = True

    run_pipeline_hdf5(cfg)
