#!/usr/bin/env python3
"""Does a high OOD score mean the force field is actually wrong there?

This is the question that decides whether the detector is useful. A score says a
frame is unlike the training data; it does not, by itself, say the model's
prediction there is bad. If score and error track each other, the score is a
usable reliability estimate and flagged frames are worth recomputing. If they do
not, the detector is measuring novelty that the model handles fine — which is a
real and publishable result, just a different one.

What you need
-------------
The same frames twice: once with reference forces (the method you trust — DFT,
CCSD, whatever the training data came from) and once with the force field's
predicted forces.

    python scripts/score_vs_error.py \\
        --config config/local.yaml \\
        --reference-xyz frames_with_reference_forces.xyz \\
        --predicted predicted_forces.xyz

``--predicted`` accepts an extxyz carrying forces, or an .npz with a ``forces``
array of shape (n_frames, n_atoms, 3). Frames must be in the same order in both.

Choosing the frames (``--mode sample``)
---------------------------------------
Reference forces are expensive, so the frames you pay for have to span the whole
score range. Sampling only the flagged frames truncates the range and drives the
correlation towards zero no matter what the truth is. This mode scores a whole
trajectory, splits the scores into equal-population bins, and draws the same
number of frames from each bin:

    python scripts/score_vs_error.py --mode sample \\
        --config config/local.yaml \\
        --trajectory aspirin.xc.xyz \\
        --predicted-trajectory aspirin.fc.xyz \\
        --predicted-trajectory-unit hartree_bohr \\
        --n-per-bin 30 --n-bins 10

It writes ``sample_geometries.extxyz`` (the geometries to run the reference
method on, in order), ``sample_predicted.npz`` (the force field's forces for the
same frames, already converted to eV/Å) and ``sample_index.csv`` (frame, step,
score, bin). Run the reference method on the geometries **keeping that order**,
then feed the result back through ``--mode analyse``.

``--scan-stride`` subsamples the trajectory while building the score
distribution. Consecutive PIMD frames are strongly correlated, so a stride costs
almost no information and saves most of the scan time.

Units: both force sets must be in the same units, conventionally eV/Å. i-PI writes
Hartree/Bohr — pass ``--predicted-force-unit hartree_bohr`` and it will be
converted (1 Hartree/Bohr = 51.4221 eV/Å).

Output
------
- ``score_vs_error.csv``   per-frame score, error, and chemistry flags
- ``error_by_decile.csv``  mean and p95 error per score decile — the table to show
- ``score_vs_error.json``  Spearman and Pearson correlations, decile error ratio
- ``score_vs_error.png``   scatter plus the decile means

The frames scored here must not be part of the training or calibration set. The
script does not fit anything: it loads the reference sets from the config, fits
exactly as the pipeline does, then scores the frames you supply.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ase.io import iread

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.evaluation import (
    error_by_score_decile,
    per_frame_force_error,
    score_error_correlation,
)
from detectana.io import MoleculeSpec, load_reference_frames, validate_frame
from detectana.scorer import MahalanobisScorer
from detectana.topology import build_topology, check_chemistry_batch
from detectana.xyz_reader import load_or_build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("detectana.score_vs_error")

HARTREE_BOHR_TO_EV_A = 51.42208619083232

_RE_STEP = re.compile(rb"Step:\s*(\d+)")


def _load_forces_and_positions(
    path: Path,
    spec: MoleculeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and forces from an extxyz file."""
    positions, forces = [], []
    for i, atoms in enumerate(iread(str(path), format="extxyz")):
        validate_frame(atoms, i, source=path.name, spec=spec)
        positions.append(atoms.get_positions())
        try:
            forces.append(atoms.get_forces())
        except Exception:
            info_forces = atoms.info.get("REF_forces", atoms.arrays.get("REF_forces"))
            if info_forces is None:
                raise ValueError(
                    f"{path.name}: frame {i} carries no forces. The reference file "
                    "must have forces; use --predicted for the model's forces."
                ) from None
            forces.append(np.asarray(info_forces, dtype=np.float64))
    if not positions:
        raise ValueError(f"No frames found in {path}")
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(forces, dtype=np.float64),
    )


def _load_predicted(path: Path, spec: MoleculeSpec) -> np.ndarray:
    """Read predicted forces from an .npz or an extxyz."""
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "forces" not in data:
                raise ValueError(
                    f"{path.name}: expected a 'forces' array, found {list(data.keys())}"
                )
            return data["forces"].astype(np.float64)
    _, forces = _load_forces_and_positions(path, spec)
    return forces


# ─────────────────────────────────────────────────────────────────────────────
# Fitting and trajectory reading
# ─────────────────────────────────────────────────────────────────────────────

def _fit_reference(cfg: dict, initial_xyz: str | None = None):
    """Build topology and fit the descriptor pipeline and scorer, as the pipeline does.

    ``initial_xyz`` overrides the config's run entry, which lets a trajectory file
    stand in as its own topology source when the run has no separate initial.xyz.
    """
    chem_cfg = cfg["chemistry"]
    topo = build_topology(
        initial_xyz or cfg["runs"][0]["initial_xyz"],
        nl_mult=chem_cfg["nl_mult"],
        ring_atoms=chem_cfg.get("ring_atoms"),
    )
    spec = MoleculeSpec(topo.n_atoms, tuple(topo.atom_types))
    train_pos, _, _ = load_reference_frames(cfg["reference"]["train"], spec=spec)
    pipe = DescriptorPipeline(
        pca_variance=cfg["descriptor"]["pca_variance"],
        random_seed=cfg["descriptor"]["random_seed"],
    ).fit(compute_descriptor_batch(train_pos, topo))
    scorer = MahalanobisScorer().fit(
        pipe.transform(compute_descriptor_batch(train_pos, topo))
    )
    log.info(
        "Fitted on %d training frames, %d PCA components",
        len(train_pos), pipe.n_components,
    )
    return topo, spec, pipe, scorer, len(train_pos)


def _read_frames(
    path: Path,
    frame_ids: np.ndarray,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the given frame indices from an iPI xyz file by byte offset.

    Works for any iPI xyz payload — positions, or forces, which share the layout.
    Returns ``(steps, values)`` with values shaped (len(frame_ids), n_atoms, 3).
    """
    idx = load_or_build_index(path, cache_path)
    counts = np.unique(idx.atom_count)
    if len(counts) != 1:
        raise ValueError(f"{path.name}: mixed atom counts across frames: {counts.tolist()}")
    n_atoms = int(counts[0])

    steps = np.empty(len(frame_ids), dtype=np.int64)
    values = np.empty((len(frame_ids), n_atoms, 3), dtype=np.float64)
    with open(path, "rb") as fh:
        for out_i, fi in enumerate(frame_ids):
            fh.seek(int(idx.byte_offset[int(fi)]))
            fh.readline()                                   # atom-count line
            comment = fh.readline()
            m = _RE_STEP.search(comment)
            steps[out_i] = int(m.group(1)) if m else int(fi)
            for row in range(n_atoms):
                parts = fh.readline().split()
                values[out_i, row] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return steps, values


def _write_extxyz(
    path: Path,
    symbols: list[str],
    positions: np.ndarray,
    info_rows: list[dict],
) -> None:
    """Write geometries as extxyz, carrying provenance in the comment line."""
    with open(path, "w") as fh:
        for frame_i, pos in enumerate(positions):
            info = " ".join(f"{k}={v}" for k, v in info_rows[frame_i].items())
            fh.write(f"{len(symbols)}\n")
            fh.write(f"Properties=species:S:1:pos:R:3 pbc=\"F F F\" {info}\n")
            for sym, (x, y, z) in zip(symbols, pos):
                fh.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


def _bin_edges(
    scores: np.ndarray,
    n_bins: int,
    scheme: str,
    tail_start_q: float,
) -> np.ndarray:
    """Bin edges for the stratified draw.

    ``quantile`` gives equal-population bins — the honest default, but on a
    trajectory whose scores are concentrated it spends nearly the whole budget
    inside a narrow band and never reaches the scores the detector actually
    flags.

    ``tail-enriched`` splits the bins between the bulk and the upper tail, so
    half the budget lands above ``tail_start_q``. That buys score range at the
    cost of a sample that no longer mirrors the trajectory, which makes the
    pooled correlation biased — report the per-bin table alongside it.
    """
    if scheme == "quantile":
        return np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    n_tail = max(1, n_bins // 2)
    n_bulk = n_bins - n_tail
    bulk = np.quantile(scores, np.linspace(0.0, tail_start_q, n_bulk + 1))
    tail = np.quantile(scores, np.linspace(tail_start_q, 1.0, n_tail + 1))
    return np.concatenate([bulk[:-1], tail])


# ─────────────────────────────────────────────────────────────────────────────
# Mode: sample
# ─────────────────────────────────────────────────────────────────────────────

def run_sample(args: argparse.Namespace, cfg: dict, out_dir: Path) -> None:
    """Score a whole trajectory and draw equal counts from each score bin."""
    traj = Path(args.trajectory)
    cache_dir = Path(args.index_cache_dir) if args.index_cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_for(p: Path) -> Path | None:
        return cache_dir / (p.name + ".frameindex.npz") if cache_dir else None

    # The trajectory's own first frame defines the molecule when the run has no
    # separate initial.xyz — the centroid file is a valid topology source.
    topo, spec, pipe, scorer, n_train = _fit_reference(
        cfg, initial_xyz=args.initial_xyz or str(traj)
    )

    if args.reuse_scan:
        cached = np.load(args.reuse_scan)
        scan_ids, steps, scores = cached["frame"], cached["step"], cached["ood_score"]
        log.info("Reusing %d cached scores from %s", len(scores), args.reuse_scan)
    else:
        idx = load_or_build_index(traj, _cache_for(traj))
        n_frames = idx.n_frames
        scan_ids = np.arange(0, n_frames, args.scan_stride, dtype=np.int64)
        log.info(
            "Scanning %d of %d frames from %s (stride %d)",
            len(scan_ids), n_frames, traj.name, args.scan_stride,
        )

        # ── Score the scanned frames in chunks ───────────────────────────────
        chunk = int(cfg["io"].get("chunk_size", 5000))
        scores = np.empty(len(scan_ids), dtype=np.float64)
        steps = np.empty(len(scan_ids), dtype=np.int64)
        for start in range(0, len(scan_ids), chunk):
            block = scan_ids[start:start + chunk]
            blk_steps, blk_pos = _read_frames(traj, block, _cache_for(traj))
            scores[start:start + len(block)] = scorer.score(
                pipe.transform(compute_descriptor_batch(blk_pos, topo))
            )
            steps[start:start + len(block)] = blk_steps
            log.info("  scored %d / %d", start + len(block), len(scan_ids))

        np.savez(
            out_dir / "scan_scores.npz",
            frame=scan_ids, step=steps, ood_score=scores,
            scan_stride=args.scan_stride,
        )

    # ── Stratified bins, equal draw per bin ──────────────────────────────────
    edges = _bin_edges(scores, args.n_bins, args.bin_scheme, args.tail_start_quantile)
    edges[-1] = np.nextafter(edges[-1], np.inf)     # make the top edge inclusive
    bin_of = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, args.n_bins - 1)

    rng = np.random.default_rng(args.seed)
    picked: list[int] = []
    for b in range(args.n_bins):
        members = np.flatnonzero(bin_of == b)
        if len(members) == 0:
            log.warning("Bin %d is empty — no frames drawn from it", b)
            continue
        take = min(args.n_per_bin, len(members))
        if take < args.n_per_bin:
            log.warning(
                "Bin %d holds only %d frames; drawing %d instead of %d",
                b, len(members), take, args.n_per_bin,
            )
        picked.extend(rng.choice(members, size=take, replace=False).tolist())

    order = np.array(sorted(picked), dtype=np.int64)     # positions within scan_ids
    sel_frames = scan_ids[order]
    sel_scores = scores[order]
    sel_steps = steps[order]
    sel_bins = bin_of[order]
    log.info("Selected %d frames across %d bins", len(sel_frames), args.n_bins)

    # ── Write the geometries to run the reference method on ──────────────────
    _, sel_pos = _read_frames(traj, sel_frames, _cache_for(traj))
    info_rows = [
        {
            "frame": int(sel_frames[i]),
            "step": int(sel_steps[i]),
            "ood_score": f"{sel_scores[i]:.6f}",
            "score_bin": int(sel_bins[i]),
        }
        for i in range(len(sel_frames))
    ]
    geom_path = out_dir / "sample_geometries.extxyz"
    _write_extxyz(geom_path, topo.atom_types, sel_pos, info_rows)

    index = pd.DataFrame({
        "sample_index": np.arange(len(sel_frames)),
        "frame": sel_frames,
        "step": sel_steps,
        "ood_score": sel_scores,
        "score_bin": sel_bins,
    })
    index.to_csv(out_dir / "sample_index.csv", index=False)

    # ── The force field's own forces for the same frames, if available ───────
    pred_path = None
    if args.predicted_trajectory:
        pred_path = Path(args.predicted_trajectory)
        pred_steps, pred_forces = _read_frames(
            pred_path, sel_frames, _cache_for(pred_path)
        )
        if not np.array_equal(pred_steps, sel_steps):
            raise SystemExit(
                f"Step mismatch between {traj.name} and {pred_path.name}: the two "
                "files do not hold the same frames in the same order."
            )
        if args.predicted_trajectory_unit == "hartree_bohr":
            pred_forces = pred_forces * HARTREE_BOHR_TO_EV_A
            log.info("Converted predicted forces from Hartree/Bohr to eV/Å")
        np.savez(
            out_dir / "sample_predicted.npz",
            forces=pred_forces, frame=sel_frames, step=sel_steps,
        )

    # ── Report ───────────────────────────────────────────────────────────────
    summary = index.groupby("score_bin").agg(
        n=("frame", "size"),
        score_min=("ood_score", "min"),
        score_max=("ood_score", "max"),
    ).reset_index()
    print(f"\nScanned {len(scan_ids)} frames; score range "
          f"{scores.min():.2f} – {scores.max():.2f}")
    print("\nSampled frames per score bin")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nGeometries for the reference method: {geom_path}")
    if pred_path is not None:
        mags = np.linalg.norm(np.load(out_dir / "sample_predicted.npz")["forces"], axis=-1)
        print(f"Force field forces:                 {out_dir / 'sample_predicted.npz'}")
        print(f"  per-atom |F|: mean {mags.mean():.3f}, max {mags.max():.3f} eV/Å "
              "— check this is physically sensible before spending on the reference.")
    print(
        "\nNext: run the reference method on every frame of "
        f"{geom_path.name}, keeping that order,\nthen re-run this script with "
        "--mode analyse --reference-xyz <result> --predicted "
        f"{(out_dir / 'sample_predicted.npz').name}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode: analyse
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Pipeline YAML config.")
    parser.add_argument(
        "--mode", default="analyse", choices=["analyse", "sample"],
        help="'sample' picks frames spanning the score range; 'analyse' correlates "
             "score against force error once the reference forces exist.",
    )
    parser.add_argument(
        "--reference-xyz",
        help="analyse: extxyz with the frames to analyse and their reference forces.",
    )
    parser.add_argument(
        "--predicted",
        help="analyse: force field's forces for the same frames — extxyz or .npz.",
    )
    parser.add_argument(
        "--trajectory",
        help="sample: iPI xyz trajectory to score and draw frames from.",
    )
    parser.add_argument(
        "--predicted-trajectory",
        help="sample: matching iPI force file for --trajectory (e.g. aspirin.fc.xyz). "
             "Optional — omit when the force field will be re-run on the sample.",
    )
    parser.add_argument(
        "--predicted-trajectory-unit", default="hartree_bohr",
        choices=["ev_a", "hartree_bohr"],
        help="sample: units in --predicted-trajectory (iPI writes Hartree/Bohr).",
    )
    parser.add_argument(
        "--initial-xyz",
        help="sample: topology source. Defaults to the trajectory's own first frame.",
    )
    parser.add_argument(
        "--scan-stride", type=int, default=10,
        help="sample: score every Nth frame when building the score distribution.",
    )
    parser.add_argument(
        "--n-per-bin", type=int, default=30,
        help="sample: frames drawn from each score bin.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="sample: RNG seed for the draw.",
    )
    parser.add_argument(
        "--bin-scheme", default="quantile", choices=["quantile", "tail-enriched"],
        help="sample: 'quantile' = equal-population bins, faithful to the "
             "trajectory. 'tail-enriched' = half the bins above "
             "--tail-start-quantile, buying score range at the cost of a biased "
             "pooled correlation.",
    )
    parser.add_argument(
        "--tail-start-quantile", type=float, default=0.99,
        help="sample: where the tail begins for --bin-scheme tail-enriched.",
    )
    parser.add_argument(
        "--reuse-scan",
        help="sample: reuse a scan_scores.npz instead of rescoring the trajectory.",
    )
    parser.add_argument(
        "--index-cache-dir",
        help="sample: where to keep byte-offset indices. Defaults to alongside the "
             "trajectory; point it elsewhere to avoid writing to read-only data.",
    )
    parser.add_argument(
        "--predicted-force-unit", default="ev_a", choices=["ev_a", "hartree_bohr"],
        help="Units of the predicted forces (default: eV/Å).",
    )
    parser.add_argument(
        "--reference-force-unit", default="ev_a", choices=["ev_a", "hartree_bohr"],
        help="Units of the reference forces (default: eV/Å).",
    )
    parser.add_argument(
        "--metric", default="mae", choices=["mae", "rmse", "max"],
        help="Per-frame force error: mae, rmse, or max per-atom norm (default: mae).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    out_dir = Path(args.output_dir or Path(cfg["io"]["output_dir"]) / "score_vs_error")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        if not args.trajectory:
            raise SystemExit("--mode sample needs --trajectory")
        run_sample(args, cfg, out_dir)
        return

    if not (args.reference_xyz and args.predicted):
        raise SystemExit("--mode analyse needs --reference-xyz and --predicted")

    # ── Fit exactly as the pipeline does ─────────────────────────────────────
    chem_cfg = cfg["chemistry"]
    topo, spec, pipe, scorer, n_train = _fit_reference(cfg, args.initial_xyz)

    # ── Load the frames to analyse ───────────────────────────────────────────
    positions, ref_forces = _load_forces_and_positions(Path(args.reference_xyz), spec)
    pred_forces = _load_predicted(Path(args.predicted), spec)
    if pred_forces.shape != ref_forces.shape:
        raise SystemExit(
            f"Shape mismatch: reference forces {ref_forces.shape} vs predicted "
            f"{pred_forces.shape}. The two files must hold the same frames in the "
            "same order."
        )
    if args.reference_force_unit == "hartree_bohr":
        ref_forces = ref_forces * HARTREE_BOHR_TO_EV_A
        log.info("Converted reference forces from Hartree/Bohr to eV/Å")
    if args.predicted_force_unit == "hartree_bohr":
        pred_forces = pred_forces * HARTREE_BOHR_TO_EV_A
        log.info("Converted predicted forces from Hartree/Bohr to eV/Å")

    log.info("Analysing %d frames", len(positions))

    # ── Score, error, correlate ──────────────────────────────────────────────
    scores = scorer.score(pipe.transform(compute_descriptor_batch(positions, topo)))
    errors = per_frame_force_error(ref_forces, pred_forces, metric=args.metric)
    flags = check_chemistry_batch(
        positions, topo, chem_cfg["bond_break_cutoff"], chem_cfg["close_contact_cutoff"]
    )

    per_frame = pd.DataFrame({
        "frame": np.arange(len(scores)),
        "ood_score": scores,
        f"force_{args.metric}_eV_per_A": errors,
        "broken_bond": [f.has_broken_bond for f in flags],
        "close_contact": [f.has_close_contact for f in flags],
    })
    per_frame.to_csv(out_dir / "score_vs_error.csv", index=False)

    correlation = score_error_correlation(scores, errors)
    deciles = error_by_score_decile(scores, errors, n_bins=args.n_bins)
    deciles.to_csv(out_dir / "error_by_decile.csv", index=False)

    ratio = (
        float(deciles["error_mean"].iloc[-1] / deciles["error_mean"].iloc[0])
        if len(deciles) >= 2 and deciles["error_mean"].iloc[0] > 0
        else None
    )
    summary = {
        **correlation,
        "metric": args.metric,
        "n_frames": int(len(scores)),
        "top_over_bottom_decile_error_ratio": ratio,
        "n_train": int(n_train),
        "pca_n_components": int(pipe.n_components),
    }
    with open(out_dir / "score_vs_error.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    try:
        _plot(scores, errors, deciles, args.metric, out_dir)
    except Exception as exc:
        log.warning("Plotting failed (non-fatal): %s", exc)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\nForce error ({args.metric}, eV/Å) by OOD-score decile")
    print(deciles.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"\nSpearman(score, error) = {correlation['spearman']:+.3f} "
        f"(p = {correlation['spearman_p']:.2g}), n = {correlation['n']}"
    )
    if ratio is not None:
        print(f"Top decile has {ratio:.1f}x the mean error of the bottom decile.")
    print(
        "\nInterpretation: a strongly positive Spearman and a ratio well above 1 "
        "mean the\nscore is a usable reliability estimate. A flat table means it "
        "detects novelty the\nforce field handles correctly — worth reporting as "
        "such, not worth hiding."
    )


def _plot(
    scores: np.ndarray,
    errors: np.ndarray,
    deciles: pd.DataFrame,
    metric: str,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(scores, errors, s=8, alpha=0.4, color="steelblue")
    axes[0].set_xlabel("OOD score (Mahalanobis distance)")
    axes[0].set_ylabel(f"Force {metric} (eV/Å)")
    axes[0].set_title("Per frame")
    axes[0].grid(alpha=0.3)

    centres = (deciles["score_min"] + deciles["score_max"]) / 2
    axes[1].plot(centres, deciles["error_mean"], "o-", color="darkorange", label="mean")
    axes[1].plot(centres, deciles["error_p95"], "s--", color="firebrick", label="p95")
    axes[1].set_xlabel("OOD score (decile centre)")
    axes[1].set_ylabel(f"Force {metric} (eV/Å)")
    axes[1].set_title("By score decile")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "score_vs_error.png", dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_dir / "score_vs_error.png")


if __name__ == "__main__":
    main()
