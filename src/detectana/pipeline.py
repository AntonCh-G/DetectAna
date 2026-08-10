"""Pipeline orchestrator: the ten workflow steps, end to end.

Step 1.  Load trajectories and metadata.
Step 2.  Validate units, atom order, indexing.
Step 3.  Compute hard-chemistry checks.
Step 4.  Compute internal-coordinate fingerprints.
Step 5.  Fit OOD statistics on training frames.
Step 6.  Calibrate threshold on validation frames.
Step 7.  Score bead and centroid frames.
Step 8.  Aggregate bead scores per timestep.
Step 9.  Detect persistent anomaly onset.
Step 10. Save tables, plots, and manifest.
"""

from __future__ import annotations

import glob
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from detectana import __version__
from detectana.aggregator import add_embedding_scores, aggregate_bead_scores, bead_score_summary
from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.embedding_scorer import EmbeddingPipeline
from detectana.io import iter_bead_positions, load_embeddings_h5, load_pimd_trajectory_hdf5, load_reference_frames, load_single_frame
from detectana.onset import OnsetResult, detect_onset
from detectana.scorer import MahalanobisScorer
from detectana.topology import AspirinTopology, build_topology, check_chemistry_batch  # noqa: F401

log = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to Python natives."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> None:
    """Execute the full anomaly-onset workflow for all configured runs.

    Parameters
    ----------
    cfg : dict loaded from YAML config (see config/default.yaml).
    """
    out_root = Path(cfg["io"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1–2: Reference data + topology ─────────────────────────────────
    log.info("Loading reference data …")
    ref_cfg = cfg["reference"]
    train_pos, _, _ = load_reference_frames(ref_cfg["train"])
    valid_pos, _, _ = load_reference_frames(ref_cfg["valid"])

    # Topology from the first run's initial.xyz (all runs use same molecule)
    first_run = cfg["runs"][0]
    topo = build_topology(
        first_run["initial_xyz"],
        nl_mult=cfg["chemistry"]["nl_mult"],
    )
    log.info(
        "Topology: %d bonds, %d angles, %d dihedrals, ring=%s → %d features",
        len(topo.bonds), len(topo.angles), len(topo.dihedrals),
        topo.ring_atoms, topo.n_features,
    )

    # ── Step 4–5: Descriptors + PCA fit on training data ─────────────────────
    log.info("Computing training descriptors …")
    X_train = compute_descriptor_batch(train_pos, topo)   # (2500, n_feat)

    desc_cfg = cfg["descriptor"]
    pipe = DescriptorPipeline(
        pca_variance=desc_cfg["pca_variance"],
        random_seed=desc_cfg["random_seed"],
    )
    pipe.fit(X_train)
    log.info("PCA: %d components explain %.1f%% variance",
             pipe.n_components, pipe.explained_variance_ratio.sum() * 100)

    # ── Step 6: Calibrate threshold on validation data ────────────────────────
    log.info("Calibrating OOD threshold on validation set …")
    X_val = compute_descriptor_batch(valid_pos, topo)
    X_val_pca = pipe.transform(X_val)
    X_train_pca = pipe.transform(X_train)

    scorer = MahalanobisScorer()
    scorer.fit(X_train_pca)
    threshold = scorer.calibrate(X_val_pca, percentile=cfg["threshold"]["percentile"])
    log.info("OOD threshold (%.1f-th pct of validation): %.4f", cfg["threshold"]["percentile"], threshold)

    # Save fitted objects
    models_dir = out_root / "models"
    models_dir.mkdir(exist_ok=True)
    pipe.save(models_dir / "descriptor_pipeline.pkl")
    scorer.save(models_dir / "scorer.pkl")

    # ── Embedding scorer (optional) ────────────────────────────────────────────
    emb_cfg = cfg.get("embedding", {})
    emb_pipe: EmbeddingPipeline | None = None
    emb_threshold: float | None = None

    if emb_cfg.get("enabled", False):
        log.info("Fitting embedding OOD scorer …")
        from detectana.io import load_embeddings_h5 as _load_emb
        ref_train_emb, _ = _load_emb(emb_cfg["reference_train_h5"])
        ref_val_emb, _   = _load_emb(emb_cfg["reference_valid_h5"])

        emb_pipe = EmbeddingPipeline()
        emb_pipe.fit(ref_train_emb)
        emb_threshold = emb_pipe.calibrate(
            ref_val_emb,
            percentile=cfg["threshold"]["percentile"],
        )
        log.info(
            "Embedding OOD threshold (%.1f-th pct of validation): %.4f",
            cfg["threshold"]["percentile"], emb_threshold,
        )
        emb_pipe.save(models_dir / "embedding_pipeline.pkl")

    # ── Per-run processing ────────────────────────────────────────────────────
    manifest_base = {
        "detectana_version": __version__,
        "config": cfg,
        "ood_threshold": threshold,
        "pca_n_components": pipe.n_components,
        "pca_variance_explained": float(pipe.explained_variance_ratio.sum()),
        "n_features": topo.n_features,
    }

    for run_cfg in cfg["runs"]:
        run_name = run_cfg["name"]
        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        log.info("=== Processing run: %s ===", run_name)
        onset = _process_run(
            run_cfg=run_cfg,
            topo=topo,
            pipe=pipe,
            scorer=scorer,
            threshold=threshold,
            emb_pipe=emb_pipe,
            emb_threshold=emb_threshold,
            emb_cfg=emb_cfg,
            cfg=cfg,
            out_root=out_root,
        )
        row = {"run": run_name, **onset.to_dict()}
        log.info("Onset for %s: %s", run_name, onset.to_dict())

        # ── Step 10: Per-run onset table and manifest ─────────────────────────
        pd.DataFrame([row]).to_csv(run_out / "onset_table.csv", index=False)
        log.info("Saved onset_table.csv → %s", run_out)

        manifest = {
            **manifest_base,
            "run": run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(run_out / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, cls=_NumpyEncoder)
        log.info("Saved manifest.json → %s", run_out)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _process_run(
    run_cfg: dict,
    topo: AspirinTopology,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    threshold: float,
    emb_pipe: EmbeddingPipeline | None,
    emb_threshold: float | None,
    emb_cfg: dict,
    cfg: dict,
    out_root: Path,
) -> OnsetResult:
    run_name = run_cfg["name"]
    run_out = out_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    desc_cache_dir = run_out / "descriptor_cache"
    desc_cache_dir.mkdir(exist_ok=True)

    io_cfg = cfg["io"]
    chem_cfg = cfg["chemistry"]
    chunk_size = io_cfg["chunk_size"]
    stride = run_cfg["stride"]
    timestep_fs = run_cfg["timestep_fs"]
    frame_time_fs = timestep_fs * stride
    force_recompute = io_cfg["force_recompute"]
    n_jobs = cfg.get("pipeline", {}).get("n_jobs", -1)

    if "hdf5" in run_cfg:
        # ── HDF5 path ─────────────────────────────────────────────────────────
        log.info("Loading HDF5 trajectory for run '%s' …", run_name)
        traj = load_pimd_trajectory_hdf5(run_cfg["hdf5"])
        _, n_beads, _, _ = traj.bead_positions.shape
        log.info("Found %d beads for run '%s'", n_beads, run_name)

        # ── Step 3+4+7: Per-bead from array (parallel threads, shared memory) ─
        log.info("Processing %d beads with n_jobs=%d …", n_beads, n_jobs)
        results: list[tuple[int, np.ndarray, np.ndarray]] = Parallel(
            n_jobs=n_jobs, prefer="threads"
        )(
            delayed(_process_bead_array)(
                bead_idx=bead_idx,
                bead_positions=traj.bead_positions[:, bead_idx, :, :],
                desc_cache_dir=desc_cache_dir,
                topo=topo,
                pipe=pipe,
                scorer=scorer,
                chem_cfg=chem_cfg,
                force_recompute=force_recompute,
                run_out=run_out,
                threshold=threshold,
            )
            for bead_idx in range(n_beads)
        )
        results.sort(key=lambda r: r[0])

        n_frames_ref = results[0][1].shape[0]
        for bead_idx, scores, _ in results:
            if scores.shape[0] != n_frames_ref:
                raise ValueError(
                    f"Bead {bead_idx:02d} has {scores.shape[0]} frames; "
                    f"expected {n_frames_ref} (same as bead 00)."
                )

        bead_score_arrays = [scores for _, scores, _ in results]
        step_arrays = results[0][2]

        bead_scores = np.stack(bead_score_arrays)
        np.save(run_out / "bead_scores.npy", bead_scores)

        centroid_scores, centroid_steps = _score_centroid_array(
            centroid_positions=traj.centroid_positions,
            topo=topo,
            pipe=pipe,
            scorer=scorer,
            force_recompute=force_recompute,
            cache_dir=desc_cache_dir,
        )
        np.save(run_out / "centroid_scores.npy", centroid_scores)
        del traj  # release RAM early

    else:
        # ── XYZ / bead_glob path (original) ──────────────────────────────────
        bead_files = sorted(glob.glob(run_cfg["bead_glob"]))
        if not bead_files:
            raise FileNotFoundError(f"No bead files matched: {run_cfg['bead_glob']}")
        n_beads = len(bead_files)
        log.info("Found %d bead files for run '%s'", n_beads, run_name)

        log.info("Processing %d beads with n_jobs=%d …", n_beads, n_jobs)
        results: list[tuple[int, np.ndarray, np.ndarray]] = Parallel(n_jobs=n_jobs)(
            delayed(_process_bead)(
                bead_idx=bead_idx,
                bead_path=bead_path,
                desc_cache_dir=desc_cache_dir,
                topo=topo,
                pipe=pipe,
                scorer=scorer,
                chem_cfg=chem_cfg,
                chunk_size=chunk_size,
                stride=stride,
                force_recompute=force_recompute,
                run_out=run_out,
                threshold=threshold,
            )
            for bead_idx, bead_path in enumerate(bead_files)
        )
        results.sort(key=lambda r: r[0])

        n_frames_ref = results[0][1].shape[0]
        for bead_idx, scores, _ in results:
            if scores.shape[0] != n_frames_ref:
                raise ValueError(
                    f"Bead {bead_idx:02d} has {scores.shape[0]} frames; "
                    f"expected {n_frames_ref} (same as bead 00). "
                    "Check for truncated bead files."
                )

        bead_score_arrays = [scores for _, scores, _ in results]
        step_arrays = results[0][2]

        bead_scores = np.stack(bead_score_arrays)
        np.save(run_out / "bead_scores.npy", bead_scores)

        centroid_scores, centroid_steps = _score_centroid(
            centroid_xyz=run_cfg["centroid_xyz"],
            topo=topo,
            pipe=pipe,
            scorer=scorer,
            chunk_size=chunk_size,
            stride=stride,
            force_recompute=force_recompute,
            cache_dir=desc_cache_dir,
        )
        np.save(run_out / "centroid_scores.npy", centroid_scores)

    # Align step arrays (bead and centroid should match; warn if not)
    if len(step_arrays) != len(centroid_steps):
        log.warning(
            "Bead step count %d ≠ centroid step count %d — truncating to min",
            len(step_arrays), len(centroid_steps),
        )
        n = min(len(step_arrays), len(centroid_steps))
        step_arrays = step_arrays[:n]
        bead_scores = bead_scores[:, :n]
        centroid_scores = centroid_scores[:n]

    # ── Step 8: Aggregate ─────────────────────────────────────────────────────
    agg_df = aggregate_bead_scores(
        bead_scores=bead_scores,
        centroid_scores=centroid_scores,
        steps=np.array(step_arrays),
        threshold=threshold,
        frame_time_fs=frame_time_fs,
    )

    # ── Embedding track (optional) ────────────────────────────────────────────
    if emb_pipe is not None and emb_threshold is not None:
        emb_bead_glob = run_cfg.get("embedding_glob", "")
        emb_centroid_h5 = run_cfg.get("centroid_embedding_h5", "")

        if emb_bead_glob and emb_centroid_h5:
            log.info("Scoring embedding track for run '%s' …", run_name)
            agg_df = _add_embedding_track(
                agg_df=agg_df,
                emb_bead_glob=emb_bead_glob,
                emb_centroid_h5=emb_centroid_h5,
                emb_pipe=emb_pipe,
                emb_threshold=emb_threshold,
                emb_cfg=emb_cfg,
                run_out=run_out,
                n_jobs=n_jobs,
            )
        else:
            log.warning(
                "Embedding enabled but 'embedding_glob' or 'centroid_embedding_h5' "
                "missing from run '%s' — skipping embedding track.", run_name,
            )

    agg_df.to_csv(run_out / "frame_aggregate.csv", index=False)
    log.info("Saved frame_aggregate.csv (%d frames)", len(agg_df))

    # ── Step 9: Onset detection ───────────────────────────────────────────────
    onset_cfg = cfg["onset"]
    onset = detect_onset(
        aggregate_df=agg_df,
        threshold=threshold,
        window_frames=onset_cfg["window_frames"],
        step_frames=onset_cfg["step_frames"],
        fraction_threshold=onset_cfg["fraction_threshold"],
    )

    # Which bead first crossed the threshold (earliest first-OOD frame across beads)
    ood_mask = bead_scores > threshold  # (n_beads, n_frames)
    if ood_mask.any():
        # argmax on bool returns index of first True; inf sentinel for beads with no OOD frame
        first_ood_frame = np.where(
            ood_mask.any(axis=1),
            np.argmax(ood_mask, axis=1),
            np.iinfo(np.int64).max,
        )
        onset.first_anomaly_bead_idx = int(np.argmin(first_ood_frame))

    # ── Step 10: Plots ────────────────────────────────────────────────────────
    try:
        _make_plots(agg_df, onset, threshold, run_name, run_out)
    except Exception as exc:
        log.warning("Plotting failed (non-fatal): %s", exc)

    return onset


# ---------------------------------------------------------------------------
# Per-bead worker (called by joblib.Parallel)
# ---------------------------------------------------------------------------

def _process_bead(
    bead_idx: int,
    bead_path: str,
    desc_cache_dir: Path,
    topo: AspirinTopology,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    chem_cfg: dict,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    run_out: Path,
    threshold: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Load/compute descriptors, run chemistry check (bead 0 only), and score.

    Returns (bead_idx, scores, steps).
    """
    bead_name = f"bead_{bead_idx:02d}"
    cache_path = desc_cache_dir / f"{bead_name}_descriptors.npz"

    steps_all, descs_all = _load_or_compute_descriptors(
        bead_path=bead_path,
        cache_path=cache_path,
        topo=topo,
        chunk_size=chunk_size,
        stride=stride,
        force_recompute=force_recompute,
        bead_name=bead_name,
    )

    if bead_idx == 0:
        log.info("Running hard-chemistry checks on bead 00 (sample) …")
        chem_rows = _chemistry_check_chunked(
            bead_path=bead_path,
            topo=topo,
            steps=steps_all,
            bond_break_cutoff=chem_cfg["bond_break_cutoff"],
            close_contact_cutoff=chem_cfg["close_contact_cutoff"],
            chunk_size=chunk_size,
            stride=stride,
        )
        chem_df = pd.DataFrame(chem_rows)
        chem_df.to_csv(run_out / "chemistry_flags_bead00.csv", index=False)
        log.info(
            "Chemistry flags — broken bonds: %d frames, close contacts: %d frames",
            chem_df["broken_bond"].sum(), chem_df["close_contact"].sum(),
        )

    X_pca = pipe.transform(descs_all)
    scores = scorer.score(X_pca)
    log.info("Bead %02d scored: max=%.3f, frac_ood=%.4f",
             bead_idx, scores.max(), (scores > threshold).mean())
    return bead_idx, scores, steps_all


# ---------------------------------------------------------------------------
# Descriptor cache helpers
# ---------------------------------------------------------------------------

def _load_or_compute_descriptors(
    bead_path: str,
    cache_path: Path,
    topo: AspirinTopology,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    bead_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, descriptors) from cache or by parsing the XYZ."""
    if cache_path.exists() and not force_recompute:
        log.info("Loading descriptor cache: %s", cache_path.name)
        data = np.load(cache_path)
        return data["steps"], data["descriptors"]

    log.info("Computing descriptors for %s …", bead_name)
    all_steps: list[np.ndarray] = []
    all_descs: list[np.ndarray] = []

    for steps_chunk, pos_chunk in iter_bead_positions(bead_path, chunk_size=chunk_size, stride=stride):
        descs_chunk = compute_descriptor_batch(pos_chunk, topo)
        all_steps.append(steps_chunk)
        all_descs.append(descs_chunk)

    steps = np.concatenate(all_steps)
    descs = np.concatenate(all_descs)

    np.savez_compressed(cache_path, steps=steps, descriptors=descs)
    log.info("Cached %d frames → %s", len(steps), cache_path.name)
    return steps, descs


def _process_bead_array(
    bead_idx: int,
    bead_positions: np.ndarray,
    desc_cache_dir: Path,
    topo: AspirinTopology,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    chem_cfg: dict,
    force_recompute: bool,
    run_out: Path,
    threshold: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Score one bead from an in-memory position array (HDF5 path)."""
    bead_name = f"bead_{bead_idx:02d}"
    cache_path = desc_cache_dir / f"{bead_name}_descriptors.npz"

    if cache_path.exists() and not force_recompute:
        log.info("Loading descriptor cache: %s", cache_path.name)
        data = np.load(cache_path)
        descs, steps = data["descriptors"], data["steps"]
    else:
        log.info("Computing descriptors for %s …", bead_name)
        descs = compute_descriptor_batch(bead_positions, topo)
        steps = np.arange(len(bead_positions), dtype=np.int64)
        np.savez_compressed(cache_path, steps=steps, descriptors=descs)
        log.info("Cached %d frames → %s", len(steps), cache_path.name)

    if bead_idx == 0:
        log.info("Running hard-chemistry checks on bead 00 …")
        flags_batch = check_chemistry_batch(
            bead_positions, topo,
            chem_cfg["bond_break_cutoff"],
            chem_cfg["close_contact_cutoff"],
        )
        chem_rows = [{"step": int(s), **f.to_dict()} for s, f in zip(steps, flags_batch)]
        chem_df = pd.DataFrame(chem_rows)
        chem_df.to_csv(run_out / "chemistry_flags_bead00.csv", index=False)
        log.info(
            "Chemistry flags — broken bonds: %d frames, close contacts: %d frames",
            chem_df["broken_bond"].sum(), chem_df["close_contact"].sum(),
        )

    X_pca = pipe.transform(descs)
    scores = scorer.score(X_pca)
    log.info("Bead %02d scored: max=%.3f, frac_ood=%.4f",
             bead_idx, scores.max(), (scores > threshold).mean())
    return bead_idx, scores, steps


def _score_centroid_array(
    centroid_positions: np.ndarray,
    topo: AspirinTopology,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    force_recompute: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Score centroid from an in-memory position array (HDF5 path)."""
    cache_path = cache_dir / "centroid_descriptors.npz"

    if cache_path.exists() and not force_recompute:
        log.info("Loading descriptor cache: centroid_descriptors.npz")
        data = np.load(cache_path)
        descs, steps = data["descriptors"], data["steps"]
    else:
        log.info("Computing descriptors for centroid …")
        descs = compute_descriptor_batch(centroid_positions, topo)
        steps = np.arange(len(centroid_positions), dtype=np.int64)
        np.savez_compressed(cache_path, steps=steps, descriptors=descs)
        log.info("Cached %d frames → centroid_descriptors.npz", len(steps))

    X_pca = pipe.transform(descs)
    scores = scorer.score(X_pca)
    return scores, steps


def _score_centroid(
    centroid_xyz: str,
    topo: AspirinTopology,
    pipe: DescriptorPipeline,
    scorer: MahalanobisScorer,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / "centroid_descriptors.npz"
    steps, descs = _load_or_compute_descriptors(
        bead_path=centroid_xyz,
        cache_path=cache_path,
        topo=topo,
        chunk_size=chunk_size,
        stride=stride,
        force_recompute=force_recompute,
        bead_name="centroid",
    )
    X_pca = pipe.transform(descs)
    scores = scorer.score(X_pca)
    return scores, steps


# ---------------------------------------------------------------------------
# Chemistry check (chunked, keeps only per-frame flag summary)
# ---------------------------------------------------------------------------

def _chemistry_check_chunked(
    bead_path: str,
    topo: AspirinTopology,
    steps: np.ndarray,
    bond_break_cutoff: float,
    close_contact_cutoff: float,
    chunk_size: int,
    stride: int,
) -> list[dict]:
    rows: list[dict] = []
    for steps_chunk, pos_chunk in iter_bead_positions(bead_path, chunk_size=chunk_size, stride=stride):
        flags_chunk = check_chemistry_batch(pos_chunk, topo, bond_break_cutoff, close_contact_cutoff)
        for step, flags in zip(steps_chunk, flags_chunk):
            row = {"step": int(step)}
            row.update(flags.to_dict())
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Embedding track helpers
# ---------------------------------------------------------------------------

def _score_bead_embedding(
    bead_h5: str,
    emb_pipe: EmbeddingPipeline,
    bead_idx: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Load one bead's pre-computed embeddings and return its OOD scores.

    Returns (bead_idx, scores, steps).
    """
    embeddings, steps = load_embeddings_h5(bead_h5)
    scores = emb_pipe.score(embeddings)
    log.info(
        "Embedding bead %02d scored: max=%.3f, n_frames=%d",
        bead_idx, scores.max(), len(scores),
    )
    return bead_idx, scores, steps


def _add_embedding_track(
    agg_df: pd.DataFrame,
    emb_bead_glob: str,
    emb_centroid_h5: str,
    emb_pipe: EmbeddingPipeline,
    emb_threshold: float,
    emb_cfg: dict,
    run_out: Path,
    n_jobs: int,
) -> pd.DataFrame:
    """Score all bead and centroid embedding HDF5 files and merge into agg_df."""
    bead_h5_files = sorted(glob.glob(emb_bead_glob))
    if not bead_h5_files:
        raise FileNotFoundError(f"No embedding bead files matched: {emb_bead_glob}")
    log.info("Found %d embedding bead files", len(bead_h5_files))

    emb_results: list[tuple[int, np.ndarray, np.ndarray]] = Parallel(n_jobs=n_jobs)(
        delayed(_score_bead_embedding)(bead_h5, emb_pipe, bead_idx)
        for bead_idx, bead_h5 in enumerate(bead_h5_files)
    )
    emb_results.sort(key=lambda r: r[0])

    # Guard against mismatched frame counts across bead embedding files
    n_emb_frames_ref = emb_results[0][1].shape[0]
    for bead_idx, scores, _ in emb_results:
        if scores.shape[0] != n_emb_frames_ref:
            raise ValueError(
                f"Embedding bead {bead_idx:02d} has {scores.shape[0]} frames; "
                f"expected {n_emb_frames_ref}. Check for mismatched HDF5 files."
            )

    emb_bead_scores = np.stack([s for _, s, _ in emb_results])  # (n_beads, n_emb_frames)
    emb_steps = emb_results[0][2]
    np.save(run_out / "emb_bead_scores.npy", emb_bead_scores)

    # Centroid
    emb_centroid_emb, emb_centroid_steps = load_embeddings_h5(emb_centroid_h5)
    emb_centroid_scores = emb_pipe.score(emb_centroid_emb)
    np.save(run_out / "emb_centroid_scores.npy", emb_centroid_scores)

    # Align bead and centroid step arrays
    if len(emb_steps) != len(emb_centroid_steps):
        log.warning(
            "Embedding bead step count %d ≠ centroid step count %d — truncating to min",
            len(emb_steps), len(emb_centroid_steps),
        )
        n = min(len(emb_steps), len(emb_centroid_steps))
        emb_steps = emb_steps[:n]
        emb_bead_scores = emb_bead_scores[:, :n]
        emb_centroid_scores = emb_centroid_scores[:n]

    return add_embedding_scores(
        agg_df=agg_df,
        emb_bead_scores=emb_bead_scores,
        emb_centroid_scores=emb_centroid_scores,
        emb_steps=emb_steps,
        emb_threshold=emb_threshold,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(
    agg_df: pd.DataFrame,
    onset: OnsetResult,
    threshold: float,
    run_name: str,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    has_emb = "emb_bead_max" in agg_df.columns
    n_panels = 6 if has_emb else 3
    time_ps = agg_df["time_ps"].to_numpy()

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

    # ── Geometric track ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(time_ps, agg_df["bead_max"], lw=0.5, label="bead max", color="steelblue")
    ax.plot(time_ps, agg_df["bead_p95"], lw=0.5, label="bead p95", color="cornflowerblue")
    ax.axhline(threshold, color="red", ls="--", lw=1, label=f"threshold={threshold:.2f}")
    _mark_onset(ax, time_ps, onset.persistent_bead_anomaly_frame, "geo bead onset", "orange")
    ax.set_ylabel("Mahalanobis distance")
    ax.legend(fontsize=7)
    ax.set_title(f"{run_name} — geometric track")

    ax = axes[1]
    ax.plot(time_ps, agg_df["bead_frac_ood"], lw=0.5, color="darkorange", label="bead frac OOD")
    ax.set_ylabel("Fraction OOD")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(time_ps, agg_df["centroid_score"], lw=0.5, color="forestgreen", label="centroid score")
    ax.axhline(threshold, color="red", ls="--", lw=1)
    _mark_onset(ax, time_ps, onset.centroid_anomaly_frame, "geo centroid onset", "purple")
    ax.set_ylabel("Mahalanobis distance")
    ax.legend(fontsize=7)
    if not has_emb:
        ax.set_xlabel("Time (ps)")

    # ── Embedding track (when present) ───────────────────────────────────────
    if has_emb:
        ax = axes[3]
        ax.plot(time_ps, agg_df["emb_bead_max"], lw=0.5, label="emb bead max", color="royalblue")
        ax.plot(time_ps, agg_df["emb_bead_p95"], lw=0.5, label="emb bead p95", color="skyblue")
        _mark_onset(ax, time_ps, onset.embedding_persistent_bead_onset_frame, "emb bead onset", "orange")
        ax.set_ylabel("Emb. Mahalanobis")
        ax.legend(fontsize=7)
        ax.set_title(f"{run_name} — embedding track")

        ax = axes[4]
        ax.plot(time_ps, agg_df["emb_bead_frac_ood"], lw=0.5, color="darkorange", label="emb bead frac OOD")
        ax.set_ylabel("Fraction OOD")
        ax.legend(fontsize=7)

        ax = axes[5]
        ax.plot(time_ps, agg_df["emb_centroid_score"], lw=0.5, color="teal", label="emb centroid score")
        _mark_onset(ax, time_ps, onset.embedding_centroid_onset_frame, "emb centroid onset", "purple")
        ax.set_ylabel("Emb. Mahalanobis")
        ax.set_xlabel("Time (ps)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(plots_dir / "score_vs_time.png", dpi=150)
    plt.close(fig)
    log.info("Saved score_vs_time.png")


def _mark_onset(ax, time_ps, frame_idx, label, color):
    if frame_idx is not None and frame_idx < len(time_ps):
        ax.axvline(time_ps[frame_idx], color=color, ls=":", lw=1.2, label=f"{label} onset")
