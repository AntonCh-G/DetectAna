"""Pipeline orchestrator: the ten workflow steps, end to end.

Step 1.  Load trajectories and metadata.
Step 2.  Validate units, atom order, indexing.
Step 3.  Compute hard-chemistry checks.
Step 4.  Compute internal-coordinate fingerprints.
Step 5.  Fit OOD statistics on training frames (a reference bundle, scored by T²).
Step 6.  Calibrate thresholds on validation frames (percentiles of T²).
Step 7.  Score bead and centroid frames.
Step 8.  Aggregate bead scores per timestep.
Step 9.  Detect persistent anomaly onset.
Step 10. Save tables, plots, and manifest.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from detectana import __version__
from detectana.aggregator import add_embedding_scores, aggregate_bead_scores
from detectana.attribution import GroupStrength
from detectana.bundle import SCORES, ReferenceBundle
from detectana.bundle import fit as fit_bundle
from detectana.descriptors import compute_descriptor_batch
from detectana.embedding_scorer import EmbeddingPipeline
from detectana.io import (
    MoleculeSpec,
    iter_bead_positions,
    load_embeddings_h5,
    load_pimd_trajectory_hdf5,
    load_reference_frames,
    load_single_frame,
)
from detectana.onset import OnsetResult, detect_onset, resolve_onset_rule
from detectana.topology import MoleculeTopology, build_topology, check_chemistry_batch  # noqa: F401

log = logging.getLogger(__name__)


def _stable_segment(scores: np.ndarray, stable_fraction: float) -> np.ndarray:
    """Leading part of a score series, used to measure frame autocorrelation.

    The measurement has to come from a quiet stretch: a series containing the
    anomaly would overestimate the correlation and make the false-alarm bound
    look better than it is. This assumes the run starts inside the training
    distribution, which is the same assumption the onset question rests on.
    """
    n = max(3, int(len(scores) * float(stable_fraction)))
    return scores[:n]


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


def resolve_threshold_percentiles(threshold_cfg: dict) -> tuple[float, float]:
    """Return ``(bead_percentile, centroid_percentile)`` from the threshold config.

    Two schemas are accepted, so older configs keep working:

    - ``percentile`` alone — both tracks share it. This is the common case.
    - ``bead_percentile`` and/or ``centroid_percentile`` — per-track values, each
      falling back to ``percentile`` when only one of them is given. A sensitive
      threshold for the beads with a strict one for the centroid is a real
      choice: beads are the early-warning signal, the centroid is the claim that
      the whole molecule has moved.
    """
    base = threshold_cfg.get("percentile")
    bead = threshold_cfg.get("bead_percentile", base)
    centroid = threshold_cfg.get("centroid_percentile", base)
    if bead is None or centroid is None:
        raise KeyError(
            "threshold config needs 'percentile', or 'bead_percentile' and "
            f"'centroid_percentile'; got keys {sorted(threshold_cfg)}"
        )
    return float(bead), float(centroid)


#: Exceedance above this marks a frame: some test is above its own threshold.
SIGNAL_THRESHOLD = 1.0


def _test_key(track: str, score: str) -> str:
    return f"{track}/{score}"


def _test_keys(bundle: ReferenceBundle) -> list[str]:
    return [_test_key(t, s) for t, s in bundle.tests]


def _test_scores(bundle: ReferenceBundle, features: np.ndarray) -> dict[str, np.ndarray]:
    """Raw score of every active test, keyed ``track/score`` (T² or SPE units)."""
    scores = bundle.score_features(features)
    return {_test_key(t, s): scores[t][s] for t, s in bundle.tests}


def _exceedance(
    bundle: ReferenceBundle, raw: dict[str, np.ndarray], role: str
) -> tuple[np.ndarray, np.ndarray]:
    """Largest score ÷ threshold over the tests, and the index of that test.

    Works on arrays of any shape (beads × frames or frames). Above 1 = marked.
    """
    ratios = np.stack([
        raw[_test_key(t, s)] / bundle.thresholds(t)[s][role] for t, s in bundle.tests
    ])
    return ratios.max(axis=0), ratios.argmax(axis=0)


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Named arrays to a compressed .npz (keys such as ``bonds/T2``)."""
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _top_groups(
    bundle: ReferenceBundle,
    features: np.ndarray,
    keys: list[str],
    top_test: np.ndarray,
    frames: np.ndarray,
    k: int,
) -> dict[int, list[GroupStrength]]:
    """The ``k`` strongest descriptor groups (RBC) of each frame, for its top test."""
    out: dict[int, list[GroupStrength]] = {}
    for i, key in enumerate(keys):
        rows = frames[top_test[frames] == i]
        if len(rows) == 0:
            continue
        track, score = key.split("/")
        att = bundle.attribute_features(features[rows], track, score)
        for j, frame in enumerate(rows):
            out[int(frame)] = att.top(frame=j, k=k)
    return out


def _builtin_layout(bundle: ReferenceBundle, topo: MoleculeTopology) -> bool:
    """True when the bundle's numbers are the joined built-in descriptor, so
    ``compute_descriptor_batch`` and its existing cache files serve it."""
    return bundle.all_feature_names == tuple(topo.feature_names)


def _features(bundle: ReferenceBundle, topo: MoleculeTopology, positions: np.ndarray) -> np.ndarray:
    if _builtin_layout(bundle, topo):
        return compute_descriptor_batch(positions, topo)
    return bundle.compute_features(positions)


def _cache_file(cache_dir: Path, name: str, bundle: ReferenceBundle, topo: MoleculeTopology) -> Path:
    """Descriptor cache path; other descriptor sets get their own tagged file."""
    if _builtin_layout(bundle, topo):
        return cache_dir / f"{name}_descriptors.npz"
    descriptors = bundle.manifest["descriptors"]
    key = json.dumps({n: [d["code_sha256"], d["feature_names"]] for n, d in descriptors.items()})
    return cache_dir / f"{name}_descriptors_{hashlib.sha256(key.encode()).hexdigest()[:10]}.npz"


def time_per_step_fs(run_cfg: dict) -> float:
    """Femtoseconds per unit of the ``step`` column of a run.

    XYZ runs carry i-PI's MD step number (the output stride is already in it), so
    one step is one timestep. HDF5 runs carry the frame index, one per output
    stride, so one step is timestep × stride.
    """
    if "hdf5" in run_cfg:
        return float(run_cfg["timestep_fs"]) * float(run_cfg["stride"])
    return float(run_cfg["timestep_fs"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> None:
    """Execute the full anomaly-onset workflow for all configured runs.

    Parameters
    ----------
    cfg : dict loaded from YAML config (see config/example.yaml).
    """
    out_root = Path(cfg["io"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1–2: Topology + reference data ─────────────────────────────────
    # Topology comes first: the first run's initial.xyz defines the molecule, and
    # every reference and trajectory frame is then validated against it.
    chem_cfg_global = cfg["chemistry"]
    first_run = cfg["runs"][0]
    topo = build_topology(
        first_run["initial_xyz"],
        nl_mult=chem_cfg_global["nl_mult"],
        ring_atoms=chem_cfg_global.get("ring_atoms"),
    )
    spec = MoleculeSpec(n_atoms=topo.n_atoms, atom_types=tuple(topo.atom_types))
    log.info(
        "Molecule: %d atoms (%s) from %s",
        spec.n_atoms, "".join(spec.atom_types), first_run["initial_xyz"],
    )
    log.info(
        "Topology: %d bonds, %d angles, %d dihedrals, ring=%s → %d features",
        len(topo.bonds), len(topo.angles), len(topo.dihedrals),
        topo.ring_atoms or "none", topo.n_features,
    )

    # Every other run must be the same molecule in the same atom order.
    for run_cfg in cfg["runs"][1:]:
        load_single_frame(run_cfg["initial_xyz"], spec=spec)

    log.info("Loading reference data …")
    ref_cfg = cfg["reference"]
    train_pos, _, _ = load_reference_frames(ref_cfg["train"], spec=spec)
    valid_pos, _, _ = load_reference_frames(ref_cfg["valid"], spec=spec)

    # ── Step 4–6: Fit the reference bundle on training data, calibrate on valid ─
    # One fitting path for batch and live use (ADR 0005): descriptors,
    # standardisation, PCA and T² on the training frames; thresholds are
    # percentiles of validation T². Scores everywhere below are T².
    log.info("Fitting the reference bundle (descriptors, PCA, T²) on training frames …")
    bead_percentile, centroid_percentile = resolve_threshold_percentiles(cfg["threshold"])
    desc_cfg = cfg["descriptor"]
    onset_cfg_global = cfg["onset"]
    if "fraction_threshold" not in onset_cfg_global:
        # Budget mode derives the fraction per run from its length, so the bundle
        # has no fit-time value; it records the library default instead and the
        # batch pipeline below always resolves its own per run.
        log.info(
            "onset.fraction_threshold not set (false_alarm_budget mode): the bundle "
            "records the default 0.20; each run derives its own fraction."
        )
    # Tracks and scores come from the config (absent → the library default,
    # ADR 0006). Every test is thresholded at its share of each role's budget.
    tracks_cfg = cfg.get("tracks")
    if tracks_cfg is not None and not isinstance(tracks_cfg, dict):
        raise ValueError("config 'tracks' must map track names to lists of descriptors")
    bundle = fit_bundle(
        train_pos, valid_pos, topo,
        tracks=tracks_cfg,
        scores=tuple(cfg.get("scores", SCORES)),
        pca_variance=desc_cfg["pca_variance"],
        random_seed=desc_cfg["random_seed"],
        percentiles={"bead": bead_percentile, "centroid": centroid_percentile},
        window_frames=onset_cfg_global["window_frames"],
        step_frames=onset_cfg_global["step_frames"],
        fraction_threshold=onset_cfg_global.get("fraction_threshold", 0.20),
    )
    for track in bundle.tracks:
        t = bundle.manifest["tracks"][track]
        log.info(
            "Track %-10s %3d numbers → %3d PCA components (%.1f%% variance)",
            track, t["n_features"], t["n_components"], t["pca_variance_explained"] * 100,
        )
    for name, why in bundle.manifest["dropped_tracks"].items():
        log.info("Track %s dropped: %s", name, why)
    thresholds = {
        _test_key(t, s): bundle.thresholds(t)[s] for t, s in bundle.tests
    }
    log.info(
        "%d tests; thresholds at %.1f-th (bead) / %.1f-th (centroid) percentile of "
        "validation in total, split equally: %s",
        len(thresholds), bead_percentile, centroid_percentile,
        ", ".join(f"{k} {v['bead']:.3g}/{v['centroid']:.3g}" for k, v in thresholds.items()),
    )

    # Save the fit
    models_dir = out_root / "models"
    models_dir.mkdir(exist_ok=True)
    bundle_path = models_dir / "geometric.bundle"
    bundle.save(bundle_path)
    log.info("Saved reference bundle → %s", bundle_path)

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
            percentile=bead_percentile,
        )
        log.info(
            "Embedding OOD threshold (%.1f-th pct of validation): %.4f",
            bead_percentile, emb_threshold,
        )
        emb_pipe.save(models_dir / "embedding_pipeline.pkl")

    # ── Per-run processing ────────────────────────────────────────────────────
    manifest_base = {
        "detectana_version": __version__,
        "config": cfg,
        # bead_scores.npy / centroid_scores.npy and the score columns hold the
        # exceedance (largest score ÷ threshold over the tests; > 1 = marked) for
        # the bead and centroid roles. Raw per-test scores (T², SPE) are in
        # bead_test_scores.npz / centroid_test_scores.npz.
        "statistic": "exceedance",
        "signal_threshold": SIGNAL_THRESHOLD,
        "test_score_files": {"bead": "bead_test_scores.npz", "centroid": "centroid_test_scores.npz"},
        "reference_bundle": str(bundle_path),
        "tests": list(thresholds),
        "thresholds": thresholds,
        "budget": bundle.manifest["budget"],
        "tracks": {
            t: {k: bundle.manifest["tracks"][t][k]
                for k in ("descriptors", "n_features", "n_components", "pca_variance_explained")}
            for t in bundle.tracks
        },
        "dropped_tracks": bundle.manifest["dropped_tracks"],
        "bead_percentile": bead_percentile,
        "centroid_percentile": centroid_percentile,
        "n_features": len(bundle.all_feature_names),
        "molecule": {
            "n_atoms": topo.n_atoms,
            "atom_types": list(topo.atom_types),
            "ring_atoms": list(topo.ring_atoms),
            "initial_xyz": str(first_run["initial_xyz"]),
        },
    }

    onset_rows: list[dict] = []
    for run_cfg in cfg["runs"]:
        run_name = run_cfg["name"]
        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        log.info("=== Processing run: %s ===", run_name)
        onset, onset_design = _process_run(
            run_cfg=run_cfg,
            topo=topo,
            spec=spec,
            bundle=bundle,
            bead_percentile=bead_percentile,
            centroid_percentile=centroid_percentile,
            emb_pipe=emb_pipe,
            emb_threshold=emb_threshold,
            emb_cfg=emb_cfg,
            cfg=cfg,
            out_root=out_root,
        )
        row = {"run": run_name, **onset.to_dict()}
        onset_rows.append(row)
        log.info("Onset for %s: %s", run_name, onset.to_dict())

        # ── Step 10: Per-run onset table and manifest ─────────────────────────
        pd.DataFrame([row]).to_csv(run_out / "onset_table.csv", index=False)
        log.info("Saved onset_table.csv → %s", run_out)

        manifest = {
            **manifest_base,
            "run": run_name,
            "onset_design": onset_design,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(run_out / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, cls=_NumpyEncoder)
        log.info("Saved manifest.json → %s", run_out)

    # Runs are the independent statistical units, so the cross-run table is what
    # a model comparison is read from.
    pd.DataFrame(onset_rows).to_csv(out_root / "onset_summary.csv", index=False)
    log.info("Saved onset_summary.csv (%d runs) → %s", len(onset_rows), out_root)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _process_run(
    run_cfg: dict,
    topo: MoleculeTopology,
    spec: MoleculeSpec,
    bundle: ReferenceBundle,
    bead_percentile: float,
    centroid_percentile: float,
    emb_pipe: EmbeddingPipeline | None,
    emb_threshold: float | None,
    emb_cfg: dict,
    cfg: dict,
    out_root: Path,
) -> tuple[OnsetResult, dict]:
    run_name = run_cfg["name"]
    run_out = out_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    desc_cache_dir = run_out / "descriptor_cache"
    desc_cache_dir.mkdir(exist_ok=True)

    io_cfg = cfg["io"]
    chem_cfg = cfg["chemistry"]
    chunk_size = io_cfg["chunk_size"]
    stride = run_cfg["stride"]
    step_time_fs = time_per_step_fs(run_cfg)
    force_recompute = io_cfg["force_recompute"]
    n_jobs = cfg.get("pipeline", {}).get("n_jobs", -1)

    # (bead_idx, raw test scores, steps) per bead, from whichever reader the run uses.
    results: list[tuple[int, dict[str, np.ndarray], np.ndarray]]
    keys = _test_keys(bundle)

    if "hdf5" in run_cfg:
        # ── HDF5 path ─────────────────────────────────────────────────────────
        log.info("Loading HDF5 trajectory for run '%s' …", run_name)
        traj = load_pimd_trajectory_hdf5(run_cfg["hdf5"], spec=spec)
        _, n_beads, _, _ = traj.bead_positions.shape
        log.info("Found %d beads for run '%s'", n_beads, run_name)

        # ── Step 3+4+7: Per-bead from array (parallel threads, shared memory) ─
        log.info("Processing %d beads with n_jobs=%d …", n_beads, n_jobs)
        results = Parallel(
            n_jobs=n_jobs, prefer="threads"
        )(
            delayed(_process_bead_array)(
                bead_idx=bead_idx,
                bead_positions=traj.bead_positions[:, bead_idx, :, :],
                desc_cache_dir=desc_cache_dir,
                topo=topo,
                bundle=bundle,
                chem_cfg=chem_cfg,
                force_recompute=force_recompute,
                run_out=run_out,
            )
            for bead_idx in range(n_beads)
        )
        results.sort(key=lambda r: r[0])

        n_frames_ref = len(results[0][2])
        for bead_idx, _, bead_steps in results:
            if len(bead_steps) != n_frames_ref:
                raise ValueError(
                    f"Bead {bead_idx:02d} has {len(bead_steps)} frames; "
                    f"expected {n_frames_ref} (same as bead 00)."
                )

        bead_raw = {k: np.stack([raw[k] for _, raw, _ in results]) for k in keys}
        step_arrays = results[0][2]

        centroid_raw, centroid_steps, centroid_descs = _score_centroid_array(
            centroid_positions=traj.centroid_positions,
            topo=topo,
            bundle=bundle,
            force_recompute=force_recompute,
            cache_dir=desc_cache_dir,
        )
        del traj  # release RAM early

    else:
        # ── XYZ / bead_glob path (original) ──────────────────────────────────
        bead_files = sorted(glob.glob(run_cfg["bead_glob"]))
        if not bead_files:
            raise FileNotFoundError(f"No bead files matched: {run_cfg['bead_glob']}")
        n_beads = len(bead_files)
        log.info("Found %d bead files for run '%s'", n_beads, run_name)

        log.info("Processing %d beads with n_jobs=%d …", n_beads, n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_bead)(
                bead_idx=bead_idx,
                bead_path=bead_path,
                desc_cache_dir=desc_cache_dir,
                topo=topo,
                spec=spec,
                bundle=bundle,
                chem_cfg=chem_cfg,
                chunk_size=chunk_size,
                stride=stride,
                force_recompute=force_recompute,
                run_out=run_out,
            )
            for bead_idx, bead_path in enumerate(bead_files)
        )
        results.sort(key=lambda r: r[0])

        n_frames_ref = len(results[0][2])
        for bead_idx, _, bead_steps in results:
            if len(bead_steps) != n_frames_ref:
                raise ValueError(
                    f"Bead {bead_idx:02d} has {len(bead_steps)} frames; "
                    f"expected {n_frames_ref} (same as bead 00). "
                    "Check for truncated bead files."
                )

        bead_raw = {k: np.stack([raw[k] for _, raw, _ in results]) for k in keys}
        step_arrays = results[0][2]

        centroid_raw, centroid_steps, centroid_descs = _score_centroid(
            centroid_xyz=run_cfg["centroid_xyz"],
            topo=topo,
            spec=spec,
            bundle=bundle,
            chunk_size=chunk_size,
            stride=stride,
            force_recompute=force_recompute,
            cache_dir=desc_cache_dir,
        )

    # Align step arrays (bead and centroid should match; warn if not)
    if len(step_arrays) != len(centroid_steps):
        log.warning(
            "Bead step count %d ≠ centroid step count %d — truncating to min",
            len(step_arrays), len(centroid_steps),
        )
        n = min(len(step_arrays), len(centroid_steps))
        step_arrays = step_arrays[:n]
        bead_raw = {k: v[:, :n] for k, v in bead_raw.items()}
        centroid_raw = {k: v[:n] for k, v in centroid_raw.items()}
        centroid_descs = centroid_descs[:n]

    # ── Exceedance: one signal per role, threshold 1 ──────────────────────────
    # Beads use the bead role, the centroid the centroid role. A frame is marked
    # when some test is above its own (budget-split) threshold.
    bead_scores, _ = _exceedance(bundle, bead_raw, "bead")
    centroid_scores, centroid_top = _exceedance(bundle, centroid_raw, "centroid")
    np.save(run_out / "bead_scores.npy", bead_scores)
    np.save(run_out / "centroid_scores.npy", centroid_scores)
    _save_arrays(run_out / "bead_test_scores.npz", bead_raw)
    _save_arrays(run_out / "centroid_test_scores.npz", centroid_raw)

    # ── Step 8: Aggregate ─────────────────────────────────────────────────────
    agg_df = aggregate_bead_scores(
        bead_scores=bead_scores,
        centroid_scores=centroid_scores,
        steps=np.array(step_arrays),
        threshold=SIGNAL_THRESHOLD,
        frame_time_fs=step_time_fs,
    )
    agg_df["centroid_top_test"] = [keys[i] for i in centroid_top]
    for k in keys:
        agg_df[f"centroid_{k.replace('/', '_')}"] = centroid_raw[k]
    # Attribution (RBC) of marked centroid frames, for the test that marked them.
    top_groups: list[str | None] = [None] * len(agg_df)
    top_atoms: list[str | None] = [None] * len(agg_df)
    for frame, groups in _top_groups(
        bundle, centroid_descs, keys, centroid_top, np.flatnonzero(centroid_scores > SIGNAL_THRESHOLD), k=3,
    ).items():
        top_groups[frame] = "; ".join(f"{g.group_id}={g.strength:.4g}" for g in groups)
        top_atoms[frame] = "-".join(map(str, groups[0].atoms))
    agg_df["centroid_top_groups"] = top_groups
    agg_df["centroid_top_atoms"] = top_atoms

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
    # The window rule, not the OOD threshold, is what controls false alarms, so
    # resolve it explicitly and record the arithmetic (see onset.resolve_onset_rule).
    onset_cfg = cfg["onset"]
    stable_series = _stable_segment(
        agg_df["centroid_score"].to_numpy(dtype=np.float64),
        onset_cfg.get("stable_fraction", 0.1),
    )
    fraction_threshold, onset_design = resolve_onset_rule(
        onset_cfg=onset_cfg,
        false_flag_rate=1.0 - bead_percentile / 100.0,
        n_frames=len(agg_df),
        stable_series=stable_series,
    )
    if centroid_percentile != bead_percentile:
        # The two tracks have different false-flag rates but share one
        # fraction_threshold, so take the stricter of the two rules: the fraction
        # the looser track needs to stay inside its false-alarm budget.
        centroid_fraction, centroid_design = resolve_onset_rule(
            onset_cfg=onset_cfg,
            false_flag_rate=1.0 - centroid_percentile / 100.0,
            n_frames=len(agg_df),
            stable_series=stable_series,
        )
        bead_design = onset_design
        fraction_threshold = max(fraction_threshold, centroid_fraction)
        onset_design = {
            "fraction_threshold_used": fraction_threshold,
            "bead": bead_design,
            "centroid": centroid_design,
        }
        log.info(
            "Onset rule across two tracks: fraction_threshold=%.4f "
            "(bead needs %.4f, centroid %.4f)",
            fraction_threshold, bead_design["fraction_threshold"], centroid_fraction,
        )
    onset = detect_onset(
        aggregate_df=agg_df,
        threshold=SIGNAL_THRESHOLD,
        window_frames=onset_cfg["window_frames"],
        step_frames=onset_cfg["step_frames"],
        fraction_threshold=fraction_threshold,
    )

    # Which bead first crossed the threshold (earliest first-OOD frame across beads)
    ood_mask = bead_scores > SIGNAL_THRESHOLD  # (n_beads, n_frames)
    if ood_mask.any():
        # argmax on bool returns index of first True; inf sentinel for beads with no OOD frame
        first_ood_frame = np.where(
            ood_mask.any(axis=1),
            np.argmax(ood_mask, axis=1),
            np.iinfo(np.int64).max,
        )
        onset.first_anomaly_bead_idx = int(np.argmin(first_ood_frame))

    # What drove each alarm (RBC of the centroid's top test). An onset frame is
    # the start of the first window that fired and may itself be unmarked, and
    # the first marked frame may be a stray one, so the explanation is of the
    # most exceeded centroid frame within that window.
    window = int(onset_cfg["window_frames"])
    onset_frames = {
        kind: getattr(onset, f"{kind}_frame")
        for kind in ("first_bead_anomaly", "persistent_bead_anomaly", "centroid_anomaly", "collective_anomaly")
        if getattr(onset, f"{kind}_frame") is not None
    }
    explained_frame = {}
    for kind, start in onset_frames.items():
        explained_frame[kind] = start + int(np.argmax(centroid_scores[start:start + window]))
    explained = _top_groups(
        bundle, centroid_descs, keys, centroid_top, np.array(sorted(set(explained_frame.values())), dtype=int), k=5,
    )
    att_columns = ["onset", "onset_frame", "frame", "step", "test", "rank", "group", "atoms", "size", "strength"]
    att_rows = [
        {"onset": kind, "onset_frame": onset_frames[kind], "frame": frame, "step": int(step_arrays[frame]),
         "test": keys[centroid_top[frame]], "rank": rank, "group": g.group_id,
         "atoms": "-".join(map(str, g.atoms)), "size": g.size, "strength": g.strength}
        for kind, frame in explained_frame.items()
        for rank, g in enumerate(explained[frame], start=1)
    ]
    pd.DataFrame(att_rows, columns=att_columns).to_csv(run_out / "onset_attribution.csv", index=False)

    # ── Step 10: Plots ────────────────────────────────────────────────────────
    try:
        _make_plots(agg_df, onset, SIGNAL_THRESHOLD, SIGNAL_THRESHOLD, run_name, run_out)
    except Exception as exc:
        log.warning("Plotting failed (non-fatal): %s", exc)

    return onset, onset_design


# ---------------------------------------------------------------------------
# Per-bead worker (called by joblib.Parallel)
# ---------------------------------------------------------------------------

def _process_bead(
    bead_idx: int,
    bead_path: str,
    desc_cache_dir: Path,
    topo: MoleculeTopology,
    spec: MoleculeSpec,
    bundle: ReferenceBundle,
    chem_cfg: dict,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    run_out: Path,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    """Load/compute descriptors, run chemistry check (bead 0 only), and score.

    Returns (bead_idx, raw score per test, steps).
    """
    bead_name = f"bead_{bead_idx:02d}"
    cache_path = _cache_file(desc_cache_dir, bead_name, bundle, topo)

    steps_all, descs_all = _load_or_compute_descriptors(
        bead_path=bead_path,
        cache_path=cache_path,
        bundle=bundle,
        topo=topo,
        spec=spec,
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
            spec=spec,
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

    raw = _test_scores(bundle, descs_all)
    signal, _ = _exceedance(bundle, raw, "bead")
    log.info("Bead %02d scored: max exceedance=%.3f, fraction marked=%.4f",
             bead_idx, signal.max(), (signal > SIGNAL_THRESHOLD).mean())
    return bead_idx, raw, steps_all


# ---------------------------------------------------------------------------
# Descriptor cache helpers
# ---------------------------------------------------------------------------

def _load_or_compute_descriptors(
    bead_path: str,
    cache_path: Path,
    bundle: ReferenceBundle,
    topo: MoleculeTopology,
    spec: MoleculeSpec,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    bead_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, descriptors) from cache or by parsing the XYZ."""
    cached = _read_cache(cache_path, bundle, force_recompute)
    if cached is not None:
        return cached

    log.info("Computing descriptors for %s …", bead_name)
    all_steps: list[np.ndarray] = []
    all_descs: list[np.ndarray] = []

    for steps_chunk, pos_chunk in iter_bead_positions(
        bead_path, chunk_size=chunk_size, stride=stride, spec=spec
    ):
        descs_chunk = _features(bundle, topo, pos_chunk)
        all_steps.append(steps_chunk)
        all_descs.append(descs_chunk)

    steps = np.concatenate(all_steps)
    descs = np.concatenate(all_descs)

    np.savez_compressed(cache_path, steps=steps, descriptors=descs)
    log.info("Cached %d frames → %s", len(steps), cache_path.name)
    return steps, descs


def _read_cache(
    cache_path: Path, bundle: ReferenceBundle, force_recompute: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    """(steps, descriptors) from a cache file whose width fits the bundle, else None."""
    if force_recompute or not cache_path.exists():
        return None
    data = np.load(cache_path)
    if data["descriptors"].shape[1] != len(bundle.all_feature_names):
        log.info("Descriptor cache %s does not fit the bundle; recomputing", cache_path.name)
        return None
    log.info("Loading descriptor cache: %s", cache_path.name)
    return data["steps"], data["descriptors"]


def _process_bead_array(
    bead_idx: int,
    bead_positions: np.ndarray,
    desc_cache_dir: Path,
    topo: MoleculeTopology,
    bundle: ReferenceBundle,
    chem_cfg: dict,
    force_recompute: bool,
    run_out: Path,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    """Score one bead from an in-memory position array (HDF5 path)."""
    bead_name = f"bead_{bead_idx:02d}"
    cache_path = _cache_file(desc_cache_dir, bead_name, bundle, topo)

    cached = _read_cache(cache_path, bundle, force_recompute)
    if cached is not None:
        steps, descs = cached
    else:
        log.info("Computing descriptors for %s …", bead_name)
        descs = _features(bundle, topo, bead_positions)
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

    raw = _test_scores(bundle, descs)
    signal, _ = _exceedance(bundle, raw, "bead")
    log.info("Bead %02d scored: max exceedance=%.3f, fraction marked=%.4f",
             bead_idx, signal.max(), (signal > SIGNAL_THRESHOLD).mean())
    return bead_idx, raw, steps


def _score_centroid_array(
    centroid_positions: np.ndarray,
    topo: MoleculeTopology,
    bundle: ReferenceBundle,
    force_recompute: bool,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Score centroid from an in-memory position array (HDF5 path).

    Returns (raw score per test, steps, descriptors)."""
    cache_path = _cache_file(cache_dir, "centroid", bundle, topo)

    cached = _read_cache(cache_path, bundle, force_recompute)
    if cached is not None:
        steps, descs = cached
    else:
        log.info("Computing descriptors for centroid …")
        descs = _features(bundle, topo, centroid_positions)
        steps = np.arange(len(centroid_positions), dtype=np.int64)
        np.savez_compressed(cache_path, steps=steps, descriptors=descs)
        log.info("Cached %d frames → %s", len(steps), cache_path.name)

    return _test_scores(bundle, descs), steps, descs


def _score_centroid(
    centroid_xyz: str,
    topo: MoleculeTopology,
    spec: MoleculeSpec,
    bundle: ReferenceBundle,
    chunk_size: int,
    stride: int,
    force_recompute: bool,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Returns (raw score per test, steps, descriptors)."""
    cache_path = _cache_file(cache_dir, "centroid", bundle, topo)
    steps, descs = _load_or_compute_descriptors(
        bead_path=centroid_xyz,
        cache_path=cache_path,
        bundle=bundle,
        topo=topo,
        spec=spec,
        chunk_size=chunk_size,
        stride=stride,
        force_recompute=force_recompute,
        bead_name="centroid",
    )
    return _test_scores(bundle, descs), steps, descs


# ---------------------------------------------------------------------------
# Chemistry check (chunked, keeps only per-frame flag summary)
# ---------------------------------------------------------------------------

def _chemistry_check_chunked(
    bead_path: str,
    topo: MoleculeTopology,
    spec: MoleculeSpec,
    steps: np.ndarray,
    bond_break_cutoff: float,
    close_contact_cutoff: float,
    chunk_size: int,
    stride: int,
) -> list[dict]:
    rows: list[dict] = []
    for steps_chunk, pos_chunk in iter_bead_positions(
        bead_path, chunk_size=chunk_size, stride=stride, spec=spec
    ):
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
    centroid_threshold: float,
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
    ax.set_ylabel("exceedance (score ÷ threshold)")
    ax.legend(fontsize=7)
    ax.set_title(f"{run_name} — geometric track")

    ax = axes[1]
    ax.plot(time_ps, agg_df["bead_frac_ood"], lw=0.5, color="darkorange", label="bead frac OOD")
    ax.set_ylabel("Fraction OOD")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(time_ps, agg_df["centroid_score"], lw=0.5, color="forestgreen", label="centroid score")
    if centroid_threshold == threshold:
        ax.axhline(threshold, color="red", ls="--", lw=1)
    else:
        ax.axhline(
            centroid_threshold, color="purple", ls="--", lw=1,
            label=f"centroid threshold={centroid_threshold:.2f}",
        )
    _mark_onset(ax, time_ps, onset.centroid_anomaly_frame, "geo centroid onset", "purple")
    ax.set_ylabel("exceedance (score ÷ threshold)")
    ax.legend(fontsize=7)
    if not has_emb:
        ax.set_xlabel("Time (ps)")

    # ── Embedding track (when present) ───────────────────────────────────────
    if has_emb:
        ax = axes[3]
        ax.plot(time_ps, agg_df["emb_bead_max"], lw=0.5, label="emb bead max", color="royalblue")
        ax.plot(time_ps, agg_df["emb_bead_p95"], lw=0.5, label="emb bead p95", color="skyblue")
        _mark_onset(ax, time_ps, onset.embedding_persistent_bead_onset_frame, "emb bead onset", "orange")
        ax.set_ylabel("Emb. T² (max over atoms)")
        ax.legend(fontsize=7)
        ax.set_title(f"{run_name} — embedding track")

        ax = axes[4]
        ax.plot(time_ps, agg_df["emb_bead_frac_ood"], lw=0.5, color="darkorange", label="emb bead frac OOD")
        ax.set_ylabel("Fraction OOD")
        ax.legend(fontsize=7)

        ax = axes[5]
        ax.plot(time_ps, agg_df["emb_centroid_score"], lw=0.5, color="teal", label="emb centroid score")
        _mark_onset(ax, time_ps, onset.embedding_centroid_onset_frame, "emb centroid onset", "purple")
        ax.set_ylabel("Emb. T² (max over atoms)")
        ax.set_xlabel("Time (ps)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(plots_dir / "score_vs_time.png", dpi=150)
    plt.close(fig)
    log.info("Saved score_vs_time.png")


def _mark_onset(ax, time_ps, frame_idx, label, color):
    if frame_idx is not None and frame_idx < len(time_ps):
        ax.axvline(time_ps[frame_idx], color=color, ls=":", lw=1.2, label=f"{label} onset")
