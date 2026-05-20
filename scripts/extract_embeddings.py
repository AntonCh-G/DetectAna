"""Extract MlffModel inv_features and write per-bead HDF5 embedding files.

This script is intentionally kept outside the detectana package so that
detectana itself has no dependency on mlff_torch or PyTorch.

Run on a GPU node before executing run_pipeline.py.  The HDF5 files it
produces are consumed by detectana when ``embedding.enabled: true`` in the
YAML config.

Output layout (mirrors bead_glob / centroid_xyz paths in the config):

    <output_dir>/
        ref_train_embeddings.h5          # reference training embeddings
        ref_valid_embeddings.h5          # reference validation embeddings
        <run_name>/
            aspirin.emb_00.h5            # bead 00 embeddings
            ...
            aspirin.emb_15.h5            # bead 15 embeddings
            aspirin.emb_xc.h5            # centroid embeddings

Each HDF5 file contains:
    inv_features : (n_frames, n_atoms, n_features)  float32
    steps        : (n_frames,)                       int64

Usage
-----
python scripts/extract_embeddings.py \\
    --config configs/default_pbe0_pimd_1.yaml \\
    --checkpoint /path/to/mlff_pbe0.ckpt \\
    --output-dir outputs/embeddings \\
    [--stride 10] \\
    [--frame-start 0] \\
    [--frame-end 100000] \\
    [--batch-size 64] \\
    [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"Step:\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, device: str):
    """Load a MlffModel model from a checkpoint file."""
    import torch
    from mlff_torch.modules.models import MlffModel, MlffModelLR

    state = torch.load(checkpoint, map_location=device)

    # Support both raw state-dicts and Lightning-style checkpoints
    if "hyper_parameters" in state:
        cfg = state["hyper_parameters"]
        # Instantiate the correct model class
        model_cls = MlffModelLR if cfg.get("model_type", "mlff") == "mlff_lr" else MlffModel
        model = model_cls(**{k: v for k, v in cfg.items() if k != "model_type"})
        model.load_state_dict(state["state_dict"])
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {checkpoint}. "
            "Expected a Lightning checkpoint with 'hyper_parameters' key."
        )

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _atoms_to_batch(atoms_list, device: str):
    """Convert a list of ASE Atoms to a MlffModel-compatible batch dict."""
    import torch
    from mlff_torch.data.utils import atoms_to_graph

    graphs = [atoms_to_graph(a) for a in atoms_list]
    # Collate into a single batched dict expected by the model
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(graphs).to(device)
    return batch


def extract_embeddings_from_frames(
    model,
    atoms_list,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run MlffModel forward pass on a list of ASE Atoms and return inv_features.

    Returns
    -------
    embeddings : (n_frames, n_atoms, n_features)  float32
    """
    import torch

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(atoms_list), batch_size):
        batch_atoms = atoms_list[start : start + batch_size]
        batch = _atoms_to_batch(batch_atoms, device)

        with torch.no_grad():
            output = model(batch, return_descriptors=True, compute_force=False)

        inv_feat = output["inv_features"]  # (n_batch_atoms, n_features)
        n_atoms = batch_atoms[0].get_positions().shape[0]
        n_frames = len(batch_atoms)
        inv_feat = inv_feat.reshape(n_frames, n_atoms, -1).cpu().numpy().astype(np.float32)
        all_embeddings.append(inv_feat)

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# XYZ loading with stride + frame range
# ---------------------------------------------------------------------------

def load_xyz_frames(xyz_path: str, stride: int, frame_start: int | None, frame_end: int | None):
    """Load ASE Atoms from an XYZ file with stride and optional frame range.

    Returns (atoms_list, steps).
    """
    from ase.io import iread

    atoms_list = []
    steps = []
    frame_idx = 0

    for atoms in iread(xyz_path, format="extxyz"):
        if frame_start is not None and frame_idx < frame_start:
            frame_idx += 1
            continue
        if frame_end is not None and frame_idx >= frame_end:
            break
        if (frame_idx - (frame_start or 0)) % stride == 0:
            atoms_list.append(atoms)
            # Parse step from comment if available, else use frame index
            comment = atoms.info.get("comment", "")
            m = _STEP_RE.search(str(atoms.info))
            steps.append(int(m.group(1)) if m else frame_idx)
        frame_idx += 1

    return atoms_list, np.array(steps, dtype=np.int64)


def load_ipi_xyz_frames(xyz_path: str, stride: int, frame_start: int | None, frame_end: int | None):
    """Load ASE Atoms from an iPI-format XYZ file (bead/centroid trajectories).

    Returns (atoms_list, steps).
    """
    from ase.io import iread

    atoms_list = []
    steps = []
    frame_idx = 0

    for atoms in iread(xyz_path, format="xyz"):
        if frame_start is not None and frame_idx < frame_start:
            frame_idx += 1
            continue
        if frame_end is not None and frame_idx >= frame_end:
            break
        if (frame_idx - (frame_start or 0)) % stride == 0:
            atoms_list.append(atoms)
            # iPI stores step in the comment line
            comment = ""
            if hasattr(atoms, "info"):
                comment = str(atoms.info)
            m = _STEP_RE.search(comment)
            steps.append(int(m.group(1)) if m else frame_idx * 50)
        frame_idx += 1

    return atoms_list, np.array(steps, dtype=np.int64)


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def write_h5(path: str | Path, embeddings: np.ndarray, steps: np.ndarray) -> None:
    """Write embeddings and steps to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("inv_features", data=embeddings, compression="gzip", compression_opts=4)
        fh.create_dataset("steps", data=steps)
    log.info("Wrote %s  shape=%s", path, embeddings.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract MlffModel embeddings to HDF5")
    parser.add_argument("--config",      required=True, help="DetectAna YAML config file")
    parser.add_argument("--checkpoint",  required=True, help="MlffModel model checkpoint (.ckpt)")
    parser.add_argument("--output-dir",  required=True, help="Directory for output HDF5 files")
    parser.add_argument("--stride",      type=int, default=10,
                        help="Extract every Nth frame (default: 10)")
    parser.add_argument("--frame-start", type=int, default=None,
                        help="First frame index to include (default: beginning)")
    parser.add_argument("--frame-end",   type=int, default=None,
                        help="Last frame index (exclusive) to include (default: end)")
    parser.add_argument("--batch-size",  type=int, default=64,
                        help="Forward-pass batch size (default: 64)")
    parser.add_argument("--device",      default="cuda",
                        help="PyTorch device string (default: cuda)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading model from %s …", args.checkpoint)
    model = load_model(args.checkpoint, args.device)

    # ── Reference embeddings ──────────────────────────────────────────────────
    for split, key in [("train", "train"), ("valid", "valid")]:
        ref_path = cfg["reference"][key]
        log.info("Processing reference %s: %s", split, ref_path)
        atoms_list, steps = load_xyz_frames(ref_path, stride=1, frame_start=None, frame_end=None)
        embeddings = extract_embeddings_from_frames(model, atoms_list, args.batch_size, args.device)
        write_h5(out_root / f"ref_{split}_embeddings.h5", embeddings, steps)

    # ── Per-run bead + centroid embeddings ────────────────────────────────────
    for run_cfg in cfg["runs"]:
        run_name = run_cfg["name"]
        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        log.info("=== Run: %s ===", run_name)

        bead_files = sorted(glob.glob(run_cfg["bead_glob"]))
        if not bead_files:
            log.warning("No bead files matched: %s — skipping run", run_cfg["bead_glob"])
            continue

        for bead_idx, bead_path in enumerate(bead_files):
            log.info("Bead %02d: %s", bead_idx, bead_path)
            atoms_list, steps = load_ipi_xyz_frames(
                bead_path, args.stride, args.frame_start, args.frame_end
            )
            if not atoms_list:
                log.warning("No frames extracted from bead %02d — skipping", bead_idx)
                continue
            embeddings = extract_embeddings_from_frames(
                model, atoms_list, args.batch_size, args.device
            )
            out_path = run_out / f"aspirin.emb_{bead_idx:02d}.h5"
            write_h5(out_path, embeddings, steps)

        # Centroid
        centroid_path = run_cfg["centroid_xyz"]
        log.info("Centroid: %s", centroid_path)
        atoms_list, steps = load_ipi_xyz_frames(
            centroid_path, args.stride, args.frame_start, args.frame_end
        )
        if atoms_list:
            embeddings = extract_embeddings_from_frames(
                model, atoms_list, args.batch_size, args.device
            )
            write_h5(run_out / "aspirin.emb_xc.h5", embeddings, steps)

    log.info("Done. Update your YAML config with:")
    log.info("  embedding:")
    log.info("    enabled: true")
    log.info("    reference_train_h5: %s/ref_train_embeddings.h5", out_root)
    log.info("    reference_valid_h5: %s/ref_valid_embeddings.h5", out_root)
    log.info("  runs:")
    for run_cfg in cfg["runs"]:
        run_name = run_cfg["name"]
        log.info("    - embedding_glob: %s/%s/aspirin.emb_*.h5", out_root, run_name)
        log.info("      centroid_embedding_h5: %s/%s/aspirin.emb_xc.h5", out_root, run_name)


if __name__ == "__main__":
    main()
