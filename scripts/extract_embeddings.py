"""Extract per-atom force-field embeddings and write per-bead HDF5 files.

This script is intentionally kept outside the detectana package so that
detectana itself has no dependency on any force-field code or on PyTorch.
It is the only file in the repository that touches the model.

Run on a GPU node before executing run_pipeline.py.  The HDF5 files it
produces are consumed by detectana when ``embedding.enabled: true`` in the
YAML config.

The force field is not named here and is not a dependency of this project:
``--model-package`` names the Python package to load it from, so any model
satisfying the contract below works. The MLFF used during development is part
of unpublished work and is deliberately left unnamed.

Adapter contract
----------------
Given ``--model-package PKG``, this script expects:

- ``PKG.modules.models`` to expose the model classes. The class is chosen by
  ``--model-class``, or by a case-insensitive match against the checkpoint's
  ``hyper_parameters["model_type"]``.
- ``PKG.data.utils.atoms_to_graph`` to convert one ASE ``Atoms`` to a graph
  object that ``torch_geometric.data.Batch.from_data_list`` can collate.
- the model to accept ``model(batch, return_descriptors=True,
  compute_force=False)`` and return a mapping containing ``--feature-key``
  (default ``inv_features``): invariant per-atom features of shape
  ``(n_batch_atoms, n_features)``.

Only invariant (rotation-invariant) per-atom features are usable — an
equivariant tensor would make the OOD score orientation-dependent.

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
    --config config/your_run.yaml \\
    --checkpoint path/to/checkpoint.ckpt \\
    --model-package your_mlff_package \\
    --output-dir outputs/embeddings \\
    [--model-class ModelClassName] \\
    [--feature-key inv_features] \\
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

# Key of the invariant per-atom features in the model's output mapping. Also the
# dataset name in the HDF5 files this script writes, which is part of the format
# detectana reads (``io.load_embeddings_h5``) and is not affected by
# ``--feature-key``.
DEFAULT_FEATURE_KEY = "inv_features"
H5_FEATURE_DATASET = "inv_features"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _import_attr(module_path: str, attr: str):
    """Import ``attr`` from ``module_path``, with a message that names the fix."""
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import {module_path}. Check --model-package and that the "
            f"force-field package is installed in this environment.\n{exc}"
        ) from None
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(
            f"{module_path} has no attribute {attr!r}. Available: "
            f"{sorted(n for n in dir(module) if not n.startswith('_'))}"
        ) from None


def _resolve_model_class(models_module: str, model_class: str | None, model_type: str | None):
    """Pick the model class: an explicit name, else a match on ``model_type``.

    ``model_type`` comes from the checkpoint and is matched case-insensitively,
    so a checkpoint recording e.g. ``"mymodel"`` finds the class ``MyModel``.
    """
    import importlib

    if model_class is not None:
        return _import_attr(models_module, model_class)
    if model_type is None:
        raise ValueError(
            f"The checkpoint records no 'model_type', so the class cannot be "
            f"inferred. Pass --model-class with one of the classes in {models_module}."
        )
    module = importlib.import_module(models_module)
    for name in dir(module):
        if name.lower() == str(model_type).lower().replace("_", ""):
            return getattr(module, name)
    raise ValueError(
        f"No class in {models_module} matches the checkpoint's model_type "
        f"{model_type!r}. Pass --model-class explicitly. Available: "
        f"{sorted(n for n in dir(module) if not n.startswith('_'))}"
    )


def load_model(checkpoint: str, device: str, model_package: str, model_class: str | None = None):
    """Load a force-field model from a Lightning-style checkpoint.

    Parameters
    ----------
    model_package : Python package holding the model, e.g. the MLFF codebase.
        ``<model_package>.modules.models`` must expose the model classes.
    model_class : class name to instantiate. Inferred from the checkpoint's
        ``model_type`` when omitted.
    """
    import torch

    state = torch.load(checkpoint, map_location=device)

    if "hyper_parameters" not in state:
        raise ValueError(
            f"Unrecognised checkpoint format in {checkpoint}. "
            "Expected a Lightning checkpoint with 'hyper_parameters' key."
        )

    cfg = state["hyper_parameters"]
    model_cls = _resolve_model_class(
        f"{model_package}.modules.models", model_class, cfg.get("model_type")
    )
    model = model_cls(**{k: v for k, v in cfg.items() if k != "model_type"})
    model.load_state_dict(state["state_dict"])

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _atoms_to_batch(atoms_list, device: str, model_package: str):
    """Convert a list of ASE Atoms into one batch the model can consume."""
    atoms_to_graph = _import_attr(f"{model_package}.data.utils", "atoms_to_graph")

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
    model_package: str,
    feature_key: str = DEFAULT_FEATURE_KEY,
) -> np.ndarray:
    """Run the model forward and return its invariant per-atom features.

    Parameters
    ----------
    feature_key : key of the invariant per-atom features in the model's output
        mapping. Must be rotation-invariant: an equivariant tensor would make
        the OOD score depend on molecular orientation.

    Returns
    -------
    embeddings : (n_frames, n_atoms, n_features)  float32
    """
    import torch

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(atoms_list), batch_size):
        batch_atoms = atoms_list[start : start + batch_size]
        batch = _atoms_to_batch(batch_atoms, device, model_package)

        with torch.no_grad():
            output = model(batch, return_descriptors=True, compute_force=False)

        if feature_key not in output:
            raise KeyError(
                f"The model output has no {feature_key!r}. Pass --feature-key with "
                f"one of: {sorted(output)}"
            )
        inv_feat = output[feature_key]  # (n_batch_atoms, n_features)
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
            # Parse the step out of the frame's metadata if it is there, else use
            # the frame index. The whole info dict is searched rather than the
            # "comment" key alone, because extxyz readers split the comment line
            # into keys and the step may end up under any of them.
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
        fh.create_dataset(
            H5_FEATURE_DATASET, data=embeddings, compression="gzip", compression_opts=4
        )
        fh.create_dataset("steps", data=steps)
    log.info("Wrote %s  shape=%s", path, embeddings.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Extract per-atom force-field embeddings to HDF5"
    )
    parser.add_argument("--config",      required=True, help="DetectAna YAML config file")
    parser.add_argument("--checkpoint",  required=True, help="Model checkpoint (.ckpt)")
    parser.add_argument("--output-dir",  required=True, help="Directory for output HDF5 files")
    parser.add_argument("--model-package", required=True,
                        help="Python package holding the force field (see the adapter "
                             "contract in this file's docstring)")
    parser.add_argument("--model-class",  default=None,
                        help="Model class to instantiate (default: inferred from the "
                             "checkpoint's model_type)")
    parser.add_argument("--feature-key",  default=DEFAULT_FEATURE_KEY,
                        help=f"Key of the invariant per-atom features in the model "
                             f"output (default: {DEFAULT_FEATURE_KEY})")
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

    log.info("Loading model from %s (package %s) …", args.checkpoint, args.model_package)
    model = load_model(args.checkpoint, args.device, args.model_package, args.model_class)

    # ── Reference embeddings ──────────────────────────────────────────────────
    for split, key in [("train", "train"), ("valid", "valid")]:
        ref_path = cfg["reference"][key]
        log.info("Processing reference %s: %s", split, ref_path)
        atoms_list, steps = load_xyz_frames(ref_path, stride=1, frame_start=None, frame_end=None)
        embeddings = extract_embeddings_from_frames(
            model, atoms_list, args.batch_size, args.device,
            args.model_package, args.feature_key,
        )
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
                model, atoms_list, args.batch_size, args.device,
                args.model_package, args.feature_key,
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
                model, atoms_list, args.batch_size, args.device,
                args.model_package, args.feature_key,
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
