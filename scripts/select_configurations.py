"""Select configurations from a trajectory based on descriptor-space distance.

Selects the N frames farthest from a reference configuration in PCA-reduced
internal-coordinate descriptor space (geometric OOD track), subject to a
maximum Euclidean distance radius. The DescriptorPipeline (StandardScaler +
PCA) is fit on the trajectory itself, so the descriptor space reflects the
trajectory's own variance, not the training-set distribution.

Output: a single extxyz file with the reference frame first, followed by the
N selected frames in descending order of descriptor-space distance. Each frame
carries source_frame, source_step, and descriptor_distance in the comment line.
The reference frame carries source=reference.

Usage
-----
python scripts/select_configurations.py \\
    --reference initial.xyz \\
    --trajectory aspirin.xc.xyz \\
    --radius 5.0 \\
    --n-configs 50 \\
    --output selected.xyz \\
    [--pimd] \\
    [--pca-variance 0.95]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from detectana.descriptors import DescriptorPipeline, compute_descriptor_batch
from detectana.io import load_single_frame, load_trajectory_frames
from detectana.topology import build_topology

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Select N configurations farthest from a reference frame in "
            "PCA-reduced internal-coordinate descriptor space, within a radius."
        )
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to reference configuration XYZ file (single frame).",
    )
    parser.add_argument(
        "--trajectory", required=True,
        help="Path to MD or PIMD centroid XYZ trajectory file.",
    )
    parser.add_argument(
        "--radius", type=float, required=True,
        help="Maximum Euclidean distance in descriptor space.",
    )
    parser.add_argument(
        "--n-configs", type=int, required=True,
        help="Number of configurations to select from the trajectory.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output extxyz file path.",
    )
    parser.add_argument(
        "--pimd", action="store_true",
        help="Trajectory is in iPI format (PIMD centroid file, e.g. aspirin.xc.xyz).",
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="Fraction of variance retained by PCA (default: 0.95).",
    )
    args = parser.parse_args()

    # ── Reference configuration ───────────────────────────────────────────────
    log.info("Loading reference configuration from %s", args.reference)
    ref_atoms = load_single_frame(args.reference)

    # ── Topology (built from reference) ──────────────────────────────────────
    topo = build_topology(args.reference)

    # ── Trajectory ────────────────────────────────────────────────────────────
    log.info("Loading trajectory from %s (--pimd=%s)", args.trajectory, args.pimd)
    atoms_list, steps = load_trajectory_frames(args.trajectory, pimd=args.pimd)
    n_frames = len(atoms_list)
    log.info("Loaded %d trajectory frames", n_frames)

    # ── Internal-coordinate descriptors ───────────────────────────────────────
    log.info("Computing internal-coordinate descriptors...")
    traj_positions = np.array(
        [a.get_positions() for a in atoms_list], dtype=np.float64
    )  # (n_frames, n_atoms, 3)
    traj_descriptors = compute_descriptor_batch(traj_positions, topo)
    # (n_frames, n_features)

    # ── Fit DescriptorPipeline on trajectory ──────────────────────────────────
    log.info("Fitting DescriptorPipeline (pca_variance=%.2f)...", args.pca_variance)
    pipeline = DescriptorPipeline(pca_variance=args.pca_variance)
    pipeline.fit(traj_descriptors)
    log.info("PCA retained %d components", pipeline.n_components)

    # ── Project into descriptor space ─────────────────────────────────────────
    traj_pca = pipeline.transform(traj_descriptors)  # (n_frames, n_components)

    ref_positions = ref_atoms.get_positions()[np.newaxis, :]  # (1, n_atoms, 3)
    ref_descriptor = compute_descriptor_batch(ref_positions, topo)  # (1, n_features)
    ref_pca = pipeline.transform(ref_descriptor)  # (1, n_components)

    # ── Euclidean distances from reference ────────────────────────────────────
    distances = np.linalg.norm(traj_pca - ref_pca, axis=1)  # (n_frames,)

    # ── Filter by radius ──────────────────────────────────────────────────────
    within_indices = np.where(distances <= args.radius)[0]
    n_within = len(within_indices)

    if n_within == 0:
        log.error(
            "No trajectory frames found within radius %.4f. "
            "Try increasing --radius.",
            args.radius,
        )
        return

    n_select = min(args.n_configs, n_within)
    if n_within < args.n_configs:
        warnings.warn(
            f"Only {n_within} frames within radius {args.radius}; "
            f"requested {args.n_configs}. Writing {n_within} selected frames.",
            stacklevel=2,
        )

    # ── Select N farthest within radius ───────────────────────────────────────
    order = np.argsort(distances[within_indices])[::-1]
    selected_indices = within_indices[order[:n_select]]

    # ── Build output frames (positions only) ──────────────────────────────────
    ref_out = Atoms(
        symbols=ref_atoms.get_chemical_symbols(),
        positions=ref_atoms.get_positions(),
    )
    ref_out.info = {"source": "reference", "descriptor_distance": 0.0}

    output_frames: list[Atoms] = [ref_out]
    for idx in selected_indices:
        out = Atoms(
            symbols=atoms_list[idx].get_chemical_symbols(),
            positions=atoms_list[idx].get_positions(),
        )
        out.info = {
            "source_frame": int(idx),
            "source_step": int(steps[idx]),
            "descriptor_distance": float(distances[idx]),
        }
        output_frames.append(out)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), output_frames, format="extxyz")
    log.info(
        "Wrote %d frames (%d selected + 1 reference) to %s",
        len(output_frames),
        n_select,
        out_path,
    )


if __name__ == "__main__":
    main()
