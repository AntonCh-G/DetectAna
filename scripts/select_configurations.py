"""Select configurations from a trajectory based on descriptor-space distance.

Two modes, selected by whether --primary-dihedrals is given:

Primary-dihedral mode (recommended for aspirin conformational analysis)
-----------------------------------------------------------------------
Distance is computed in the sin/cos space of the specified dihedral angles only.
No PCA or StandardScaler is needed. The radius has a direct geometric meaning in
the 2N-dimensional sin/cos torus (N = number of specified dihedrals).

    python scripts/select_configurations.py \\
        --reference initial.xyz \\
        --trajectory aspirin.xc.xyz \\
        --radius 0.5 \\
        --n-configs 50 \\
        --output selected.xyz \\
        --pimd \\
        --primary-dihedrals 6 5 10 7 \\
        --primary-dihedrals 5 6 12 11

Full-descriptor mode (default)
------------------------------
Distance is computed in PCA-reduced internal-coordinate space (bonds, angles,
all dihedrals, ring planarity). The DescriptorPipeline is fit on the trajectory.

    python scripts/select_configurations.py \\
        --reference initial.xyz \\
        --trajectory aspirin.xc.xyz \\
        --radius 5.0 \\
        --n-configs 50 \\
        --output selected.xyz \\
        --pimd

In both modes the output extxyz carries source_frame, source_step, and
descriptor_distance in the ASE comment line. The reference frame is written
first with source=reference.
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


# ---------------------------------------------------------------------------
# Dihedral sin/cos helper (used only in primary-dihedral mode)
# ---------------------------------------------------------------------------

def _dihedral_sincos_batch(
    positions: np.ndarray,
    dihedral_list: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Compute sin/cos pairs for a set of dihedrals across all frames.

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3)
    dihedral_list : list of (i, j, k, l) atom-index tuples

    Returns
    -------
    features : (n_frames, 2 * len(dihedral_list))
        Columns are [sin_d0, cos_d0, sin_d1, cos_d1, ...].
    """
    parts = []
    for i, j, k, l in dihedral_list:
        p0, p1, p2, p3 = positions[:, i], positions[:, j], positions[:, k], positions[:, l]
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_hat = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
        v = b0 - np.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
        w = b2 - np.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat
        x = np.sum(v * w, axis=-1)
        y = np.sum(np.cross(b1_hat, v) * w, axis=-1)
        angle = np.arctan2(y, x)
        parts.append(np.stack([np.sin(angle), np.cos(angle)], axis=-1))
    return np.concatenate(parts, axis=-1)  # (n_frames, 2*N)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Select N configurations farthest from a reference frame within a "
            "distance radius. Distance is measured either in the sin/cos space "
            "of specified dihedrals (--primary-dihedrals) or in PCA-reduced "
            "full internal-coordinate descriptor space (default)."
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
        help="Maximum Euclidean distance in the chosen descriptor space.",
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
        "--primary-dihedrals",
        nargs=4, type=int, action="append",
        metavar=("I", "J", "K", "L"),
        help=(
            "Use only these dihedral angle(s) for distance computation instead of "
            "the full PCA descriptor. Specify 4 atom indices (0-based). Repeat the "
            "flag to include additional dihedrals. "
            "Example: --primary-dihedrals 6 5 10 7 --primary-dihedrals 5 6 12 11"
        ),
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="(Full-descriptor mode only) Fraction of variance retained by PCA (default: 0.95).",
    )
    args = parser.parse_args()

    primary_dihedrals: list[tuple[int, int, int, int]] | None = (
        [tuple(d) for d in args.primary_dihedrals]  # type: ignore[misc]
        if args.primary_dihedrals else None
    )

    # ── Reference configuration ───────────────────────────────────────────────
    log.info("Loading reference configuration from %s", args.reference)
    ref_atoms = load_single_frame(args.reference)
    ref_positions = ref_atoms.get_positions()[np.newaxis, :]  # (1, n_atoms, 3)

    # ── Trajectory ────────────────────────────────────────────────────────────
    log.info("Loading trajectory from %s (--pimd=%s)", args.trajectory, args.pimd)
    atoms_list, steps = load_trajectory_frames(args.trajectory, pimd=args.pimd)
    n_frames = len(atoms_list)
    log.info("Loaded %d trajectory frames", n_frames)

    traj_positions = np.array(
        [a.get_positions() for a in atoms_list], dtype=np.float64
    )  # (n_frames, n_atoms, 3)

    # ── Compute features and distances ────────────────────────────────────────
    if primary_dihedrals:
        log.info(
            "Primary-dihedral mode: %d dihedral(s) → %d sin/cos features",
            len(primary_dihedrals), 2 * len(primary_dihedrals),
        )
        for atoms in primary_dihedrals:
            log.info("  dihedral atoms: %s", list(atoms))

        traj_features = _dihedral_sincos_batch(traj_positions, primary_dihedrals)
        ref_features = _dihedral_sincos_batch(ref_positions, primary_dihedrals)
        distances = np.linalg.norm(traj_features - ref_features, axis=1)

    else:
        log.info("Full-descriptor mode: building topology and computing descriptors...")
        topo = build_topology(args.reference)
        traj_descriptors = compute_descriptor_batch(traj_positions, topo)
        ref_descriptor = compute_descriptor_batch(ref_positions, topo)

        log.info("Fitting DescriptorPipeline (pca_variance=%.2f)...", args.pca_variance)
        pipeline = DescriptorPipeline(pca_variance=args.pca_variance)
        pipeline.fit(traj_descriptors)
        log.info("PCA retained %d components", pipeline.n_components)

        traj_features = pipeline.transform(traj_descriptors)
        ref_features = pipeline.transform(ref_descriptor)
        distances = np.linalg.norm(traj_features - ref_features, axis=1)

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
