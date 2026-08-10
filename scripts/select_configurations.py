"""Select configurations from a trajectory based on descriptor-space distance.

Two modes, selected by whether --primary-dihedrals is given:

Primary-dihedral mode (recommended for conformational analysis)
----------------------------------------------------------------
--radius constrains frames in the sin/cos space of the specified dihedrals.
Accepts one value (isotropic, circular constraint shared across all primary
dihedrals) or one value per --primary-dihedrals (anisotropic, elliptic
constraint).  A frame passes when sum((dᵢ/Rᵢ)²) ≤ 1, where dᵢ is the chord
distance in sin/cos space for dihedral i and Rᵢ is its radius.  The output
descriptor_distance is sqrt(sum((dᵢ/Rᵢ)²)); values ≤ 1 lie inside the
ellipse.

Among frames that pass, N configurations are selected to maximise diversity in
the *complementary* raw internal-coordinate space (all bonds, angles, other
dihedrals, ring planarity if present — the primary dihedral sin/cos columns
are excluded).

In short: keep the specified dihedral(s) fixed near the reference value, then
pick diverse structures in the remaining conformational degrees of freedom.

Isotropic (circular) example:
    python scripts/select_configurations.py \\
        --reference initial.xyz \\
        --trajectory aspirin.xc.xyz \\
        --radius 0.2 \\
        --n-configs 50 \\
        --output selected.xyz \\
        --pimd \\
        --primary-dihedrals 5 6 12 11

Anisotropic (elliptic) example — tight carbonyl, loose ester:
    python scripts/select_configurations.py \\
        --reference initial.xyz \\
        --trajectory aspirin.xc.xyz \\
        --radius 0.1 0.4 \\
        --n-configs 50 \\
        --output selected.xyz \\
        --pimd \\
        --primary-dihedrals 6 5 10 7 \\
        --primary-dihedrals 5 6 12 11

Full-descriptor mode (default)
------------------------------
Distance is computed in PCA-reduced internal-coordinate space (bonds, angles,
all dihedrals, and ring planarity when the molecule has a ring). The
DescriptorPipeline is fit on the trajectory.

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
from detectana.io import (
    MoleculeSpec,
    load_single_frame,
    load_trajectory_positions,
)
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


def _fps_select(
    features: np.ndarray,
    ref_features: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """Farthest-Point Sampling starting from the reference.

    Greedily picks the frame that maximises the minimum distance to any
    already-selected point (starting with the reference).  Guarantees
    diversity among selected frames while respecting the pre-filter.

    Parameters
    ----------
    features : (n_candidates, n_features) — already filtered to within radius
    ref_features : (1, n_features)
    n_select : number of frames to select

    Returns
    -------
    indices : (n_select,) — indices into ``features``
    """
    min_dists = np.linalg.norm(features - ref_features, axis=1).copy()
    selected: list[int] = []
    for _ in range(n_select):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        new_dists = np.linalg.norm(features - features[idx], axis=1)
        np.minimum(min_dists, new_dists, out=min_dists)
        min_dists[idx] = -1.0  # exclude from future picks
    return np.array(selected, dtype=np.intp)


def _complementary_feature_mask(
    topo,
    primary_dihedrals: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Boolean mask over full internal-coordinate features; False = primary dihedral column.

    The full descriptor layout is:
        [bond_lengths | angles | sin_d0..sin_dN | cos_d0..cos_dN | planarity]

    For each primary dihedral (i,j,k,l) the canonical form is located in
    topo.dihedrals; the corresponding sin and cos columns are masked out.

    Parameters
    ----------
    topo : MoleculeTopology
    primary_dihedrals : list of (i, j, k, l) tuples

    Returns
    -------
    mask : (n_features,) bool — True = keep in complementary space
    """
    n_bonds = len(topo.bonds)
    n_angles = len(topo.angles)
    n_dih = len(topo.dihedrals)
    mask = np.ones(topo.n_features, dtype=bool)

    for quad in primary_dihedrals:
        i, j, k, l = quad
        canonical = min((i, j, k, l), (l, k, j, i))
        try:
            dih_idx = list(topo.dihedrals).index(canonical)
        except ValueError:
            raise ValueError(
                f"Primary dihedral {quad} (canonical {canonical}) not found in "
                f"topology. Check 0-based atom indices."
            )
        # sin column: offset n_bonds + n_angles + dih_idx
        # cos column: offset n_bonds + n_angles + n_dih + dih_idx
        mask[n_bonds + n_angles + dih_idx] = False
        mask[n_bonds + n_angles + n_dih + dih_idx] = False

    return mask


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
        "--radius", type=float, nargs="+", required=True,
        help=(
            "Proximity radius in the chosen descriptor space. Accepts one value "
            "(isotropic, circular constraint) or one value per --primary-dihedrals "
            "(anisotropic, elliptic constraint). Multiple values require "
            "--primary-dihedrals."
        ),
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
    radii: list[float] = args.radius

    # ── Validate --radius / --primary-dihedrals combination ──────────────────
    if len(radii) > 1 and not primary_dihedrals:
        parser.error("Multiple --radius values require --primary-dihedrals.")
    if primary_dihedrals and len(radii) not in (1, len(primary_dihedrals)):
        parser.error(
            f"--radius accepts 1 value (isotropic) or {len(primary_dihedrals)} values "
            f"(one per --primary-dihedrals); got {len(radii)}."
        )

    # ── Reference configuration ───────────────────────────────────────────────
    log.info("Loading reference configuration from %s", args.reference)
    ref_atoms = load_single_frame(args.reference)
    ref_positions = ref_atoms.get_positions()[np.newaxis, :]  # (1, n_atoms, 3)
    # The reference frame defines the molecule; the trajectory must match it.
    spec = MoleculeSpec.from_atoms(ref_atoms)
    log.info("Molecule: %d atoms (%s)", spec.n_atoms, "".join(spec.atom_types))

    # ── Trajectory ────────────────────────────────────────────────────────────
    log.info("Loading trajectory from %s (--pimd=%s)", args.trajectory, args.pimd)
    traj_positions, steps = load_trajectory_positions(
        args.trajectory, pimd=args.pimd, spec=spec
    )
    n_frames = len(traj_positions)
    log.info("Loaded %d trajectory frames", n_frames)

    # ── Compute features and distances ────────────────────────────────────────
    if primary_dihedrals:
        log.info(
            "Primary-dihedral mode: %d dihedral(s) as proximity constraint",
            len(primary_dihedrals),
        )
        for d in primary_dihedrals:
            log.info("  dihedral atoms: %s", list(d))

        # Per-dihedral chord distances in sin/cos space: (n_frames, n_primary)
        traj_dih = _dihedral_sincos_batch(traj_positions, primary_dihedrals)
        ref_dih = _dihedral_sincos_batch(ref_positions, primary_dihedrals)
        d_per_dih = np.sqrt(
            (traj_dih[:, 0::2] - ref_dih[:, 0::2]) ** 2 +
            (traj_dih[:, 1::2] - ref_dih[:, 1::2]) ** 2
        )
        n_prim = len(primary_dihedrals)
        radii_arr = np.array(radii if len(radii) > 1 else radii * n_prim)
        # Normalized ellipse distance: sqrt(sum((dᵢ/Rᵢ)²)); ≤ 1 = inside ellipse
        distances = np.sqrt(np.sum((d_per_dih / radii_arr) ** 2, axis=1))

        # Filter first — cheap dihedral distances already computed
        within_indices = np.where(distances <= 1.0)[0]
        n_within = len(within_indices)
        log.info(
            "Ellipse filter: %d / %d frames within constraint (radii=%s)",
            n_within, len(distances), radii,
        )
        if n_within == 0:
            log.error(
                "No trajectory frames found within ellipse constraint (radii=%s). "
                "Try increasing --radius.",
                radii,
            )
            return

        # Full internal-coordinate descriptor minus primary dihedral columns
        # — computed only on the filtered subset, not all 1M frames
        topo = build_topology(args.reference)
        traj_full = compute_descriptor_batch(traj_positions[within_indices], topo)
        ref_full = compute_descriptor_batch(ref_positions, topo)
        comp_mask = _complementary_feature_mask(topo, primary_dihedrals)
        comp_features = traj_full[:, comp_mask]   # (n_within, n_comp)
        ref_comp = ref_full[:, comp_mask]          # (1, n_comp)
        log.info(
            "Complementary descriptor: %d features (removed %d primary dihedral columns)",
            comp_mask.sum(), (~comp_mask).sum(),
        )

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

    # ── Filter by radius (full-descriptor mode only; primary-dihedral filtered above) ──
    if not primary_dihedrals:
        within_indices = np.where(distances <= radii[0])[0]
        n_within = len(within_indices)

        if n_within == 0:
            log.error(
                "No trajectory frames found within radius %.4f. "
                "Try increasing --radius.",
                radii[0],
            )
            return

    n_select = min(args.n_configs, n_within)
    if n_within < args.n_configs:
        warnings.warn(
            f"Only {n_within} frames within constraint (radii={radii}); "
            f"requested {args.n_configs}. Writing {n_within} selected frames.",
            stacklevel=2,
        )

    # ── Select N diverse frames via Farthest-Point Sampling ───────────────────
    # Primary-dihedral mode: diversity in complementary space (constraint = dihedral).
    #   comp_features already indexed to within_indices — pass directly.
    # Full-descriptor mode: diversity in PCA-reduced full space.
    if primary_dihedrals:
        fps_local = _fps_select(comp_features, ref_comp, n_select)
    else:
        fps_local = _fps_select(traj_features[within_indices], ref_features, n_select)
    selected_indices = within_indices[fps_local]

    # ── Build output frames (positions only) ──────────────────────────────────
    ref_out = Atoms(
        symbols=ref_atoms.get_chemical_symbols(),
        positions=ref_atoms.get_positions(),
    )
    ref_out.info = {"source": "reference", "descriptor_distance": 0.0}

    output_frames: list[Atoms] = [ref_out]
    for idx in selected_indices:
        out = Atoms(
            symbols=list(spec.atom_types),
            positions=traj_positions[idx],
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
