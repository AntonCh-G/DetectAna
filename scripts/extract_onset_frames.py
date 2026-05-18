#!/usr/bin/env python3
"""Extract frames around anomaly onset from the first-anomaly bead.

Reads onset_table.csv produced by run_pipeline.py to locate the onset frame
and the bead that first exceeded the OOD threshold, then writes N frames
before and M frames after that onset frame into a single extended-XYZ file
containing both positions (Å) and forces (eV/Å, converted from Hartree/Bohr).

The force file is derived from the position file by replacing ``pos_`` with
``for_`` in the filename (iPI convention: aspirin.pos_NN.xyz → aspirin.for_NN.xyz).

Comment lines carry iPI metadata (Step, Bead, Lattice) alongside the
extxyz Properties tag so the file is parseable by ASE / OVITO while
preserving full frame provenance.

Usage
-----
    python scripts/extract_onset_frames.py \\
        --config config/default.yaml \\
        --run ccsd_naive \\
        --n-before 100 \\
        --n-after 100
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectana.aggregator import aggregate_bead_scores
from detectana.onset import detect_onset
from detectana.xyz_reader import load_or_build_index


# NIST 2018 CODATA: 1 Hartree/Bohr = 51.4221 eV/Å
_HARTREE_BOHR_TO_EV_ANG: float = 51.4221

_RE_CELL = re.compile(
    rb"CELL\(abcABC\):\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)"
    rb"\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)"
)
_RE_STEP = re.compile(rb"Step:\s*(\d+)")
_RE_BEAD = re.compile(rb"Bead:\s*(\d+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_force_path(pos_path: str | Path) -> Path:
    """Derive aspirin.for_NN.xyz from aspirin.pos_NN.xyz (iPI convention)."""
    p = Path(pos_path)
    name = p.name.replace("pos_", "for_", 1)
    if name == p.name:
        raise ValueError(
            f"Cannot derive force file from '{p.name}': expected 'pos_' in filename."
        )
    force_path = p.parent / name
    if not force_path.exists():
        raise FileNotFoundError(
            f"Force file not found: {force_path}\n"
            f"  (derived from position file: {pos_path})"
        )
    return force_path


def _cell_abc_to_lattice_str(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> str:
    """Convert (a, b, c, alpha, beta, gamma) to extxyz Lattice= row-major string."""
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)
    ax = a
    bx = b * math.cos(ga)
    by = b * math.sin(ga)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    return (
        f"{ax:.6f} 0.000000 0.000000 "
        f"{bx:.6f} {by:.6f} 0.000000 "
        f"{cx:.6f} {cy:.6f} {cz:.6f}"
    )


def _write_extxyz_frame(
    dst,
    pos_atom_lines: list[bytes],
    for_atom_lines: list[bytes],
    comment: bytes,
) -> None:
    """Write one extxyz frame merging positions (Å) and forces (eV/Å)."""
    n_atoms = len(pos_atom_lines)

    # Build comment key=value pairs in extxyz order.
    parts: list[str] = []

    m = _RE_CELL.search(comment)
    if m:
        abc = tuple(float(m.group(i)) for i in range(1, 7))
        lattice = _cell_abc_to_lattice_str(*abc)
        parts.append(f'Lattice="{lattice}"')

    parts.append("Properties=species:S:1:pos:R:3:forces:R:3")
    parts.append("forces_unit=eV/Ang")

    m = _RE_STEP.search(comment)
    if m:
        parts.append(f"Step={m.group(1).decode()}")
    m = _RE_BEAD.search(comment)
    if m:
        parts.append(f"Bead={m.group(1).decode()}")

    dst.write(f"{n_atoms}\n".encode())
    dst.write((" ".join(parts) + "\n").encode())

    for pline, fline in zip(pos_atom_lines, for_atom_lines):
        pp = pline.split()
        fp = fline.split()
        sym = pp[0].decode()
        px, py, pz = float(pp[1]), float(pp[2]), float(pp[3])
        fx = float(fp[1]) * _HARTREE_BOHR_TO_EV_ANG
        fy = float(fp[2]) * _HARTREE_BOHR_TO_EV_ANG
        fz = float(fp[3]) * _HARTREE_BOHR_TO_EV_ANG
        dst.write(
            f"{sym}  {px:.10f}  {py:.10f}  {pz:.10f}"
            f"  {fx:.10f}  {fy:.10f}  {fz:.10f}\n".encode()
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames around anomaly onset from the first-anomaly bead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to config YAML.")
    p.add_argument("--run", default=None,
                   help="Run name. Defaults to first run in config.")
    p.add_argument("--n-before", type=int, default=100, metavar="N",
                   help="Frames to extract before onset frame.")
    p.add_argument("--n-after", type=int, default=100, metavar="M",
                   help="Frames to extract after onset frame.")
    p.add_argument("--onset-type", default="persistent",
                   choices=["persistent", "first"],
                   help=(
                       "Which onset frame to centre on. "
                       "'persistent' = first window where bead_frac_ood > threshold (canonical onset). "
                       "'first' = first single frame any bead exceeded threshold."
                   ))
    p.add_argument("--threshold", type=float, default=None, metavar="SCORE",
                   help=(
                       "Override the Mahalanobis distance threshold (e.g. 15.0). "
                       "Read the value directly from the score_vs_time.png plot. "
                       "Re-detects onset from saved bead_scores.npy. "
                       "If omitted, uses onset frame from onset_table.csv as-is."
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # ── Resolve run ───────────────────────────────────────────────────────────
    run_name = args.run or cfg["runs"][0]["name"]
    run_cfg = next((r for r in cfg["runs"] if r["name"] == run_name), None)
    if run_cfg is None:
        sys.exit(
            f"Run '{run_name}' not found in config. "
            f"Available: {[r['name'] for r in cfg['runs']]}"
        )

    # ── Load onset table ──────────────────────────────────────────────────────
    out_root = Path(cfg["io"]["output_dir"])
    run_out = out_root / run_name
    onset_csv = run_out / "onset_table.csv"
    if not onset_csv.exists():
        sys.exit(f"onset_table.csv not found at {onset_csv}. Run run_pipeline.py first.")

    onset_df = pd.read_csv(onset_csv)
    rows = onset_df[onset_df["run"] == run_name]
    if rows.empty:
        sys.exit(f"Run '{run_name}' not found in {onset_csv}.")
    row = rows.iloc[0]

    frame_col = (
        "persistent_bead_anomaly_frame"
        if args.onset_type == "persistent"
        else "first_bead_anomaly_frame"
    )
    step_col = (
        "persistent_bead_anomaly_step"
        if args.onset_type == "persistent"
        else "first_bead_anomaly_step"
    )

    onset_frame = row[frame_col]
    bead_idx = row["first_anomaly_bead_idx"]

    if pd.isna(onset_frame) or pd.isna(bead_idx):
        sys.exit(
            f"No anomaly detected for run '{run_name}' "
            f"(onset_type='{args.onset_type}'). Nothing to extract."
        )

    onset_frame = int(onset_frame)
    bead_idx = int(bead_idx)

    # ── Optional: re-detect onset at a different Mahalanobis threshold ────────
    if args.threshold is not None:
        bead_scores_path = run_out / "bead_scores.npy"
        centroid_scores_path = run_out / "centroid_scores.npy"
        agg_csv_path = run_out / "frame_aggregate.csv"

        for p in [bead_scores_path, centroid_scores_path, agg_csv_path]:
            if not p.exists():
                sys.exit(f"Required file not found: {p}. Run run_pipeline.py first.")

        new_threshold = args.threshold
        print(f"Threshold override: {new_threshold:.4f}")

        bead_scores = np.load(bead_scores_path)       # (n_beads, n_frames)
        centroid_scores = np.load(centroid_scores_path)
        agg_df_src = pd.read_csv(agg_csv_path)
        steps = agg_df_src["step"].to_numpy()
        frame_time_fs = run_cfg["timestep_fs"] * run_cfg["stride"]

        agg_df_new = aggregate_bead_scores(
            bead_scores=bead_scores,
            centroid_scores=centroid_scores,
            steps=steps,
            threshold=new_threshold,
            frame_time_fs=frame_time_fs,
        )
        onset_cfg = cfg["onset"]
        new_onset = detect_onset(
            aggregate_df=agg_df_new,
            threshold=new_threshold,
            window_frames=onset_cfg["window_frames"],
            step_frames=onset_cfg["step_frames"],
            fraction_threshold=onset_cfg["fraction_threshold"],
        )

        new_onset_frame = (
            new_onset.persistent_bead_anomaly_frame
            if args.onset_type == "persistent"
            else new_onset.first_bead_anomaly_frame
        )
        if new_onset_frame is None:
            sys.exit(
                f"No anomaly detected at threshold={new_threshold}. "
                "Try a lower value."
            )

        # Re-detect first_anomaly_bead_idx at new threshold
        ood_mask = bead_scores > new_threshold
        if ood_mask.any():
            first_ood = np.where(
                ood_mask.any(axis=1),
                np.argmax(ood_mask, axis=1),
                np.iinfo(np.int64).max,
            )
            bead_idx = int(np.argmin(first_ood))

        onset_frame = int(new_onset_frame)

    # ── Resolve bead and force files ──────────────────────────────────────────
    bead_files = sorted(glob.glob(run_cfg["bead_glob"]))
    if not bead_files:
        sys.exit(f"No bead files matched: {run_cfg['bead_glob']}")
    if bead_idx >= len(bead_files):
        sys.exit(
            f"first_anomaly_bead_idx={bead_idx} out of range "
            f"(found {len(bead_files)} bead files)."
        )
    bead_path = bead_files[bead_idx]
    force_path = _derive_force_path(bead_path)

    # ── Build / load byte-offset indices ─────────────────────────────────────
    print(f"Loading frame index for bead {bead_idx:02d} …")
    pos_idx = load_or_build_index(bead_path)
    for_idx = load_or_build_index(force_path)
    n_frames = pos_idx.n_frames

    if for_idx.n_frames != n_frames:
        sys.exit(
            f"Frame count mismatch: position file has {n_frames} frames, "
            f"force file has {for_idx.n_frames} frames."
        )

    # ── Compute extraction window ─────────────────────────────────────────────
    start = max(0, onset_frame - args.n_before)
    end = min(n_frames - 1, onset_frame + args.n_after)
    frame_range = range(start, end + 1)
    n_extract = len(frame_range)

    if start > onset_frame - args.n_before:
        print(f"Warning: n_before clamped to {onset_frame} (start of trajectory).")
    if end < onset_frame + args.n_after:
        print(f"Warning: n_after clamped to {n_frames - 1 - onset_frame} (end of trajectory).")

    print(f"Run:          {run_name}")
    print(f"Bead file:    {bead_path}")
    print(f"Force file:   {force_path}")
    print(f"Onset type:   {args.onset_type}")
    print(f"Onset frame:  {onset_frame}  (step {int(row[step_col])})")
    print(f"Extracting:   frames {start}–{end} ({n_extract} frames)")

    # ── Output path ───────────────────────────────────────────────────────────
    pct_tag = f"_thr{args.threshold}" if args.threshold is not None else ""
    out_name = (
        f"extraction_bead{bead_idx:02d}_frame{onset_frame}"
        f"_N{args.n_before}_M{args.n_after}{pct_tag}.extxyz"
    )
    out_path = run_out / out_name

    # ── Write extxyz (positions + forces, iPI metadata preserved) ─────────────
    n_atoms = int(pos_idx.atom_count[0])

    with (
        open(bead_path, "rb") as pos_src,
        open(force_path, "rb") as for_src,
        open(out_path, "wb") as dst,
    ):
        for fi in frame_range:
            pos_src.seek(int(pos_idx.byte_offset[fi]))
            pos_src.readline()                          # atom count
            comment = pos_src.readline().rstrip(b"\r\n")
            pos_lines = [pos_src.readline().rstrip(b"\r\n") for _ in range(n_atoms)]

            for_src.seek(int(for_idx.byte_offset[fi]))
            for_src.readline()                          # atom count
            for_src.readline()                          # comment (use pos comment for metadata)
            for_lines = [for_src.readline().rstrip(b"\r\n") for _ in range(n_atoms)]

            _write_extxyz_frame(dst, pos_lines, for_lines, comment)

    print(f"Saved:        {out_path}")


if __name__ == "__main__":
    main()
