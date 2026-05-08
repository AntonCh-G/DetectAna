#!/usr/bin/env python3
"""Extract frames around anomaly onset from the first-anomaly bead.

Reads onset_table.csv produced by run_pipeline.py to locate the onset frame
and the bead that first exceeded the OOD threshold, then copies N frames
before and M frames after that onset frame into a new XYZ file.

Frame bytes are copied verbatim — iPI comment lines (CELL, Step, Bead) are
preserved exactly as in the source file.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detectana.aggregator import aggregate_bead_scores
from detectana.onset import detect_onset
from detectana.xyz_reader import load_or_build_index



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

    # ── Resolve bead file ─────────────────────────────────────────────────────
    bead_files = sorted(glob.glob(run_cfg["bead_glob"]))
    if not bead_files:
        sys.exit(f"No bead files matched: {run_cfg['bead_glob']}")
    if bead_idx >= len(bead_files):
        sys.exit(
            f"first_anomaly_bead_idx={bead_idx} out of range "
            f"(found {len(bead_files)} bead files)."
        )
    bead_path = bead_files[bead_idx]

    # ── Build / load byte-offset index ───────────────────────────────────────
    print(f"Loading frame index for bead {bead_idx:02d} …")
    idx = load_or_build_index(bead_path)
    n_frames = idx.n_frames

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
    print(f"Onset type:   {args.onset_type}")
    print(f"Onset frame:  {onset_frame}  (step {int(row[step_col])})")
    print(f"Extracting:   frames {start}–{end} ({n_extract} frames)")

    # ── Output path ───────────────────────────────────────────────────────────
    pct_tag = f"_thr{args.threshold}" if args.threshold is not None else ""
    out_name = (
        f"extraction_bead{bead_idx:02d}_frame{onset_frame}"
        f"_N{args.n_before}_M{args.n_after}{pct_tag}.xyz"
    )
    out_path = run_out / out_name

    # ── Copy raw frame bytes (preserves iPI comment verbatim) ─────────────────
    # Frame fi spans byte_offset[fi] … byte_offset[fi+1]-1, or EOF for last frame.
    offsets = idx.byte_offset
    file_size = idx.file_size

    with open(bead_path, "rb") as src, open(out_path, "wb") as dst:
        for fi in frame_range:
            start_byte = int(offsets[fi])
            end_byte = int(offsets[fi + 1]) if fi + 1 < n_frames else file_size
            src.seek(start_byte)
            dst.write(src.read(end_byte - start_byte))

    print(f"Saved:        {out_path}")


if __name__ == "__main__":
    main()
