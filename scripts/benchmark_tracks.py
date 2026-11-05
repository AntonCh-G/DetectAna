#!/usr/bin/env python3
"""Benchmark one joined track against four tracks, with and without SPE (ADR 0006, 0007).

Four detectors at the same total false-alarm rate α. Each splits α equally over
its active tests (Bonferroni) and flags a frame when any test fires:

    A  joined track (bonds+angles+dihedrals+ring)   T²          (step-1 detector)
    B  joined track                                  T² + SPE
    C  four tracks: bonds, angles, dihedrals, ring   T²
    D  four tracks                                   T² + SPE    (provisional default)

and, reported separately, E: D's tests with each statistic replaced by its
largest per-group RBC (per-group attribution used as an alarm).

Thresholds are conformal order statistics on a calibration split of the
held-out frames (ADR 0004), not the bundles' own percentiles. Measured per
detector: the realised false-alarm rate on clean evaluation frames; detection
rates on distortion ladders (rattle, a ring bond stretched with its fragment,
one C–H stretched), on the ester torsion twisted by 130°, and on a torsion scan
split into well-sampled and never-visited slices; SPE's added detection; and,
for B and D, how often the strongest RBC group of a flagged frame touches the
distorted atoms.

Data
----
``--synthetic N_TRAIN N_HELD`` generates demo frames from the config's
initial geometry (``--motion harmonic`` or ``rattle``; see
``detectana.synthetic``). Every output is then labelled *demo data,
synthetic*: it checks the machinery and shows the direction of effects; it is
not evidence for or against the default on real data. Without ``--synthetic``
the config's reference files are used.

Usage
-----
    python scripts/benchmark_tracks.py --config config/demo.yaml \\
        --synthetic 3000 2000 --motion harmonic --false-flag-rate 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import detectana
from detectana.bundle import DEFAULT_TRACKS, JOINED_TRACKS, ReferenceBundle
from detectana.evaluation import (
    most_gap_rich_dihedral,
    perturb_bond_stretch,
    perturb_dihedral,
    perturb_gaussian,
    set_dihedral,
    union_flags,
    union_thresholds,
)
from detectana.io import MoleculeSpec, load_reference_frames
from detectana.synthetic import ESTER, reference_frames
from detectana.topology import build_topology

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("detectana.benchmark_tracks")

RATTLE_SIGMAS_A = [0.02, 0.05, 0.10, 0.15, 0.20]
STRETCH_DELTAS_A = [0.05, 0.10, 0.20, 0.30, 0.50]
TWIST_DEG = 130.0
TORSION_BINS = 12
RBC_DELTA_A = 0.30

Test = tuple[str, str]


def _flat(scores: dict[str, dict[str, np.ndarray]]) -> dict[Test, np.ndarray]:
    return {(t, s): v for t, by in scores.items() for s, v in by.items()}


class Detector:
    """A set of tests from one bundle, thresholded at α/K each on calibration frames."""

    def __init__(self, name: str, bundle: ReferenceBundle, scores: tuple[str, ...], rbc: bool = False):
        self.name, self.bundle, self.rbc = name, bundle, rbc
        self.tests = [t for t in bundle.tests if t[1] in scores]

    def statistics(self, positions: np.ndarray) -> dict[Test, np.ndarray]:
        if not self.rbc:
            flat = _flat(self.bundle.score(positions))
            return {t: flat[t] for t in self.tests}
        return {t: self.bundle.attribute(positions, t[0], score=t[1]).strength.max(axis=1) for t in self.tests}

    def calibrate(self, calib: np.ndarray, alpha: float) -> None:
        self.thresholds = union_thresholds(self.statistics(calib), alpha)

    def flags(self, positions: np.ndarray) -> np.ndarray:
        return union_flags(self.statistics(positions), self.thresholds)

    def top_groups_touch(self, positions: np.ndarray, truth: set[int]) -> np.ndarray:
        """Per flagged frame: does the strongest group of its most-exceeded test touch ``truth``?"""
        stats = self.statistics(positions)
        ratios = np.stack([stats[t] / self.thresholds[t] for t in self.tests])
        best = np.argmax(ratios, axis=0)
        flagged = ratios.max(axis=0) > 1.0
        hits = np.zeros(len(positions), dtype=bool)
        for i, test in enumerate(self.tests):
            rows = np.flatnonzero(flagged & (best == i))
            if len(rows) == 0:
                continue
            att = self.bundle.attribute(positions[rows], test[0], score=test[1])
            top = np.argmax(att.strength, axis=1)
            hits[rows] = [bool(set(att.group_atoms[g]) & truth) for g in top]
        return hits[flagged]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("N_TRAIN", "N_HELD"))
    parser.add_argument("--motion", choices=["harmonic", "rattle"], default="harmonic")
    parser.add_argument("--false-flag-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rbc-alarm", action="store_true", help="skip detector E")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    alpha = args.false_flag_rate
    rng = np.random.default_rng(args.seed)
    chem = cfg["chemistry"]
    topo = build_topology(cfg["runs"][0]["initial_xyz"], nl_mult=chem["nl_mult"], ring_atoms=chem.get("ring_atoms"))

    if args.synthetic:
        n_train, n_held = args.synthetic
        eq = topo.reference_positions
        train = reference_frames(eq, topo, n_train, args.seed + 1, motion=args.motion)
        held = reference_frames(eq, topo, n_held, args.seed + 2, motion=args.motion)
        data_label = f"DEMO DATA, synthetic ({args.motion} motion) — not evidence about real data"
    else:
        spec = MoleculeSpec(topo.n_atoms, tuple(topo.atom_types))
        train, _, _ = load_reference_frames(cfg["reference"]["train"], spec=spec)
        held, _, _ = load_reference_frames(cfg["reference"]["valid"], spec=spec)
        data_label = f"reference data from {args.config}"
    out_dir = Path(args.output_dir or Path(cfg["io"]["output_dir"]) / "benchmark_tracks")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Data: %s", data_label)

    calib, evals = held[::2], held[1::2]
    with warnings.catch_warnings():
        # The bundles' own thresholds are not used: conformal thresholds below.
        warnings.simplefilter("ignore", UserWarning)
        fit_kw = {"pca_variance": cfg["descriptor"]["pca_variance"],
                  "random_seed": cfg["descriptor"]["random_seed"]}
        joined = detectana.fit(train, calib, topo, tracks=JOINED_TRACKS, **fit_kw)
        four = detectana.fit(train, calib, topo, tracks=DEFAULT_TRACKS, **fit_kw)
    detectors = [
        Detector("A joined T2", joined, ("T2",)),
        Detector("B joined T2+SPE", joined, ("T2", "SPE")),
        Detector("C four T2", four, ("T2",)),
        Detector("D four T2+SPE", four, ("T2", "SPE")),
    ]
    rbc_alarm = None if args.no_rbc_alarm else Detector("E four max-group RBC", four, ("T2", "SPE"), rbc=True)
    for d in detectors + ([rbc_alarm] if rbc_alarm else []):
        d.calibrate(calib, alpha)
    comps = {
        t: int(b.manifest["tracks"][t]["n_components"]) for b in (joined, four) for t in b.tracks
    }
    log.info("PCA components per track: %s (train frames %d)", comps, len(train))

    # ── Positive sets, each with the atoms it distorts ───────────────────────
    sym = topo.atom_types
    ch = next(k for k, (i, j) in enumerate(topo.bonds) if {sym[i], sym[j]} == {"C", "H"})
    sets: list[tuple[str, float, np.ndarray, set[int] | None, dict]] = []
    for s in RATTLE_SIGMAS_A:
        sets.append(("rattle", s, perturb_gaussian(evals, s, rng), None, {}))
    for kind, bond in (("stretch_ring_bond", 0), ("stretch_CH", ch)):
        i, j = topo.bonds[bond]
        for d in STRETCH_DELTAS_A:
            pos = np.stack([perturb_bond_stretch(f, topo, d, bond_index=bond) for f in evals])
            sets.append((kind, d, pos, {i, j}, {}))
    twist = np.stack([perturb_dihedral(f, topo, TWIST_DEG, dihedral=ESTER) for f in evals])
    sets.append(("ester_twist", TWIST_DEG, twist, set(ESTER), {}))
    quad, coverage = most_gap_rich_dihedral(train, topo, n_bins=TORSION_BINS)
    dense_cut = max(1, int(0.01 * len(train)))
    for _, row in coverage.iterrows():
        angle = float(row["angle_centre"])
        sets.append(("torsion_scan", angle, set_dihedral(evals, topo, quad, angle),
                     set(quad), {"train_count": int(row["train_count"])}))

    rows = []
    neg = {d.name: d.flags(evals) for d in detectors + ([rbc_alarm] if rbc_alarm else [])}
    for kind, magnitude, pos, truth, extra in sets:
        flags = {d.name: d.flags(pos) for d in detectors + ([rbc_alarm] if rbc_alarm else [])}
        row = {"set": kind, "magnitude": magnitude, **extra}
        for name, f in flags.items():
            row[f"det:{name}"] = float(f.mean())
        row["spe_added:B_not_A"] = float((flags["B joined T2+SPE"] & ~flags["A joined T2"]).mean())
        row["spe_added:D_not_C"] = float((flags["D four T2+SPE"] & ~flags["C four T2"]).mean())
        if truth is not None and (kind != "torsion_scan" or extra["train_count"] == 0) \
                and (kind not in ("stretch_ring_bond", "stretch_CH") or magnitude == RBC_DELTA_A):
            for d in detectors[1::2]:  # B and D
                hits = d.top_groups_touch(pos, truth)
                row[f"rbc_hit:{d.name}"] = float(hits.mean()) if len(hits) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tracks_benchmark.csv", index=False)

    scan = df[df["set"] == "torsion_scan"]
    names = [d.name for d in detectors] + ([rbc_alarm.name] if rbc_alarm else [])
    summary = {
        "data": data_label,
        "false_flag_rate": alpha,
        "n_train": len(train), "n_calibration": len(calib), "n_evaluation": len(evals),
        "pca_components": comps,
        "tests": {d.name: [list(t) for t in d.tests] for d in detectors},
        "false_alarm_rate_on_clean_frames": {n: float(neg[n].mean()) for n in names},
        "torsion": [int(i) for i in quad],
        "torsion_scan_dense": {n: float(scan[scan["train_count"] >= dense_cut][f"det:{n}"].mean()) for n in names},
        "torsion_scan_unvisited": {n: float(scan[scan["train_count"] == 0][f"det:{n}"].mean())
                                   if (scan["train_count"] == 0).any() else None for n in names},
    }
    (out_dir / "tracks_benchmark.json").write_text(json.dumps(summary, indent=2))

    # ── Printout ─────────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print(f"\n=== {data_label} ===")
    print(f"α = {alpha:g} total per detector; {len(train)} train, {len(calib)} calibration, "
          f"{len(evals)} evaluation frames; PCA components {comps}")
    print("\nFalse-alarm rate on clean evaluation frames (should be ≤ α):")
    for n in names:
        n_tests = len(summary["tests"].get(n, [])) or "-"
        print(f"  {n:24s} {summary['false_alarm_rate_on_clean_frames'][n]:6.2%}   tests: {n_tests}")
    cols = [f"det:{n}" for n in names if not n.startswith("E")]
    ladders = df[df["set"].isin(["rattle", "stretch_ring_bond", "stretch_CH", "ester_twist"])]
    print("\nDetection rate, A–D:")
    print(ladders[["set", "magnitude", *cols]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTorsion scan on", quad, f"(dense = ≥{dense_cut} training frames in the 30° slice):")
    for key in ("torsion_scan_dense", "torsion_scan_unvisited"):
        print(f"  {key:24s}", "  ".join(f"{n.split()[0]}={v:.3f}" if v is not None else f"{n.split()[0]}=n/a"
                                         for n, v in summary[key].items() if not n.startswith("E")))
    print("\nSPE's added detection (fraction of positives caught only thanks to SPE):")
    print(df[["set", "magnitude", "spe_added:B_not_A", "spe_added:D_not_C"]]
          .groupby("set").max().to_string(float_format=lambda v: f"{v:.3f}"))
    hit_cols = [c for c in df.columns if c.startswith("rbc_hit:")]
    print("\nRBC hit rate (flagged frames whose strongest group touches the distorted atoms):")
    print(df.dropna(subset=hit_cols, how="all")[["set", "magnitude", *hit_cols]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if rbc_alarm:
        print("\nE (reported separately — per-group RBC as an alarm), detection rate:")
        print(df[["set", "magnitude", f"det:{rbc_alarm.name}", "det:D four T2+SPE"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_dir / 'tracks_benchmark.csv'} and tracks_benchmark.json")


if __name__ == "__main__":
    main()
