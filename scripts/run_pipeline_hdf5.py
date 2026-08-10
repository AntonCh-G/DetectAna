#!/usr/bin/env python3
"""Deprecated wrapper. Use ``scripts/run_pipeline.py`` instead.

``detectana.pipeline.run_pipeline`` reads HDF5 trajectories itself — a run with an
``hdf5:`` key instead of ``bead_glob``/``centroid_xyz`` takes the HDF5 path — and
it also supports the two-threshold schema this script introduced
(``threshold.bead_percentile`` / ``threshold.centroid_percentile``). This file
used to carry its own copy of the per-run loop, the plots and the manifest
writer; that duplication has been removed and it now just forwards to the
library.

Usage (unchanged)
-----------------
    python scripts/run_pipeline_hdf5.py --config config/pimd6_s4_hdf5.yaml

Config shape
------------
runs:
  - name: s1
    hdf5: /path/to/nvt_trajectory.hdf5
    initial_xyz: /path/to/input.xyz
    timestep_fs: 0.2
    stride: 50
threshold:
  bead_percentile: 99.0      # sensitive, for the individual beads
  centroid_percentile: 85.0  # strict, for the centroid

Output differences vs the old script
-----------------------------------
- ``manifest.json`` is written per run (``<out>/<run>/manifest.json``) rather
  than once at the output root, so each run records its own onset design.
- ``onset_summary.csv`` is still written at the output root.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger("detectana.run_hdf5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DetectAna — HDF5 pipeline (deprecated wrapper)")
    p.add_argument(
        "--config",
        default="config/pimd6_s4_hdf5.yaml",
        help="YAML config file",
    )
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.warning(
        "scripts/run_pipeline_hdf5.py is deprecated: run_pipeline.py handles the "
        "'hdf5:' run key and the bead/centroid threshold percentiles. "
        "Switch to: python scripts/run_pipeline.py --config %s",
        args.config,
    )

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        return 1

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.force_recompute:
        cfg.setdefault("io", {})["force_recompute"] = True

    for run_cfg in cfg.get("runs", []):
        if "hdf5" not in run_cfg:
            log.error(
                "Run %r has no 'hdf5' key. This entry point is for HDF5 "
                "trajectories; use scripts/run_pipeline.py for XYZ runs.",
                run_cfg.get("name", "?"),
            )
            return 1

    try:
        from detectana.pipeline import run_pipeline
    except ImportError as exc:
        log.error("Import failed — is detectana installed? (pip install -e .)\n%s", exc)
        return 1

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
