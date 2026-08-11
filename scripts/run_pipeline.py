#!/usr/bin/env python
"""CLI entry point for the DetectAna anomaly-onset pipeline.

Usage
-----
    python scripts/run_pipeline.py --config config/demo.yaml
    python scripts/run_pipeline.py --config config/local.yaml --verbose
    python scripts/run_pipeline.py --config config/local.yaml --force-recompute

``config/demo.yaml`` runs on the synthetic data shipped in ``data/smoke/``.
Copy ``config/example.yaml`` for a real run; anything under ``config/`` other
than those two is git-ignored, so local paths stay local.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect anomaly onset in MD and PIMD trajectories."
    )
    # No default: the old one named a local, untracked config, so a fresh clone
    # got "config file not found" instead of a usage message.
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file (start from config/demo.yaml)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing descriptor caches and recompute from XYZ",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("detectana.run")

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        return 1

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.force_recompute:
        cfg["io"]["force_recompute"] = True

    log.info("Config: %s", config_path)
    log.info("Output: %s", cfg["io"]["output_dir"])

    # Import here so import errors surface with a clean message
    try:
        from detectana.pipeline import run_pipeline
    except ImportError as exc:
        log.error("Import failed — is detectana installed? (pip install -e .)\n%s", exc)
        return 1

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
