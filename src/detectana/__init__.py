"""DetectAna — anomaly-onset detection for single-molecule PIMD/MD trajectories.

Developed on aspirin; the molecule is defined by the run's ``initial.xyz``.

Library entry points (ADR 0005)::

    bundle = detectana.fit(train, valid, topology)   # once per force field
    bundle.save("aspirin.bundle")
    bundle = detectana.load("aspirin.bundle")
    t2 = bundle.score(frames)["geometric"]
"""

__version__ = "0.3.0"

from detectana.bundle import BundleError, ReferenceBundle, fit, load  # noqa: E402

__all__ = ["BundleError", "ReferenceBundle", "__version__", "fit", "load"]
