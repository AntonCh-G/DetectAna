"""DetectAna — anomaly-onset detection for single-molecule PIMD/MD trajectories.

Developed on aspirin; the molecule is defined by the run's ``initial.xyz``.

Library entry points (ADR 0005–0007)::

    bundle = detectana.fit(train, valid, topology)   # once per force field
    bundle.save("aspirin.bundle")
    bundle = detectana.load("aspirin.bundle")
    scores = bundle.score(frames)                    # {track: {"T2": ..., "SPE": ...}}
    marked = bundle.exceedance(scores, role="bead").marked
    why = bundle.attribute(frames, "bonds", score="T2").top(k=5)

A user descriptor is a function marked ``@detectana.descriptor("name")``.
"""

__version__ = "0.3.0"

from detectana.attribution import Attribution, GroupStrength  # noqa: E402
from detectana.bundle import (  # noqa: E402
    DEFAULT_TRACKS,
    JOINED_TRACKS,
    BundleError,
    Exceedance,
    FrameCheck,
    ReferenceBundle,
    fit,
    load,
)
from detectana.registry import DescriptorValues, descriptor  # noqa: E402

__all__ = [
    "DEFAULT_TRACKS",
    "JOINED_TRACKS",
    "Attribution",
    "BundleError",
    "DescriptorValues",
    "Exceedance",
    "FrameCheck",
    "GroupStrength",
    "ReferenceBundle",
    "__version__",
    "descriptor",
    "fit",
    "load",
]
