# ADR 0005 — DetectAna is a science library; the fit is a versioned reference bundle

**Status:** Accepted (not built)
**Date:** 2026-11-04

## Context

DetectAna runs after the fact: a config, a batch pipeline, files out. It was
originally meant to watch runs as they happen. FFAST (a separate project by the same
author: a server and web viewer for molecular datasets on HPC clusters) is gaining
live watching of LAMMPS runs, and needs DetectAna's science while the run is going:
fit once per force field, then score each new frame, decide when an anomaly has
persisted, and explain it.

Two ways were live. DetectAna could own the live loop itself: sockets, processes, a
LAMMPS hook, run folders. Or DetectAna could stay a library and FFAST could own the
plumbing. Two outside reviews of the joint design (FFAST repo,
`docs/research/lammps-live-2026-11-04/`) also found that today's persisted state is
not a contract: `DescriptorPipeline.save` and `MahalanobisScorer.save` pickle Python
objects, with no record of atom order, units, feature order or software versions,
so nothing can check that a saved fit matches the run being scored.

## Decision

1. **DetectAna is a pure science library.** It owns descriptors, tracks, scores and
   thresholds, attribution, chemistry flags and the onset rule. It knows nothing
   about LAMMPS, processes, sockets or FFAST's files. The batch pipeline and scripts
   remain as they are.
2. **The fit is a reference bundle with a versioned manifest**: atom order and
   topology, units, feature order per track, parameters, fitted state, calibrated
   thresholds, DetectAna version and the code hash of each descriptor used. A public
   API exposes what callers need (atom order, elements, units, tracks, how much
   history the onset rule needs, validation of a first frame); callers never open
   the fitted state themselves.
3. **A minimal interface comes first**: `fit(train, valid, …) → bundle`, `save`,
   `load`, `score(frames) → per-track scores`, and the window onset rule usable one
   frame at a time. Today's internal-coordinate descriptor is its single track until
   ADR 0006 is built.
4. **Slim core dependencies**: numpy, scipy, scikit-learn and ase. pandas,
   matplotlib, h5py, pyyaml and joblib move to a `[pipeline]` extra for the batch
   scripts and plots.
5. Publishing on PyPI (the name is free) is a separate, later decision; FFAST ships
   the wheel to the cluster until then.

## Consequences

- A score can always be traced to the bundle that produced it, and a bundle refuses
  frames whose atom order, units or descriptor code do not match.
- Existing pickled `pipeline.pkl` / `scorer.pkl` files are not bundles; the batch
  pipeline keeps reading them, or writes bundles going forward.
- Live behaviour (when to stop, what to show) is FFAST's decision; DetectAna only
  answers "what is the score, did the window rule fire, why".
- Importing the core no longer pulls in plotting libraries.

## Alternatives considered

- **DetectAna owns the live loop.** Rejected: infrastructure in a small research
  package that backs a paper, and every live feature would need both repositories
  released together.
- **Merge DetectAna into FFAST.** Rejected: loses DetectAna as a standalone,
  citable tool.
- **Keep pickles as the saved fit.** Rejected: no way to detect a mismatch between a
  fit and the run it scores.
