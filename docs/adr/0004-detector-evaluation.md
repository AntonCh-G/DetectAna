# ADR 0004 — Evaluating the detector: coverage-labelled distortions

**Status:** Accepted
**Date:** 2026-08-10

## Context

ADR-0001 chose Mahalanobis distance in PCA space and ADR-0002 added the embedding
track, but neither established how good either detector is. The pipeline reported
which frames were out of distribution and never what fraction of genuinely
abnormal frames it would catch, or what a score of 12 meant. There were no
detection metrics anywhere in the codebase.

Real labelled anomalies are unavailable: nobody has a trajectory annotated
frame-by-frame with "the force field is unreliable here". So positives have to be
manufactured, and the obvious approach — distort frames, call them anomalies,
compute AUROC — is wrong in a way that is easy to miss.

We hit it directly. The first version of the benchmark rotated whichever torsion
divided the molecule most evenly and scored the detector near chance across
rotations from 5° to 180°, which reads as a detector blind to conformational
change. It was not. The chosen torsion (the ester) is fully sampled in the
reference set — every 30° slice occupied — so the rotated frames were
in-distribution and flagging them would have been the error. The benchmark was
measuring its own mislabelling.

**"Distorted" and "out of distribution" are different properties.** A distortion
that lands inside the training distribution is not an anomaly, no matter how large
the displacement.

## Decision

Label every synthetic positive by **training coverage**, not by distortion size.

`evaluation.torsion_coverage` histograms a torsion over the training set;
`most_gap_rich_dihedral` finds the torsion with the most unvisited slices;
`set_dihedral` drives frames to an exact target angle. The benchmark then scans a
torsion around its full circle and reports the flag rate per slice against the
training count in that slice. Two numbers come out with opposite meanings:

- flag rate in **well-sampled** slices — a false-alarm measurement, should sit near α
- flag rate in **never-visited** slices — a sensitivity measurement, should be high

and the Spearman correlation between training density and flag rate summarises
both. Pooling them, as the first version did, averages a specificity result with a
sensitivity result and reports a meaningless middle.

Torsion rotation is the right probe because it is a pure conformational change:
bond lengths and bond angles are preserved exactly, which
`tests/test_evaluation.py` asserts to 1e-9 rather than assuming. Cartesian rattle
and bond stretch are kept as ladders, the first as a floor and the second because
it can be compared against the hard-chemistry bond check.

Detection rates use the conformal order statistic of the calibration scores — the
same threshold rule the pipeline deploys — so a reported detection rate answers
"what fraction of these would a real run have flagged?" rather than describing an
idealised detector.

## Consequences

- The benchmark was run on the aspirin reference set used during development.
  **The quantitative results are withheld**: that set belongs to work which is not
  yet published, so neither the numbers nor the figures are in this repository and
  they cannot be reproduced from a clone. Qualitatively, the detector flags a
  conformer in proportion to how little training data supports it, which is what
  the project's definition of OOD asks for. The shipped demo data reproduces the
  machinery, not the result. Results will be published with the associated work.
- The benchmark confirms the "OOD is not unphysical" rule empirically: sparsely
  covered torsion slices are flagged reliably, and those conformers are chemically
  ordinary. The score measures coverage, not correctness.
- Bond stretches well below the 2.0 Å hard-chemistry cutoff are caught while that
  cutoff is still silent, so the statistical track adds sensitivity rather than
  duplicating the chemistry check.
- A separate question needs external data: whether a high score predicts actual
  force error. `scripts/score_vs_error.py` implements it, and it stays out of CI
  because it needs forces recomputed with the reference method.
- The benchmark warns when training frames number fewer than ten per PCA
  component, since the estimated covariance is then dominated by noise. The demo
  configuration is deliberately in that regime — 64 synthetic frames, 134 descriptor
  columns — and its benchmark output is a smoke test of the machinery rather than a
  measurement.

## Alternatives considered

**Distortion magnitude as the label.** Rejected: it produced the false blind-spot
result described above. Displacement size does not determine whether a frame is
outside the training distribution.

**Hold out a conformational state.** Refit with one basin removed from training and
test detection on it. Cleaner in principle, and worth doing later, but it needs a
state assignment for the reference set and changes the fit, so the numbers are not
comparable with the deployed detector. The torsion scan gets the same information
from the existing fit.

**Real labels from recomputed energies.** The honest gold standard, and what
`score_vs_error.py` is for. Not usable as the default benchmark because it requires
expensive reference calculations, which would put the evaluation out of reach of
anyone cloning the repository.
