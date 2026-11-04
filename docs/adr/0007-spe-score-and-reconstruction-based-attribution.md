# ADR 0007 — A second score for new kinds of motion, and attribution by reconstruction

**Status:** Accepted (not built). Must pass the benchmark (ADR 0004) before being
claimed in a paper.
**Date:** 2026-11-04

## Context

The score is a Mahalanobis distance in the PCA space kept from the training frames.
It sees how unusual a frame is *within* the directions the training data varied in.
It cannot see motion in directions PCA dropped: a frame doing something the training
data never did can score low. Multivariate process monitoring, which uses the same
model, watches two statistics for this reason: Hotelling T² inside the model and the
squared prediction error (SPE, also called Q) of the residual.

The synthetic demo (FFAST repo, `docs/research/lammps-live-2026-11-04/rbc_demo.py`)
twisted the ester torsion by 130°: T² stayed at 0.4 of its threshold and SPE reached
2.3, so only SPE caught it. Two reviews reproduced the ratios.

Explaining *why* a frame scored high has no working tool. `inverse_contributions`
returns the part of each number PCA throws away, not its share of the score, and its
docstring says otherwise; nothing calls it. The code also returns the distance √T²
while calling it a Mahalanobis score, which a reviewer flagged as a naming hazard.

## Decision

1. **Two scores per track**: T² (stored squared; "distance" is only a display label
   for √T²) and SPE, each with its own threshold calibrated on validation frames.
   SPE is reported as inactive when PCA keeps every direction (ADR 0006).
2. **Attribution by reconstruction-based contribution** (RBC; Alcala & Qin,
   Automatica 45(7), 2009): for each descriptor group, how much the score would drop
   if that group were reconstructed to its expected value, for T² and for SPE. A
   dihedral's sin/cos pair is reconstructed together. Values are mapped to atoms
   through each number's declared atoms (ADR 0006).
3. **Wording**: RBC values are attribution strengths. They overlap and do not sum to
   the score (for `M = [[1,1],[1,1]]`, `z = [1,1]` the score is 4 and each single
   RBC is 4), so they are never presented as shares or causes.
4. **Persistence**: an alarm uses the existing window rule (`onset.py`; glossary
   "Window rule"), not a single frame. A time-based false-alarm rate was considered and dropped: it
   needs an alarm-episode model and new statistics, and callers know the planned
   run length.
5. `inverse_contributions` is removed or renamed to say what it computes.

## Consequences

- Twice the thresholds per track, each with a smaller share of the budget: slightly
  less sensitive per score.
- "Which kind of anomaly" is visible: unusual within known motion (T²) versus a new
  kind of motion (SPE).
- Callers can show which bonds, angles or torsions drove a score.
- Persisted scores change meaning from √T² to T²; bundles carry a version so old and
  new are not mixed.

## Alternatives considered

- **One combined index** (weighted T² + SPE). Rejected: hides which kind of anomaly.
- **Plain contribution plots.** Rejected: known to blame the wrong variable for a
  large single fault; picked the same top item in the demo, so the difference is in
  correlated cases.
- **SHAP-style explainers.** Rejected: slower and approximate where RBC is exact for
  this model.
- **Per-group RBC as an alarm.** More sensitive to single bad numbers (the stretched
  bond of ADR 0006) but heavy false-alarm bookkeeping; to be tested in the benchmark
  before adoption.
