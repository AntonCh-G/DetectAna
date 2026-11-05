# ADR 0006 — Descriptors come from a registry and are scored in tracks

**Status:** Accepted; built 2026-11-05 (`registry.py`, `bundle.py` format 2). The
default track split is provisional until benchmarked on real reference data
(`scripts/benchmark_tracks.py`).
**Date:** 2026-11-04

## Context

The descriptor is one fixed function, `compute_descriptor`: bond lengths, angles,
dihedral sin/cos pairs and ring planarity joined into one vector (134 numbers for
aspirin), then standardised, reduced by PCA and scored once. Trying another
descriptor (SOAP, ACSF, distances to a reference) means editing the package.

A synthetic demo (FFAST repo, `docs/research/lammps-live-2026-11-04/rbc_demo.py`;
134 features, PCA keeping 53, 3,000 training frames) showed a cost of the single
joined score: one C–H bond stretched by 0.30 Å, about four times its normal spread,
raised neither score above threshold. The one bad number was averaged away among 53
directions. An outside review pointed out the opposite risk: fitting families
separately loses the correlations between them, which PCA and SPE (ADR 0007) can use.

Two fitting failures also surfaced: a one-number input crashes
`MahalanobisScorer.fit` (`np.cov` returns a scalar), and PCA that keeps every
direction leaves no residual, so SPE (ADR 0007) is identically zero.

## Decision

1. **A descriptor registry.** A descriptor is a function marked
   `@descriptor("name")` that maps positions and topology to named numbers, and
   declares for each number the atoms it involves and a group ID (a dihedral's sin
   and cos share one group). Built-ins are registered in DetectAna; a user module can
   be named in the config (`mypkg.mydesc:soap`) without changing DetectAna, provided
   it is installed where scoring runs.
2. **Tracks.** A track is a set of descriptor numbers fitted, scored and thresholded
   together, in the same sense as the embedding track of ADR 0002. Tracks are set in
   the config. The default is bonds, angles, dihedrals and ring as four tracks;
   this default is **provisional** until DetectAna's benchmark (ADR 0004) compares
   it with one joined track at the same total false-alarm budget.
3. **Budget over active tests.** The false-alarm budget is split over the tests that
   can fire; a track with an inactive score (ADR 0007) takes no share.
4. **Degenerate tracks work.** One-number tracks fit with a scalar covariance.
5. **Bundles record descriptor code.** The bundle stores each descriptor's name and a
   hash of its code; loading refuses a bundle whose descriptors have changed.
6. **Naming.** "Track" means a separately fitted scoring group. The existing
   `MahalanobisScorer.thresholds[track]` keys (`bead`, `centroid`) are a different
   idea, a threshold role inside one track; they are renamed *threshold role*
   (`track=` becomes `role=` in `MahalanobisScorer.calibrate`) when this is built.
   Confirmed 2026-11-04: "track" always means something with its own fitted model
   and its own score.

## Amendment (2026-11-05, when built)

Decided while building; the decisions above are unchanged.

- **Budget split rule**: Bonferroni. Each of the K active tests gets α/K of its
  role's total α; a frame is marked when any test fires. Valid whatever the
  correlation between tests, conservative when they are correlated. `fit` warns
  when n_valid·α/K < 1.
- **Code hash of a user descriptor** covers the whole source file of its module
  plus its package's installed version, not only the function: helpers in the
  same file are then covered, and false refusals (refit) are cheaper than
  silently wrong scores. Built-ins hash the function and the helpers DetectAna
  lists. Code imported from other files of an editable install is not covered.
- **Loading never imports a user module.** A bundle is data, and an import runs
  code: the caller imports the module first or passes
  `load(path, descriptor_modules=[...])`. `fit` may import, since the caller
  named the module.
- **Groups** are defined by the descriptor author with no size rule; RBC is not
  normalised by group size, and each group's size is reported with it.
- **Empty tracks** (ring on a ring-less molecule) are dropped and recorded.
- **The batch pipeline** scores the config's tracks (absent → the default) and
  stores the exceedance — largest score ÷ threshold over the tests — as each
  frame's score, so the aggregation and window rule are unchanged. The joined
  track with T² alone reproduces the earlier pipeline exactly.

## Consequences

- A new descriptor is one function in a user module, with no DetectAna release.
- Which descriptor group noticed an anomaly becomes visible per track.
- More fits per bundle, and each track gets a smaller slice of the alarm budget.
- A code hash can reject harmless edits (formatting) and miss changes in helpers a
  descriptor calls; the manifest also records the DetectAna version to narrow this.

## Alternatives considered

- **Built-ins only.** Rejected: every experiment needs a DetectAna change.
- **Installable plugin packages (entry points).** Deferred until others publish
  descriptors.
- **One joined score, as today.** Kept as the comparison baseline in the benchmark;
  may still win.
