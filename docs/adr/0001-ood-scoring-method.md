# ADR 0001 — OOD Scoring: Internal Coordinates + PCA + Mahalanobis

**Status:** Accepted
**Date:** 2026-05-07

## Context

Need a scalar OOD score per frame (bead or centroid) that:
- Is fit on training reference data only
- Is calibrated on held-out validation data
- Works for 21-atom aspirin with ~200k frames per bead (streaming + caching required)
- Is chemically interpretable (top contributing coordinates diagnosable)
- Has no dependency on atom-centered symmetry functions (deferred to future work)

Alternatives considered:
- **SOAP/ACSF descriptors**: More expressive but adds `dscribe`/`librascal` dependency; explicitly listed as future work in AGENTS.md
- **Raw Cartesian RMSD**: Rotation-sensitive; less interpretable for flexible molecule
- **Isolation Forest / kernel methods**: Black-box; harder to diagnose which coordinate drives the score

## Decision

1. **Descriptor**: Internal coordinate fingerprint — all bonded bond lengths, bond angles, dihedral torsions (sin/cos encoded), benzene ring planarity deviation. Standardized using training-set mean/std.
2. **Dimensionality reduction**: PCA fit on training frames; retain components explaining 95% of variance.
3. **OOD score**: Mahalanobis distance in PCA-reduced space, using covariance of training projections.
4. **Threshold**: 99th percentile of Mahalanobis scores on the validation set (no PIMD frames used).
5. **Descriptor cache**: Computed descriptors saved to `outputs/descriptors/` as NPZ. Bead XYZ parsed once; downstream scoring/onset logic reads cache.

## Consequences

- Interpretable: largest PCA loading or largest per-coordinate deviation identifies which bond/angle/torsion drives an anomaly.
- Threshold has clear meaning: "exceeds 99% of held-out reference frames."
- Cache decouples slow XYZ parsing (~9 GB) from fast iterative scoring.
- SOAP upgrade path: replace descriptor module without changing scorer, aggregator, or onset detector.
