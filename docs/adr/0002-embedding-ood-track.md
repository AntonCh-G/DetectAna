# ADR 0002 — Embedding OOD Track: MlffModel inv_features + Per-atom Mahalanobis

**Status:** Proposed
**Date:** 2026-05-18

## Context

ADR-0001 established geometric OOD scoring (internal coordinates → PCA →
Mahalanobis). That track detects *structural* outliers — unusual bond lengths,
angles, dihedrals, ring planarity.

A complementary question is: when does the MLFF itself start *extrapolating*?
A frame can be structurally unusual but still within the model's accuracy domain,
or structurally normal but in a region of configuration space the model has not
seen. The geometric track cannot distinguish these cases.

MlffModel exposes `inv_features` — invariant per-atom embeddings of shape
`(n_atoms, n_features)` — via `return_descriptors=True` in the forward pass.
These embeddings are shaped by the same training distribution as the MLFF weights.
A frame that is OOD in this space is, by construction, a frame the model had to
extrapolate on when predicting energy/forces. This is a tighter coupling between
OOD signal and model reliability than any geometry-based metric.

## Decision

Add a second, fully independent OOD track alongside the geometric track.

### Architecture

1. **Decoupled inference** — MlffModel runs separately (GPU node) and writes
   per-atom embeddings to HDF5 files. DetectAna never imports `mlff_torch`.
   This preserves DetectAna's CPU-only install footprint.

2. **File schema** — one HDF5 per bead trajectory, one for the centroid, one for
   the reference training set. Bead files pointed to by `embedding_glob`
   (mirrors existing `bead_glob`); centroid file by `centroid_embedding_h5`.
   Each file stores frames at `embedding_stride` intervals within
   `[frame_start, frame_end)`.

3. **Per-atom Mahalanobis** — fit 21 independent Mahalanobis scorers (one per
   atom index) on the reference training embeddings. Atom ordering is stable
   (validated against `initial.xyz`). Threshold: 99th percentile of per-atom
   scores on the validation set.

4. **Per-frame aggregation** — reduce `(21,)` per-atom scores to one scalar via
   `max`. Rationale: consistent with AGENTS.md rule of "never average away bead
   signals"; flags as OOD if *any* atom is extrapolating.

5. **Bead aggregation** — aggregate per-frame max scores over 16 beads via the
   same metrics as the geometric track (`bead_max`, `bead_p95`, `bead_frac_ood`,
   `centroid_score`, `centroid_ood`).

6. **Two parallel onset timestamps** — `geometric_onset` and `embedding_onset`
   always appear side-by-side in the output table. Neither gates nor depends on
   the other. The comparison — does the MLFF enter extrapolation before or after
   the geometry becomes anomalous? — is a first-class output.

7. **One checkpoint per method** — PBE0/CCSD/RPA/VMC/VD each have one MlffModel
   checkpoint shared across all replica runs of that method. The checkpoint path
   is used by the extraction script only, not by DetectAna.

### New config fields (under `embedding:`)

```yaml
embedding:
  enabled: true
  reference_train_h5: path/to/ref_train_embeddings.h5
  reference_valid_h5: path/to/ref_valid_embeddings.h5
  stride: 10            # run every 10th frame (relative to trajectory stride)
  frame_start: null     # null = beginning of trajectory
  frame_end: null       # null = end of trajectory
```

Per run:
```yaml
runs:
  - name: "jax_PBE0"
    ...
    embedding_glob: path/to/aspirin.emb_*.h5   # expands to 16 bead files
    centroid_embedding_h5: path/to/aspirin.emb_xc.h5
```

### New DetectAna module

`src/detectana/embedding_scorer.py` — `EmbeddingPipeline` class:
- `fit(ref_train_embeddings)` — fit 21 per-atom Mahalanobis scorers
- `calibrate(ref_valid_embeddings, percentile=99.0)` — set per-atom thresholds
- `score(embeddings)` — return per-frame max-over-atoms score
- `save()` / `load()` — serialization

## Alternatives considered

- **Replace geometric track** — rejected; the two tracks answer different
  questions. Geometric OOD is interpretable (which bond/torsion is anomalous);
  embedding OOD is model-reliability-oriented.
- **Mean-pool atoms before scoring** — rejected; averaging washes out single
  rogue atoms, violating the "never average away signals" rule.
- **Hard dependency on mlff_torch** — rejected; would break DetectAna on
  CPU-only nodes and couples two codebases with different release cycles.
- **Centroid approximated from bead embeddings** — rejected; mean of embeddings
  ≠ embedding of mean positions for a nonlinear model; the approximation error
  is unknown.
- **Sequential filter (geometric gates embedding)** — rejected; breaks the
  independence of the two tracks, making it impossible to detect cases where
  embedding OOD precedes geometric OOD.

## Consequences

- The output table gains `geometric_onset` and `embedding_onset` columns, making
  the lead/lag relationship between structural and model-reliability anomalies a
  primary scientific output.
- GPU inference (MlffModel) and CPU analysis (DetectAna) remain on separate nodes
  with a clean HDF5 handoff.
- `embedding_stride` and `frame_range` give control over the compute budget
  without affecting the geometric track.
- Per-atom attribution is preserved: the atom driving the max embedding score
  can be logged alongside the frame score for diagnostics.
