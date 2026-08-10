# Changelog

Notable changes to DetectAna. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/), with the pre-1.0 caveat that
interfaces may change between minor versions.

Scientific definitions — descriptor contents, thresholds, units, atom indexing —
are called out explicitly whenever they change, because a silent change there
produces results that still look plausible. See
[docs/scientific-rules.md](docs/scientific-rules.md).

## Unreleased

### Added
- Molecule-agnostic topology. The molecule is defined by the run's `initial.xyz`
  rather than by hard-coded constants: `io.MoleculeSpec` carries the expected atom
  count and element order, and the reference set, every run's initial geometry and
  every trajectory frame are validated against it. The ring used for the planarity
  feature is optional and selectable via `chemistry.ring_atoms`.
  ([ADR 0003](docs/adr/0003-molecule-agnostic-topology.md))
- False-alarm control for the onset rule. `onset.choose_fraction_threshold` derives
  the loosest `fraction_threshold` that stays inside a stated
  `onset.false_alarm_budget`, correcting for frame autocorrelation via an effective
  sample size. The bound is recorded in `manifest.json` whether or not the budget is
  used.
- Detector evaluation. `evaluation.py` provides detection metrics, force-error
  correlation and controlled distortions; `scripts/benchmark_detector.py` measures
  the detector using only data shipped in the repository. Synthetic positives are
  labelled by training coverage rather than distortion size.
  ([ADR 0004](docs/adr/0004-detector-evaluation.md))
- `scripts/score_vs_error.py`, relating the OOD score to force-field error.
  Validated on synthetic controls; awaits recomputed reference forces.
- ruff, mypy, a PEP 561 `py.typed` marker, and an 85 % coverage floor in CI.

### Changed
- **Descriptor column order (no effect on scores).** `compute_descriptor` used to
  interleave dihedral sin/cos while `compute_descriptor_batch` — which the pipeline
  and every cached descriptor use — emits them as separate blocks, and
  `feature_names` matched neither. All three now agree on the batch layout. This is
  a permutation of feature columns: the standardiser, PCA and Mahalanobis distance
  are invariant to it, so scores and onsets are unchanged and existing descriptor
  caches remain valid. Per-feature labels are now correct, which they were not.
- `AspirinTopology` is now an alias of `MoleculeTopology`.
- `ChemistryFlags.ring_planarity_rmsd` is `None` for a molecule with no ring, and
  the descriptor is one column shorter in that case. Aspirin is unaffected.

### Verified unchanged
- The demo pipeline reproduces bit-identical `bead_scores.npy`,
  `centroid_scores.npy`, `frame_aggregate.csv` and `onset_table.csv` against the
  previous release. Generalising the topology and adding the calibration machinery
  changed no result.

## 0.1.0 — 2026-08-10

First public version: geometric OOD scoring for aspirin PIMD trajectories,
windowed onset detection, hard-chemistry checks, the optional embedding track, and
a runnable demo on `data/smoke/`.
