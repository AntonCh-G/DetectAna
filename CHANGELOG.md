# Changelog

Notable changes to DetectAna. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/), with the pre-1.0 caveat that
interfaces may change between minor versions.

Scientific definitions — descriptor contents, thresholds, units, atom indexing —
are called out explicitly whenever they change, because a silent change there
produces results that still look plausible. See
[docs/scientific-rules.md](docs/scientific-rules.md).

## 0.3.0 — 2026-08-11

No change to the detection code. This release makes the repository standalone —
the data it ships is its own, and nothing in it points at datasets or systems
outside it — and reorganises the documentation so the method can be read without
reverse-engineering the code.

### Added
- `docs/methodology.md` — the full method in one place: problem formulation,
  the seven assumptions and what breaks if each is violated, inputs and
  preprocessing, descriptor and Mahalanobis definitions, threshold calibration,
  the onset rule with its false-alarm arithmetic, the evaluation design
  (including the mislabelling mistake that shaped it), and what a flag does and
  does not license you to claim.
- `docs/reproducibility.md` — environment, the shortest path from clone to
  result, what data a real run needs and where to put it, determinism per
  stochastic component, manifest contents, and an explicit list of what cannot
  be reproduced from a clone and what it would take.
- `docs/usage.md` — configuration keys, every script, outputs, repository layout.
- `scripts/make_demo_data.py` — regenerates `data/smoke/` deterministically.

### Changed
- README reorganised as a landing page: research question, methodology summary,
  repository map, reproduction commands, and a results section that separates
  measured results from an illustrative single run, computed false-alarm
  arithmetic, and the validation that has not been done.
- `CONTEXT.md` renamed to `docs/glossary.md`. Same content; the old name
  described its role for coding agents rather than for a reader.
- `--config` is now required by `scripts/run_pipeline.py` and
  `scripts/run_pipeline_hdf5.py`. Both previously defaulted to a local,
  untracked config, so a fresh clone got "config file not found" instead of a
  usage message. Every documented invocation already passed `--config`.
- Version numbers in `pyproject.toml`, `detectana.__version__` and
  `CITATION.cff` were still 0.1.0 while this changelog described 0.3.0. All now
  read 0.3.0.
- **The MLFF architecture is no longer named anywhere.** It is part of
  unpublished work. `scripts/extract_embeddings.py` now takes `--model-package`
  (required), with optional `--model-class` and `--feature-key`, and resolves the
  model class and the graph builder by import rather than by hard-coded name; the
  adapter contract it expects is documented in its docstring. Documentation and
  docstrings in `docs/glossary.md`, `docs/adr/0002`, `io.py` and
  `embedding_scorer.py` describe the interface — invariant per-atom features —
  instead of the architecture. The HDF5 dataset name `inv_features` is unchanged,
  since it is part of the on-disk format that `io.load_embeddings_h5` reads.
- Documentation now states that the reference dataset and the production
  trajectories belong to work that is not yet published, that `data/smoke/` stands
  in for them, and that the intention is to replace it with real reference frames
  once that work is published.
- `ruff` now checks the whole repository. Four scripts were excluded pending
  other work; the six findings (import order, two unused names, two
  `raise ... from None`) are fixed and the exclusion list is gone. No behaviour
  changed.
- **Demo data is now synthetic.** `data/smoke/` used to hold real geometries lifted
  from the reference set. It is now generated from the single equilibrium structure
  in `initial.xyz` by perturbing the two flexible torsions and rattling every atom
  with an element-dependent amplitude — no simulation, no force field, no reference
  calculation, and therefore no forces, which is all the geometric pipeline needs.
  Frame counts are unchanged (64 training, 32 validation, 24 trajectory). Demo
  output changes because the input geometries changed; it was never a result.
- Project-specific identifiers removed from the documentation and the script usage
  examples: named quantum-chemical methods, internal dataset and run directory
  names, and the MLFF package name. The pipeline never read any of them — the
  reference method is a property of the dataset, not of DetectAna.
- README rewritten for a first-time reader. The configuration reference, the
  per-script documentation and the output layout moved to `docs/usage.md`.
- The benchmark numbers in the README and [ADR 0004](docs/adr/0004-detector-evaluation.md)
  now state which set they were measured on and that the set is not in this
  repository, so nothing here reads as reproducible from a clone when it is not.
- `.gitignore` covers `.ruff_cache/`, `.mypy_cache/` and `coverage.xml`.

## 0.2.0 — 2026-08-10

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
