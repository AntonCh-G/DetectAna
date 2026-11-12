# Changelog

Notable changes to DetectAna. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/), with the pre-1.0 caveat that
interfaces may change between minor versions.

Scientific definitions — descriptor contents, thresholds, units, atom indexing —
are called out explicitly whenever they change, because a silent change there
produces results that still look plausible. See
[docs/scientific-rules.md](docs/scientific-rules.md).

## 0.4.0 — 2026-11-12

A library you can call frame by frame: a saved reference bundle, four
descriptor tracks each scored by T² and SPE under one false-alarm budget,
attribution to atoms, and runs read from any file ASE reads. **Results change**:
scores are T², batch-pipeline scores are exceedances, and configs without
`tracks:` now use four tracks; see Changed for how to reproduce 0.3.0.

### Added
- **`trajectory:` runs**: one classical MD file in any format ASE reads (`.traj`,
  extxyz, ...), with optional `format:`. The file is bead 00 and the centroid;
  steps are frame indices, so time is frame × `timestep_fs` × `stride`, as for
  HDF5. Read by `detectana.io.load_ase_trajectory`, which checks each frame's
  atom count and element order and warns that periodic frames are not
  unwrapped. A run must now name exactly one of `bead_glob`, `hdf5`,
  `trajectory`, and `bead_glob` needs `centroid_xyz`; the pipeline says so
  before fitting instead of failing on a missing key later.
- **`build_topology(..., bonds=...)`**: the caller gives the bond graph (for
  example found by a chemistry toolkit over many frames); angles, dihedrals and
  the ring follow from it. The covalent-radii search reads one geometry, and a
  single thermal frame can squeeze two non-bonded atoms into its range (its
  neighbour list adds a 0.3 Å skin per atom) or stretch a bond out of it.
- **Descriptor registry** (ADR 0006, `detectana.registry`): `@descriptor("name")`
  marks `fn(positions, topology) -> DescriptorValues` (values plus each number's
  name, atoms and group). Built-ins `bonds`, `angles`, `dihedrals`, `ring`; side by
  side they equal the joined internal-coordinate descriptor bit for bit. User
  descriptors are named `module.path:function`.
- **Tracks, SPE and the false-alarm budget** (ADR 0006, 0007): `fit(tracks=...,
  scores=...)`. Each track gets T² and SPE (the squared PCA residual); SPE is
  inactive when PCA keeps every direction or no residual variance remains. A
  threshold role's α is split equally over the active tests (Bonferroni). The
  default is four tracks — bonds, angles, dihedrals, ring — **provisional** until
  benchmarked on real data. `detectana.JOINED_TRACKS` gives the single joined track.
- `ReferenceBundle.exceedance(scores, role)`: largest score ÷ threshold per frame
  and the test behind it; above 1 = marked.
- `ReferenceBundle.attribute(frames, track, score)`: reconstruction-based
  contribution (RBC) per descriptor group, mapped to atoms (`detectana.attribution`).
- `ReferenceBundle.check_chemistry(frames)`: broken bonds, close contacts and ring
  planarity per frame against the bundle's own topology (`ChemistryFlags`, now
  exported). `ReferenceBundle.false_flag_rate(role)`: the role's total α.
  Attribution strengths overlap and do not sum to the score.
- `scripts/benchmark_tracks.py`: joined vs four tracks, with and without SPE, at
  one total false-alarm rate; per-group RBC as an alarm reported separately.
- `detectana.synthetic`: demo frames by per-atom rattle (the smoke set, unchanged)
  or by thermal sampling of a harmonic spring model (atoms move together).
- `topology.complete_topology()` fills feature names and index arrays.
- **Reference bundle** (ADR 0005): `detectana.fit(train, valid, topology)` →
  `ReferenceBundle` with `save`, `detectana.load`, `score(frames)` →
  `{"geometric": T²}`, `validate_frame`, and a metadata API (elements, atom
  count, units, topology, tracks, feature names, thresholds, onset rule,
  `history_frames`). Saved as `manifest.json` + `state.npz`, no pickle. `load`
  refuses another format version or statistic, changed descriptor code (SHA-256
  of its source), other units, or inconsistent arrays.
- `MahalanobisScorer.score_t2()` — Hotelling's T², the squared distance.
- `onset.WindowRule` and `WindowRule.stream()` — the window onset rule one frame
  at a time, equal to `detect_onset` on the same frames.
- `build_topology` accepts an ase `Atoms` as well as a path, and keeps the
  geometry it was built from (`MoleculeTopology.reference_positions`).
- `ReferenceBundle.validate_frame` returns a `FrameCheck`: bonds stretched past
  1.5× their reference length, and possible same-element atom swaps found by
  rigid alignment plus Hungarian matching against the bundle's reference
  geometry. Diagnostics only; they do not prove atom identity.
- `ReferenceBundle.score_features` scores descriptors computed elsewhere.

### Changed
- **Bundle format 2.** `score()` returns `{track: {"T2": ..., "SPE": ...}}`;
  `thresholds(track)` returns `{score: {role: value}}`. Format-1 bundles are refused
  (refit). The manifest records each descriptor's reference, code hash and numbers.
  `load(path, descriptor_modules=[...])`: a user descriptor's module is never
  imported because a bundle names it.
- **Scientific definition: thresholds of multi-test bundles.** With K active tests,
  each threshold is the 100·(1 − α/K) percentile of validation values, not the
  configured percentile. A one-test bundle (the batch pipeline's) is unchanged.
- **Scientific definition: batch pipeline scores are exceedances.** The pipeline
  fits the config's `tracks:` and `scores:` (absent → the four-track default) and
  stores, per bead and centroid frame, the largest score ÷ threshold over the
  tests (bead role for beads, centroid role for the centroid); above 1 = marked.
  `bead_scores.npy`, `centroid_scores.npy` and the score columns of
  `frame_aggregate.csv` change meaning accordingly; raw T²/SPE per test are in
  `bead_test_scores.npz` / `centroid_test_scores.npz`. Existing configs without
  `tracks:` change results on their next run; `tracks: {geometric: [bonds, angles,
  dihedrals, ring]}` with `scores: [T2]` reproduces the previous flags and onsets.
  The run manifest records `"statistic": "exceedance"`, the tests, thresholds per
  test and role, and the budget (the scalar `ood_threshold` keys are gone).
- `frame_aggregate.csv` adds the top test, per-test centroid scores, and for
  marked centroid frames the three strongest descriptor groups (RBC); a new
  `onset_attribution.csv` explains each alarm by the most exceeded centroid
  frame in the window that fired (`onset_frame` is the window start, `frame` the
  frame explained).
- `extract_onset_frames.py --threshold` is in the run's score units: exceedance
  for current runs, T² for runs written before tracks.
- `MahalanobisScorer`: `track=` is renamed `role=` (threshold role), and
  `DEFAULT_TRACK` is `DEFAULT_ROLE`. Pickles are unaffected.
- Core dependencies are `numpy`, `scipy`, `scikit-learn`, `ase`. `pandas`,
  `matplotlib`, `h5py`, `pyyaml`, `joblib` moved to a `[pipeline]` extra, which
  `[dev]` includes. Install with `pip install "detectana[pipeline]"` to run the
  batch pipeline and scripts.
- **Scientific definition: every OOD score is now T²**, the squared Mahalanobis
  distance, not the distance √T² — in the bundle, the batch pipeline
  (`bead_scores.npy`, `centroid_scores.npy`, the score columns of
  `frame_aggregate.csv`, thresholds, plots) and the embedding track. Thresholds
  are percentiles of validation T². Column and file names are unchanged; each
  run's `manifest.json` records `"statistic": "T2"`. Rank-based results (AUROC,
  average precision, conformal detection rates, Spearman) are unchanged; stored
  values, thresholds, medians and Pearson correlations are not. On the demo data
  every flag and onset is the same as in 0.3.0. With
  `frame_autocorrelation: auto`, ρ is now measured on the centroid T² series.
- The batch pipeline fits a reference bundle and saves it to
  `outputs/models/geometric.bundle`; it no longer writes
  `descriptor_pipeline.pkl` or `scorer.pkl`. `benchmark_detector.py` and
  `score_vs_error.py` fit the same way.
- `EmbeddingPipeline` pickles record their statistic; one saved before T² is
  refused on load. `score_vs_error.py --reuse-scan` refuses a pre-T² scan, and
  `extract_onset_frames.py --threshold` (now in T² units) refuses a run whose
  manifest does not record T².

### Removed
- `DescriptorPipeline.inverse_contributions`: it returned PCA residuals while its
  docstring claimed a share of the score. Use SPE and `attribute` (issue 01).

### Fixed
- **Units: the time axis of XYZ runs was too large by the output stride.** i-PI's
  step number already counts MD steps, and the pipeline multiplied it by
  `timestep_fs × stride`. `time_ps` in `frame_aggregate.csv` and the plot axes of
  XYZ runs are now `step × timestep_fs`; on the demo, 20 ps instead of 1000 ps.
  Earlier picosecond values from XYZ runs should be divided by `stride`. Step
  numbers, frame indices and HDF5 runs were right and are unchanged.
- Fitting a one-feature input crashed `MahalanobisScorer.fit` (`np.cov` returns
  a scalar for one column).
- A molecule with no angles or dihedrals (a diatomic) crashed the vectorised
  descriptor (empty index arrays were 1-d).
- `detectana.onset` no longer needs pandas to import.

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
