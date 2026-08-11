# DetectAna

[![CI](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Finding the frame where a molecular-dynamics trajectory leaves the region its
machine-learned force field was trained on — with a false-alarm rate stated in
advance rather than discovered afterwards.**

A machine-learned force field (MLFF) is only reliable near its training data. A
long simulation drifts away from it and the model keeps returning smooth,
plausible forces anyway. Nothing crashes. You notice much later, when a bond has
visibly stretched, and by then an unknown part of the run is unusable.

DetectAna scores every frame by how far it sits outside the force field's
reference training distribution, then decides where the run went out of
distribution using a **window rule whose false-alarm probability is derived from
a budget you state**. That frame is where you cut the run; the structures around
it are the ones worth recomputing at a higher level of theory and adding to the
training set.

**Method.** Internal-coordinate fingerprint → standardise → PCA → Mahalanobis
distance, fitted on reference *training* frames only, thresholded at a percentile
of held-out *validation* frames, aggregated across path-integral beads, then a
windowed onset rule with a binomial false-alarm bound corrected for frame
autocorrelation.

**Key measured result.** On the aspirin reference set used during development,
the detector fires in proportion to how little training data supports a
conformer: **Spearman −0.93** between training density and flag rate, **100 %**
detection in torsion regions the training set never visits against **0.3–4 %** in
well-sampled ones, at a threshold calibrated to flag 1 %. Bond stretches are
caught at **0.3–0.5 Å**, where a hard 2.0 Å "broken bond" check is still silent.
That reference set is not redistributable, so those numbers are **not**
reproducible from a clone — see [Results](#results).

**Stack.** Python 3.10–3.12 · NumPy · SciPy · scikit-learn · pandas · ASE · h5py
· joblib · matplotlib · YAML-configured pipelines · pytest (176 tests, ~94 %
coverage) · ruff · mypy · GitHub Actions on three Python versions. CPU only: the
force field stays external.

**Status:** v0.3.0, in active development. Interfaces may still change.
Scientific definitions, thresholds and atom indexing will not change silently —
see [docs/scientific-rules.md](docs/scientific-rules.md).

---

## Overview

Given (a) the reference dataset an MLFF was fitted on and (b) a trajectory that
MLFF produced, DetectAna produces a per-frame novelty score, a per-timestep
aggregate across path-integral beads, and a run-level onset table with four
separately-reported criteria — plus a manifest recording the threshold, the
topology and the false-alarm arithmetic behind every number.

It handles path-integral MD (one trajectory per bead plus a centroid) and
classical MD, from i-PI XYZ or HDF5 input. Developed on aspirin, but nothing is
tied to it: the molecule comes from the run's `initial.xyz`, and a molecule with
no ring simply drops the planarity feature
([ADR 0003](docs/adr/0003-molecule-agnostic-topology.md)).

An optional second track scores in the force field's own embedding space, which
asks "has the model seen anything like this?" rather than "is this structure
unusual?" ([ADR 0002](docs/adr/0002-embedding-ood-track.md)).

## Research question

> Given a trajectory and the reference set its force field was fitted on, can we
> identify the point beyond which the force field is extrapolating, with a
> false-alarm rate that is stated in advance rather than discovered afterwards?

Three requirements follow, and they drove every design decision:

1. **No leakage.** The score's fit and its threshold must come from reference
   data only. Calibrating on the trajectory would define anomalies relative to
   the anomalous run and suppress exactly the signal being looked for.
2. **A computable false-alarm rate.** The decision rule needs a probability
   under the null hypothesis that the whole run is in distribution — not a
   threshold that "looked reasonable on the plot".
3. **Honest levels of aggregation.** Bead, centroid and run are different
   physical statements and are reported separately. PIMD beads are
   path-integral images of one molecule, not independent replicas.

Deliberately out of scope: predicting the force error directly (that needs
reference-quality forces — see [Limitations](#limitations-and-future-work)), and
judging whether a structure is chemically possible (a separate, non-statistical
check, reported in its own file).

## Methodology

Full treatment, with equations and the reasoning behind each choice:
**[docs/methodology.md](docs/methodology.md)**. In short:

1. **Topology from the input geometry.** Bonds from covalent-radius cutoffs,
   then angles, dihedrals and an optional ring. Every frame of every file —
   including the reference set — is validated against `initial.xyz` for atom
   count and element order. A mismatch is a hard failure, because internal
   coordinates are index tuples and a reordered file gives silently wrong
   descriptors.
2. **Internal-coordinate fingerprint.** Bond lengths, bond angles, every
   dihedral as a $(\sin, \cos)$ pair, ring planarity RMSD. 134 columns for
   aspirin. Torsions are sin/cos encoded so the $0 \to 2\pi$ wrap is continuous;
   internal coordinates make a high score traceable to the bond, angle or
   torsion that caused it.
3. **Fit on training frames only.** Standardise, then PCA at 95 % retained
   variance (for conditioning — the raw covariance is near-singular), then a
   Mahalanobis distance from the training mean and covariance.
4. **Calibrate on held-out validation frames.** The threshold is a percentile
   (default 99) of validation scores, so "flagged" means "above 99 % of
   reference frames the fit never saw". Define $\alpha = 1 -$ percentile$/100$.
5. **Score beads and centroid; aggregate without averaging.** Per timestep: max,
   95th percentile, and fraction of beads above threshold. Never the mean — a
   single bead can leave the training region long before the centroid notices.
6. **Detect onset with a window rule, and bound its false-alarm rate.** A
   threshold that flags $\alpha$ of in-distribution frames flags ~$\alpha$ of
   *any* long run, so the first flagged frame is a property of the threshold, not
   the trajectory. The real detector is the fraction of a window that must be
   flagged. Under the null, that count is binomial over an AR(1)-corrected
   effective sample size, unioned over windows — an upper bound, deliberately.
   Set `false_alarm_budget` and the loosest (most sensitive) rule inside that
   budget is derived for you and recorded in the manifest.
7. **Hard-chemistry checks, kept separate.** Broken bonds, close contacts and
   ring planarity are checked against fixed cutoffs and written to their own
   file, because "out of distribution" and "unphysical" are different claims.

Supporting documents: [scientific-rules.md](docs/scientific-rules.md) (the
constraints that must not break silently), [glossary.md](docs/glossary.md) (terms,
units, array shapes), [docs/adr/](docs/adr/) (four decision records, including the
alternatives that were rejected and why).

## Repository structure

```
src/detectana/           the library — typed, tested, no hard-coded paths
  io.py                  XYZ and HDF5 loaders, MoleculeSpec frame validation
  xyz_reader.py          byte-offset-indexed reader for multi-GB XYZ files
  topology.py            bond graph, angles, dihedrals, ring, chemistry checks
  descriptors.py         internal-coordinate fingerprint + scaler + PCA
  scorer.py              Mahalanobis scorer, per-track threshold calibration
  embedding_scorer.py    per-atom Mahalanobis in the force field's embedding space
  aggregator.py          bead-score aggregation per timestep
  onset.py               windowed onset detector + false-alarm arithmetic
  evaluation.py          detection metrics, controlled distortions, coverage
  pipeline.py            the orchestrator

scripts/                 thin CLI entry points, orchestration and plotting only
  run_pipeline.py            main pipeline (XYZ or HDF5 input)
  benchmark_detector.py      detector benchmark against known distortions
  score_vs_error.py          OOD score against force-field error
  select_configurations.py   training-set selection in descriptor space
  extract_onset_frames.py    frames around a detected onset
  extract_embeddings.py      force-field per-atom embeddings → HDF5 (GPU)
  make_demo_data.py          regenerates data/smoke/
  run_pipeline_hdf5.py       deprecated wrapper, forwards to the library

config/
  demo.yaml              runs on data/smoke/ as-is
  example.yaml           annotated template for real data
data/smoke/              synthetic demo dataset (generated, not simulation output)
tests/                   176 tests; test_smoke.py is the scientific checklist, executable
docs/
  methodology.md         problem → assumptions → method → evaluation → interpretation
  reproducibility.md     environment, data requirements, commands, determinism
  usage.md               full configuration reference, every script, outputs
  scientific-rules.md    the constraints the pipeline is built around
  glossary.md            terms, units, array shapes
  adr/                   decision records 0001–0004
  images/                figures used in this README
```

## Reproducing the work

```bash
git clone https://github.com/AntonCh-G/DetectAna.git
cd DetectAna

uv venv .venv --python 3.11      # or: python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

pytest tests/ -v                                            # 176 tests, ~8 s
python scripts/run_pipeline.py --config config/demo.yaml    # full pipeline
python scripts/benchmark_detector.py --config config/demo.yaml
```

The demo runs in seconds and writes scores, an onset table, plots and a manifest
to `outputs/demo/`. It is the same sequence CI runs on Python 3.10, 3.11 and 3.12.

`data/smoke/` is **synthetic** — thermal-style perturbations of one equilibrium
geometry, written by [scripts/make_demo_data.py](scripts/make_demo_data.py). It
exercises every code path and is deliberately too small to mean anything: 64
training frames against a 134-column descriptor is far below what the covariance
needs, so every trajectory frame comes out flagged. It stands in for the real
reference data, which cannot be redistributed until the work this code was
developed for is published; the intention is to swap it for real frames then.

For your own data, environment details and determinism guarantees:
**[docs/reproducibility.md](docs/reproducibility.md)**. Full configuration
reference and per-script documentation: **[docs/usage.md](docs/usage.md)**.

## Results

### Measured — detector behaviour against training coverage

A score that flags frames is not evidence until you know what it catches and what
it misses. [scripts/benchmark_detector.py](scripts/benchmark_detector.py)
measures the detector against distortions of known size, with every synthetic
positive **labelled by how much training data actually covers it**, not by how
large the distortion is.

![Detection rate against distortion size, and against training coverage](docs/images/detection_benchmark.png)

| Quantity | Value | Reading |
|---|---|---|
| Spearman ρ, training density vs flag rate | **−0.93** | The detector fires in proportion to missing training data |
| Detection rate, never-visited torsion slices | **100 %** | Sensitivity where the force field must extrapolate |
| Flag rate, well-sampled torsion slices | **0.3–4 %** | Specificity, against a threshold calibrated to flag 1 % |
| Bond stretch first caught | **0.3–0.5 Å** | The hard 2.0 Å "broken bond" check is still silent here |

The right-hand panel is the argument: where the training data is, the detector
stays silent; where it is absent, it fires.

**Provenance and limits.** Measured on a 2500-frame aspirin reference set used
during development. That set belongs to work that is not yet published, so it is
not redistributable and is not in this repository, and **these numbers cannot be
reproduced from a clone** — the shipped demo data reproduces the machinery, not
the result. The intention is to replace the synthetic demo data with real
reference frames once the associated work is published. Method and reasoning:
[ADR 0004](docs/adr/0004-detector-evaluation.md).

### Illustrative — one long production run

![Mahalanobis OOD score against time for a long PIMD run](docs/images/example_score_vs_time.png)

500 ns of aspirin, 16 beads. For the first ~340 ns the bead scores sit near the
threshold, then jump by roughly a factor of five and stay there. The dotted lines
mark the detected persistent bead onset (top) and centroid onset (bottom). The
middle panel is why bead scores are never averaged: individual beads cross the
threshold on and off long before anything happens collectively — a mean would
have looked flat.

This is a single trajectory shown to demonstrate output, not a controlled
experiment, and the trajectory is not in this repository.

### Computed — what a window rule costs

Derived from the false-alarm arithmetic in
[`onset.py`](src/detectana/onset.py), for 200 000 frames at α = 1 % under a 1 %
run-level budget:

| Assumption on frame autocorrelation ρ | Derived fraction | Flags needed | Bound per run |
|---|---|---|---|
| 0 (independence assumed) | 0.038 | 19 of 500 | 0.005 |
| 0.37 (measured) | 0.057 | 13 of 230 | 0.003 |
| — shipped default `fraction_threshold: 0.20` | 0.20 | 100 of 500 | ~10⁻⁹¹ |

The default is safe to the point of being insensitive: it can miss a real but
partial excursion. Stating a budget instead of guessing a fraction is the
recommended direction, and the bound is logged for every run either way.

### Preliminary / not yet measured

Whether a high score predicts actual force-field error is **not answered**. The
machinery exists ([`score_vs_error.py`](scripts/score_vs_error.py)) and is
validated on synthetic controls, but it needs forces recomputed with the
reference method for the frames analysed, which has not been run. Until then the
score is a statement about training coverage only, which is all this repository
claims for it.

## Technical highlights

- **Leakage-proof calibration, enforced by a test.** Scaler, PCA, mean and
  covariance from `train`; threshold from held-out `valid`; `test` to confirm the
  calibration held. `tests/test_smoke.py` asserts no trajectory frame reaches
  either step.
- **A detector with an analytic false-alarm bound.** Binomial window statistics
  over an AR(1)-corrected effective sample size, union-bounded over overlapping
  windows, invertible so you state a budget and get the most sensitive rule
  inside it. Both a conservative and an optimistic bound are reported, so the
  width of the approximation is visible instead of assumed.
- **An evaluation design that survived being wrong.** The first benchmark scored
  AUROC 0.52 by rotating a torsion the training set fully samples. Those frames
  were in distribution; flagging them would have been the error. Positives are
  now labelled by training coverage, and specificity and sensitivity are reported
  separately rather than pooled ([ADR 0004](docs/adr/0004-detector-evaluation.md)).
- **Periodicity handled in the representation.** Torsions enter as
  $(\sin\theta, \cos\theta)$ rather than as unwrapped angles, with the chord↔arc
  conversion documented for anyone setting a radius in that space.
- **Molecule-agnostic by construction.** The molecule comes from `initial.xyz`;
  a ring-less molecule drops one descriptor column instead of failing.
  `tests/test_molecule_generality.py` pins this down.
- **Scales to multi-GB trajectories.** Byte-offset frame indexing for O(1)
  random access, chunked streaming, NPZ descriptor caching so re-scoring is
  seconds instead of hours, and `joblib` parallelism over beads that respects a
  SLURM allocation.
- **Reproducible by default.** Fixed seeds, YAML-configured everything, no
  hard-coded paths, and a `manifest.json` per run holding config, code version,
  threshold, topology and the full onset design report.
- **Engineering hygiene.** 176 tests at ~94 % coverage with an 85 % floor
  enforced in CI, `ruff` and `mypy` clean across the repository, a PEP 561
  `py.typed` marker, three-version test matrix, and the demo pipeline plus
  benchmark run end to end on every push.
- **Documented decisions, including rejected ones.** Four ADRs record what was
  chosen, what was not, and why — including a mistake that changed the
  evaluation design.

## Limitations and future work

**Limitations**

- **Positions only.** Forces are loaded and unit-checked (i-PI writes
  Hartree/Bohr, reference sets eV/Å, factor 51.4221) but never enter the
  descriptor. A force-based descriptor would likely catch some failures earlier.
- **Out of distribution is not unphysical.** A flag says the force field is
  extrapolating. The benchmark demonstrates this directly: a torsion slice
  holding 5 of 2500 training frames is flagged every time, and those conformers
  are chemically unremarkable.
- **The reliability claim is unvalidated.** See
  [Preliminary](#preliminary--not-yet-measured).
- **The two tracks have not been compared.** Geometric and embedding scores are
  reported side by side but never benchmarked against each other.
- **Global descriptor.** Local atomic-environment descriptors (SOAP, ACSF) would
  be more expressive; left out to avoid a heavy dependency
  ([ADR 0001](docs/adr/0001-ood-scoring-method.md)).
- **The covariance needs data.** Roughly ten training frames per retained PCA
  component is the floor; the benchmark warns below it.
- **Beads are not independent replicas.** They are path-integral images of one
  molecule, so bead scores are early warnings and the run is the unit you compare
  across models.

**Next**

- Validate the score against force-field error — the result that decides whether
  this is a reliability estimate or only a novelty measure.
- Compare the geometric and embedding tracks on one benchmark.
- Close the active-learning loop: detect, select, recompute, retrain.
- Add force-based and local atomic-environment descriptors.

**Out of scope by design:** periodic boundaries and multiple molecules (this is a
gas-phase single-molecule tool), and MLFF inference — the force field stays
external, which is what keeps the install CPU-only.

## Further reading

- [docs/methodology.md](docs/methodology.md) — the full method, with equations
- [docs/reproducibility.md](docs/reproducibility.md) — environment, data, determinism
- [docs/usage.md](docs/usage.md) — configuration reference, every script, outputs
- [docs/scientific-rules.md](docs/scientific-rules.md) — constraints; read before changing anything
- [docs/glossary.md](docs/glossary.md) — terms, units, array shapes
- [docs/adr/](docs/adr/) — decision records, including rejected alternatives
- [CHANGELOG.md](CHANGELOG.md) — what changed, with scientific changes called out

## License and citation

MIT, see [LICENSE](LICENSE). Citation metadata in
[CITATION.cff](CITATION.cff).
