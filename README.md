# DetectAna

[![CI](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

DetectAna finds the frame where an MD or PIMD trajectory leaves the region its
machine-learned force field was trained on. That frame is where you cut the run,
and the structures around it are the ones worth recomputing at a higher level of
theory and adding to the training set.

A force field is only reliable near its training data. A long simulation drifts
away from it and the model keeps returning plausible forces anyway — nothing
crashes, and you notice much later, when a bond has visibly stretched. DetectAna
scores every frame against the reference training distribution and reports the
onset with a false-alarm rate you state in advance rather than discover
afterwards.

Python 3.10–3.12, CPU only. The force field stays external.

## Install

```bash
git clone https://github.com/AntonCh-G/DetectAna.git
cd DetectAna

uv venv .venv --python 3.11      # or: python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Use

Run the whole pipeline on the demo data that ships with the repository:

```bash
python scripts/run_pipeline.py --config config/demo.yaml
```

Seconds, and writes to `outputs/demo/`: per-frame scores, a per-timestep
aggregate, the onset table, a score-against-time plot, and a `manifest.json`
recording the config, the calibrated threshold and the false-alarm arithmetic
behind the result.

Check the install:

```bash
pytest tests/ -v        # 176 tests
```

For your own data, copy the annotated template and fill in the paths:

```bash
cp config/example.yaml config/local.yaml     # config/local.yaml is git-ignored
python scripts/run_pipeline.py --config config/local.yaml
```

You need three things: an `initial.xyz` defining the molecule, the reference
`train`/`valid`/`test` split the force field was fitted on, and the trajectory —
as i-PI XYZ (one file per bead plus a centroid), as a single HDF5 file, or as one
extended-XYZ file for classical MD. Every key is documented in
**[docs/usage.md](docs/usage.md)**; data requirements and determinism are in
**[docs/reproducibility.md](docs/reproducibility.md)**.

Other entry points, all with `--help`:

| Script | Does |
|---|---|
| `scripts/benchmark_detector.py` | Measures the detector against distortions of known size |
| `scripts/extract_onset_frames.py` | Pulls the frames around a detected onset |
| `scripts/select_configurations.py` | Picks a diverse set of configurations for retraining |
| `scripts/score_vs_error.py` | Relates the OOD score to force-field error |
| `scripts/extract_embeddings.py` | Force-field per-atom embeddings → HDF5 (GPU, optional track) |
| `scripts/make_demo_data.py` | Regenerates `data/smoke/` |

## How it works

1. **Topology** from the run's `initial.xyz` — bonds, angles, dihedrals, optional
   ring. Every frame of every file is validated against it for atom count and
   element order; a mismatch is a hard failure, because internal coordinates are
   index tuples and a reordered file gives silently wrong descriptors.
2. **Descriptor**: bond lengths, bond angles, every dihedral as a $(\sin, \cos)$
   pair so the $0 \to 2\pi$ wrap is continuous, plus ring planarity.
3. **Fit on training frames only** — standardise, PCA, then a Mahalanobis
   distance from the training mean and covariance.
4. **Calibrate on held-out validation frames.** The threshold is a percentile of
   validation scores, so "flagged" means "above 99 % of reference frames the fit
   never saw". Nothing from the trajectory touches the fit or the threshold.
5. **Aggregate across beads without averaging** — per timestep: max, 95th
   percentile, and fraction above threshold. A single bead can leave the training
   region long before the centroid notices, and a mean erases that.
6. **Detect onset with a window rule.** A threshold that flags a fraction α of
   in-distribution frames flags ~α of *any* long run, so the first flagged frame
   says more about the threshold than the trajectory. The detector is the fraction
   of a window that must be flagged; under the null that count is binomial over an
   autocorrelation-corrected effective sample size. State a `false_alarm_budget`
   and the most sensitive rule inside it is derived for you.
7. **Hard-chemistry checks kept separate** — broken bonds, close contacts, ring
   planarity, in their own file, because "out of distribution" and "unphysical"
   are different claims.

The full treatment, with equations and the reasoning behind each choice, is in
**[docs/methodology.md](docs/methodology.md)**.

An optional second track scores in the force field's own invariant embedding
space, which asks "has the model seen anything like this?" rather than "is this
structure unusual?" ([ADR 0002](docs/adr/0002-embedding-ood-track.md)).

Developed on aspirin, but not tied to it: the molecule comes from the input
geometry, and one with no ring simply drops the planarity feature
([ADR 0003](docs/adr/0003-molecule-agnostic-topology.md)).

## Validation status

The detector has been evaluated against distortions of known size, with synthetic
positives labelled by how much training data covers them rather than by
distortion size — a distortion landing inside the training distribution is not an
anomaly, and flagging it would be the error
([ADR 0004](docs/adr/0004-detector-evaluation.md)).

**The quantitative results are withheld.** They were measured on a reference
dataset belonging to work that is not yet published, so neither the numbers nor
the figures are in this repository, and they cannot be reproduced from a clone.
The evaluation code is here and runs; `data/smoke/` is synthetic and stands in
for the real reference data, so the benchmark reproduces the machinery and not
the result. Both will be published together with the associated work.

Separately, whether a high score predicts actual force-field error is **not yet
answered**. `scripts/score_vs_error.py` implements the test and needs forces
recomputed with the reference method. Until it runs, the score is a statement
about training coverage only, which is all this repository claims for it.

## Repository layout

```
src/detectana/     the library: io, topology, descriptors, scorer,
                   embedding_scorer, aggregator, onset, evaluation, pipeline
scripts/           thin CLI entry points; orchestration and plotting only
config/            demo.yaml (runs as-is) and example.yaml (annotated template)
data/smoke/        synthetic demo dataset, generated by make_demo_data.py
tests/             176 tests; test_smoke.py is the scientific checklist, executable
docs/              methodology, reproducibility, usage, scientific-rules,
                   glossary, adr/
```

## Limitations

- **Positions only.** Forces are loaded and unit-checked but never enter the
  descriptor.
- **Out of distribution is not unphysical.** A flag says the force field is
  extrapolating; the hard-chemistry flags answer the separate question.
- **Beads are not independent replicas.** They are path-integral images of one
  molecule, so bead scores are early warnings and the run is the unit you compare
  across models.
- **One gas-phase molecule.** No periodic boundaries, no second molecule, no
  reactions.
- **The covariance needs data.** Roughly ten training frames per retained PCA
  component is the floor; the benchmark warns below it, and the demo data is
  deliberately below it.
- **Global descriptor.** Local atomic-environment descriptors (SOAP, ACSF) would
  be more expressive; left out to avoid a heavy dependency
  ([ADR 0001](docs/adr/0001-ood-scoring-method.md)).

Next: validate the score against force-field error; compare the geometric and
embedding tracks on one benchmark; close the active-learning loop (detect, select,
recompute, retrain); add force-based descriptors.

**Status:** v0.3.0, in active development. Interfaces may still change. Scientific
definitions, thresholds and atom indexing will not change silently — see
[docs/scientific-rules.md](docs/scientific-rules.md).

## Documentation

- [docs/methodology.md](docs/methodology.md) — the full method, with equations
- [docs/reproducibility.md](docs/reproducibility.md) — environment, data, determinism
- [docs/usage.md](docs/usage.md) — configuration reference, every script, outputs
- [docs/scientific-rules.md](docs/scientific-rules.md) — constraints; read before changing anything
- [docs/glossary.md](docs/glossary.md) — terms, units, array shapes
- [docs/adr/](docs/adr/) — decision records, including rejected alternatives
- [CHANGELOG.md](CHANGELOG.md) — what changed, with scientific changes called out

## License and citation

MIT, see [LICENSE](LICENSE). Citation metadata in [CITATION.cff](CITATION.cff).
