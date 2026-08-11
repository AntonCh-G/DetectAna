# Reproducibility

What can be reproduced from a clone, what cannot, and exactly how to run each
part. Commands here were run against this repository; the configuration
reference lives in [usage.md](usage.md) and the scientific reasoning in
[methodology.md](methodology.md).

## What is reproducible from a clone

| Item | Reproducible? | Why |
|---|---|---|
| The full test suite (176 tests) | Yes | Runs entirely on `data/smoke/` |
| The demo pipeline end to end | Yes | `config/demo.yaml` + `data/smoke/` |
| The detector benchmark *machinery* | Yes | Runs on the demo data; the numbers are meaningless there and it says so |
| `data/smoke/` itself | Yes | `scripts/make_demo_data.py`, deterministic given its seeds |
| The benchmark **numbers** in the README | **No** | Measured on a 2500-frame reference set that cannot be redistributed |
| The example score-vs-time figure | **No** | A 500 ns PIMD run; the trajectory is not in this repository |

Nothing in the repository presents an unreproducible number without saying so at
the point it appears.

The reason for the two "No" rows is the same in both cases: the reference dataset
and the production trajectories belong to work that is not yet published, so they
cannot be redistributed. The synthetic demo data stands in for them, and the
intention is to replace it with real reference frames once that work is
published — at which point the reported numbers become reproducible from a clone.

## Environment

Python 3.10–3.12, CPU only. No deep-learning framework is imported anywhere in
the package — the force field stays external, so the analysis install is small.

```bash
git clone https://github.com/AntonCh-G/DetectAna.git
cd DetectAna

uv venv .venv --python 3.11      # or: python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"       # or: pip install -e ".[dev]"
```

Runtime dependencies and their floors are declared in
[`pyproject.toml`](../pyproject.toml): `ase`, `numpy`, `scipy`, `scikit-learn`,
`pandas`, `matplotlib`, `pyyaml`, `joblib`, `h5py`. `uv.lock` pins the exact
versions the project was last developed against; installing from it rather than
from the floors is the way to reproduce bit-for-bit.

CI runs the suite on 3.10, 3.11 and 3.12, plus the demo pipeline and the
benchmark end to end ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)),
so "works on a fresh machine" is checked on every push rather than asserted.

## The shortest path to a result

```bash
pytest tests/ -v                                    # 176 tests, ~8 s
python scripts/run_pipeline.py --config config/demo.yaml
```

Writes to `outputs/demo/`, in seconds:

```
outputs/demo/
  onset_summary.csv            # one row per run
  models/
    descriptor_pipeline.pkl    # fitted scaler + PCA
    scorer.pkl                 # fitted Mahalanobis scorer
  demo/
    manifest.json              # config, version, threshold, topology, onset design
    onset_table.csv            # the four onset criteria
    bead_scores.npy            # (n_beads, n_frames)
    centroid_scores.npy        # (n_frames,)
    frame_aggregate.csv        # per-timestep max / p95 / fraction OOD / centroid
    chemistry_flags_bead00.csv # hard-chemistry flags, separate from the score
    descriptor_cache/          # per-bead descriptor NPZ
    plots/score_vs_time.png
```

Then the detector benchmark on the same data:

```bash
python scripts/benchmark_detector.py --config config/demo.yaml
```

Writes `detection_benchmark.{csv,json,png}` and `torsion_coverage.csv` under
`outputs/demo/benchmark/`.

**Read the demo output as a shape, not a result.** 64 synthetic training frames
against a 134-column descriptor is far below the ~10 frames per PCA component the
covariance needs, so every trajectory frame comes out flagged and the benchmark
prints a warning to that effect. It checks that the install works and shows what
the outputs look like.

To regenerate the demo data itself:

```bash
python scripts/make_demo_data.py
```

Deterministic given the seeds in the script; `data/smoke/initial.xyz` is the
input and is left alone.

## Running on your own data

### What you need

| File | Content | Notes |
|---|---|---|
| `initial.xyz` | One frame, the reference geometry | Defines atom count, element order and the bond graph for the whole run. Everything else is validated against it |
| Reference `train` / `valid` / `test` | Extended XYZ, disjoint splits of the set the force field was fitted on | `train` fits scaler/PCA/covariance, `valid` calibrates the threshold, `test` checks it. Positions in Å; forces (eV/Å) and energy (eV) optional for the geometric track |
| Trajectory | One of three layouts, below | Never enters the fit or the calibration |

Three accepted trajectory layouts:

1. **i-PI XYZ** — one file per bead (`bead_glob`, sorted → bead 00, 01, …) plus a
   centroid file (`centroid_xyz`).
2. **HDF5** — one file holding all beads and the centroid (`hdf5:`), read by
   `scripts/run_pipeline.py` directly; `scripts/run_pipeline_hdf5.py` is a
   deprecated wrapper kept for its two-threshold config schema.
3. **Classical MD** — point `bead_glob` and `centroid_xyz` at the same
   trajectory. With one replica, that replica is the centroid.

Array shapes and units for every format are tabulated in
[glossary.md](glossary.md).

Sizes to expect: a development run was ~200 000 frames per bead × 16 beads,
about 9 GB of XYZ text. Descriptor caching exists for exactly this reason.

### Where to put it

Anywhere. Paths are configuration, not code:

```bash
cp config/example.yaml config/local.yaml     # config/local.yaml is git-ignored
$EDITOR config/local.yaml                    # fill in your paths
python scripts/run_pipeline.py --config config/local.yaml
```

Only `config/demo.yaml` and `config/example.yaml` are tracked; everything else
under `config/` is git-ignored, so local absolute paths never reach the
repository. `outputs/` is git-ignored for the same reason.

No path is hard-coded anywhere in `src/detectana/`. If you find one, it is a bug.

## Determinism

| Source of variation | Handling |
|---|---|
| PCA | `descriptor.random_seed` (default 42) passed to `sklearn.decomposition.PCA(random_state=...)`; recorded in the manifest and in the pickled pipeline |
| Demo-data generation | `numpy.random.default_rng(seed)` with seeds fixed in `scripts/make_demo_data.py` |
| Threshold calibration | Deterministic — a percentile of the validation scores, no sampling |
| Mahalanobis fit | Deterministic — sample mean and covariance, plus a fixed `1e-8` ridge |
| Onset detection | Deterministic; the measured lag-1 autocorrelation is a function of the data |
| `joblib` parallelism (`pipeline.n_jobs`) | Affects speed only. Beads are scored independently and results are written per bead, so `n_jobs: -1` and `n_jobs: 1` give identical output |

Determinism is checked, not assumed: the 0.2.0 release verified that the demo
pipeline reproduces bit-identical `bead_scores.npy`, `centroid_scores.npy`,
`frame_aggregate.csv` and `onset_table.csv` against the previous release after
the topology was generalised ([../CHANGELOG.md](../CHANGELOG.md)).

Cross-version floating-point differences in BLAS or scikit-learn can move scores
in the last digits. Compare onsets and flag decisions across environments, not
raw score bytes.

## Experiment metadata

Every run writes `manifest.json` beside its results, holding:

- the fully resolved configuration, including every default that was applied;
- `detectana_version`;
- the calibrated threshold(s) per track, and the percentile behind each;
- the topology actually used — atom count, element order, ring indices — since
  descriptor length depends on it and the fit, the threshold and the scored
  frames must share one topology;
- the complete onset design report: α, window length, step, the fraction
  actually used, the measured frame autocorrelation, effective trials per
  window, flags needed, and the false-alarm bounds per window and per run (both
  the overlapping-window and disjoint-window variants).

A score array without the threshold that produced it is not interpretable later,
which is why this file is not optional.

## Parallelism and cluster runs

The per-bead loop is parallelised with `joblib`:

```yaml
pipeline:
  n_jobs: -1   # all cores (default)
  # n_jobs: 1  # serial, easier to debug
```

On SLURM, `-1` follows `--cpus-per-task` on its own; set
`#SBATCH --cpus-per-task=16` for 16 beads and leave `n_jobs: -1`.
Oversubscribing makes it slower, not faster.

## Verifying an installation

```bash
pytest tests/ -q                                       # 176 passed
pytest tests/ --cov=detectana --cov-fail-under=85      # currently ~94 %
ruff check .                                           # style
mypy                                                   # types, src/detectana
python scripts/run_pipeline.py --config config/demo.yaml
python scripts/benchmark_detector.py --config config/demo.yaml
```

That is the same sequence CI runs. `tests/test_smoke.py` is the executable form
of the [scientific-rules.md](scientific-rules.md) checklist — including the
assertion that no trajectory frame leaked into the fit or the calibration.

## What still cannot be reproduced, and what it would take

- **The benchmark numbers** (Spearman −0.93, 100 % detection in unvisited torsion
  slices) need the 2500-frame reference set, which is unpublished and cannot be
  shipped yet. Substituting your own reference set reproduces the *method* and
  gives numbers for your data, not these numbers.
- **The score-versus-force-error validation** has never been run.
  `scripts/score_vs_error.py` implements it and needs the same frames twice —
  once with reference-method forces, once with the force field's:

  ```bash
  python scripts/score_vs_error.py \
      --config config/local.yaml \
      --reference-xyz frames_with_reference_forces.xyz \
      --predicted predicted_forces.xyz
  ```

  It stays out of CI because it requires expensive reference calculations. Until
  it runs, the score is a statement about training coverage only.
- **The embedding track** needs pre-computed embeddings from the force field
  itself (`scripts/extract_embeddings.py`, GPU node, loads the package named by
  `--model-package`). DetectAna reads the resulting HDF5 files and never imports
  the model, so this step cannot be reproduced without that external codebase and
  a checkpoint. The architecture is part of the same unpublished work and is
  deliberately not named here; any model exposing invariant per-atom features
  satisfies the interface.
