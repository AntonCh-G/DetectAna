# Usage

Everything the [README](../README.md) leaves out: the full configuration
reference, the scripts, and what the pipeline writes.

For *why* the pipeline is built this way, see [methodology.md](methodology.md).
For environment setup, data requirements and determinism, see
[reproducibility.md](reproducibility.md).

## Install

```bash
uv venv .venv --python 3.11      # or python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Python 3.10–3.12. CPU only — no deep-learning framework is imported anywhere.

## Running on your own data

```bash
cp config/example.yaml config/local.yaml   # config/local.yaml is git-ignored
```

Fill in the paths, then:

```bash
python scripts/run_pipeline.py --config config/local.yaml
python scripts/run_pipeline.py --config config/local.yaml --verbose
python scripts/run_pipeline.py --config config/local.yaml --force-recompute
```

`--force-recompute` ignores the cached descriptors and rebuilds them from the XYZ
files. Only `config/demo.yaml` and `config/example.yaml` are tracked; anything
else you drop in `config/` is git-ignored, so local paths stay local.

Three input layouts work:

- i-PI XYZ, one file per bead (`bead_glob`) plus a centroid file (`centroid_xyz`)
- one HDF5 file holding all beads and the centroid (`hdf5`), via
  `scripts/run_pipeline_hdf5.py`
- classical MD: point `bead_glob` and `centroid_xyz` at the same trajectory. With
  one replica, that replica is the centroid.

## Configuration

One YAML file. [config/example.yaml](../config/example.yaml) is the annotated
template; the main sections are:

```yaml
reference:
  train: path/to/train.xyz
  valid: path/to/valid.xyz
  test:  path/to/test.xyz

runs:
  - name: run_name
    initial_xyz:  path/to/initial.xyz
    bead_glob:    "path/to/aspirin.pos_*.xyz"   # sorted → bead 00..15
    centroid_xyz: path/to/aspirin.xc.xyz
    timestep_fs: 0.2
    stride: 50

descriptor:
  pca_variance: 0.95   # fraction of variance retained
  random_seed: 42

threshold:
  percentile: 99.0     # of the validation-set Mahalanobis scores

chemistry:
  bond_break_cutoff: 2.0     # Å
  close_contact_cutoff: 1.2  # Å
  nl_mult: 1.1               # covalent-radius multiplier for the bond graph
  # ring_atoms: [0, 1, 2, 3, 4, 5]   # set when several rings are candidates

onset:
  window_frames: 500
  step_frames: 50
  fraction_threshold: 0.20     # how much of the window must be OOD
  frame_autocorrelation: "auto"
  # false_alarm_budget: 0.01   # recommended instead of fraction_threshold

pipeline:
  n_jobs: -1
```

Add another entry under `runs:` for a second run.

### Choosing the window rule

The `onset` block, not `threshold.percentile`, is what controls false alarms. A
threshold calibrated to flag 1 % of in-distribution frames flags about 1 % of any
long run as well: ~2000 frames out of 200,000, the first of them after ~100
frames. `first_bead_anomaly` in the output is therefore a property of the
threshold rather than of the trajectory — read it as a diagnostic, not an onset.
Only the windowed criteria carry evidence.

Every run logs what its rule costs:

```
Onset rule: 13/230 effective flags per 500-frame window (fraction 0.0565);
false-alarm bound 0.00306 per run (0.000306 counting disjoint windows only)
```

Two numbers, because the first counts every window start as a separate test even
though windows overlapping by 90 % are nearly the same test. The truth is between
them; the pipeline works from the pessimistic one.

A 500-frame window is not 500 independent chances to flag, because consecutive
frames are correlated. `frame_autocorrelation: "auto"` measures the lag-1
correlation on the first `stable_fraction` of the run (default 0.1) and converts
the window to an effective sample size. Setting it to `0` assumes independence and
makes the bound optimistic.

Set `false_alarm_budget` and the fraction is derived from it: the loosest, most
sensitive rule that keeps the run-level false-alarm probability inside the budget.
This is the recommended direction — you state the false-alarm rate you can live
with instead of guessing a fraction. For 200,000 frames, a 1 % threshold and a 1 %
budget:

| autocorrelation | derived fraction | flags needed | bound per run |
|---|---|---|---|
| 0 (assumed independent) | 0.038 | 19 of 500 | 0.005 |
| 0.4 (illustrative) | 0.057 | 12 of 214 | 0.003 |

Compare the default `fraction_threshold: 0.20`, which needs 100 of 500 frames
flagged, for a bound near 10⁻⁹¹. That is safe to the point of being insensitive:
it will miss a real but partial excursion. Defaults are unchanged, so nothing
moves unless you set the budget, but the logged bound tells you where you stand.

The arithmetic treats flags inside a window as Bernoulli(α), corrects for frame
correlation with an effective sample size, and unions over overlapping windows, so
the reported probability is an upper bound. Beads count as one observation per
timestep by default (`n_effective_beads: 1`), because beads are path-integral
images of the same molecule and far from independent.

## Parallelism

The per-bead loop is parallelised with `joblib`:

```yaml
pipeline:
  n_jobs: -1   # all cores, the default
  # n_jobs: 1  # serial, easier to debug on a laptop
```

On SLURM, `-1` follows `--cpus-per-task` on its own. Do not set it higher than the
cores you were allocated; oversubscribing makes it slower, not faster.

```bash
#SBATCH --cpus-per-task=16   # joblib spawns 16 workers, one per bead
```

## What comes out

```
outputs/
  onset_summary.csv                # one row per run
  models/
    descriptor_pipeline.pkl        # fitted scaler + PCA
    scorer.pkl                     # fitted Mahalanobis scorer
  <run_name>/
    manifest.json                  # config, version, threshold, onset design
    onset_table.csv                # run-level onset summary
    bead_scores.npy                # (n_beads, n_frames)
    centroid_scores.npy            # (n_frames,)
    frame_aggregate.csv            # aggregated scores per timestep
    chemistry_flags_bead00.csv     # hard-chemistry flags, bead 00
    descriptor_cache/              # raw descriptor NPZ caches, per bead
    plots/
      score_vs_time.png            # bead max/p95, fraction OOD, centroid
```

## Scripts

### Frames around the onset

Grabs N frames before and M frames after the onset, from the bead that crossed
the threshold first:

```bash
python scripts/extract_onset_frames.py \
    --config config/local.yaml \
    --run run_name \
    --n-before 100 \
    --n-after 100
```

Writes `outputs/<run>/extraction_bead<NN>_frame<FFFF>_N<N>_M<M>.xyz`.

`--onset-type` picks the frame to centre on:

- `persistent` (default) — the first window where the fraction of OOD beads
  exceeds the threshold. This is the onset the rest of the pipeline means.
- `first` — the first single frame any bead went over. On a long run this is
  almost always a false flag produced by the threshold itself.

### Selecting a diverse set of configurations

`select_configurations.py` selects N frames from a trajectory that are far apart in
descriptor space while staying inside a radius you set. The point is a spread of
geometries rather than N nearly identical snapshots — a candidate set to label and
train on.

Two modes, chosen by whether you pass `--primary-dihedrals`.

**Primary-dihedral mode.** `--radius` is a constraint, not a target. Only frames
whose chosen dihedrals are close to the reference value survive it. Among those,
the N frames are picked by farthest-point sampling in everything *except* those
dihedrals: bonds, angles, the remaining torsions, ring planarity. So you hold one
or two dihedrals roughly fixed and let everything else vary as much as possible.
For aspirin, the carboxyl and ester dihedrals are the two usually worth
constraining.

One radius for all of them (a circle):

```bash
python scripts/select_configurations.py \
    --reference path/to/initial.xyz \
    --trajectory path/to/aspirin.xc.xyz \
    --radius 0.2 \
    --n-configs 50 \
    --output outputs/selected.xyz \
    --pimd \
    --primary-dihedrals 5 6 12 11
```

One radius per dihedral (an ellipse), in the same order as the flags. A frame
passes when `sum((dᵢ/Rᵢ)²) ≤ 1`, where `dᵢ` is the chord distance in sin/cos space
for dihedral `i`. The `descriptor_distance` written to the output is
`sqrt(sum((dᵢ/Rᵢ)²))`, so anything at or below 1 is inside the ellipse.

```bash
# carboxyl held tight (±6°), ester left loose (±23°)
python scripts/select_configurations.py \
    --reference path/to/initial.xyz \
    --trajectory path/to/aspirin.xc.xyz \
    --radius 0.10 0.40 \
    --n-configs 50 \
    --output outputs/selected_ester_scan.xyz \
    --pimd \
    --primary-dihedrals 6 5 10 7 \
    --primary-dihedrals 5 6 12 11
```

*What a radius means in degrees.* Two angles differing by Δθ are `2·sin(Δθ/2)`
apart in (sin θ, cos θ) space. That holds per dihedral in both modes.

```python
import numpy as np
radius = 2 * np.sin(np.radians(delta_deg) / 2)     # angle → radius
delta_deg = np.degrees(2 * np.arcsin(radius / 2))  # radius → angle
```

| radius per dihedral | ±Δθ |
|---|---|
| 0.10 | ±5.7° |
| 0.20 | ±11.5° |
| 0.35 | ±20° |
| 0.52 | ±30° |
| 1.00 | ±60° |
| 2.00 | ±180° (everything) |

**Full-descriptor mode (the default).** Distances are measured in PCA-reduced
internal-coordinate space over all bonds, angles, dihedrals and ring planarity,
with the `DescriptorPipeline` fit on the trajectory itself. That PCA is not the one
the main pipeline fits on the reference set, so these distances are not comparable
to the OOD scores.

```bash
python scripts/select_configurations.py \
    --reference path/to/initial.xyz \
    --trajectory path/to/aspirin.xc.xyz \
    --radius 5.0 \
    --n-configs 50 \
    --output outputs/selected.xyz \
    --pimd
```

| Argument | Required | Description |
|---|---|---|
| `--reference` | yes | Single-frame XYZ used as the origin of the distance measurement |
| `--trajectory` | yes | MD or PIMD centroid trajectory file |
| `--radius` | yes | One value (circle), or one per `--primary-dihedrals` (ellipse). Full-descriptor mode takes one value only. |
| `--n-configs` | yes | How many configurations to select |
| `--output` | yes | Output extxyz path |
| `--pimd` | no | Pass for i-PI format files such as `aspirin.xc.xyz`; leave it off for extended-XYZ MD files |
| `--primary-dihedrals I J K L` | no | Constrain this dihedral to within `--radius` of its reference value and maximise diversity in the remaining coordinates. Repeat for more dihedrals. |
| `--pca-variance` | no | Full-descriptor mode only. Variance kept by the PCA (default `0.95`) |

The output is one extxyz file. Frame one is the reference (`source=reference`); the
rest are the selected frames, ordered by decreasing descriptor distance, each
carrying `source_frame`, `source_step` and `descriptor_distance`. What that
distance holds depends on the mode: Euclidean in PCA space (full-descriptor),
chord distance in the joint sin/cos space (one radius), or the ellipse distance
(several radii). If fewer frames than requested pass the constraint, the script
warns and writes the ones that did rather than failing.

### Benchmarking the detector

```bash
python scripts/benchmark_detector.py --config config/local.yaml
```

Writes `detection_benchmark.csv`/`.json`/`.png` and `torsion_coverage.csv` under
`outputs/<...>/benchmark/`. What it measures and why the positives are labelled by
training coverage is in [ADR 0004](adr/0004-detector-evaluation.md).

### Score against force-field error

```bash
python scripts/score_vs_error.py \
    --config config/local.yaml \
    --reference-xyz frames_with_reference_forces.xyz \
    --predicted predicted_forces.xyz
```

Needs the same frames twice — once with reference forces, once with the force
field's — and reports Spearman correlation, force error by score decile, and the
top-to-bottom decile ratio.

### Embeddings for the second track

```bash
python scripts/extract_embeddings.py \
    --config config/your_run.yaml \
    --checkpoint path/to/checkpoint.ckpt \
    --model-package your_mlff_package \
    --output-dir outputs/embeddings \
    --stride 10
```

Runs on a GPU node, imports your force field, and writes per-atom embeddings to
HDF5. DetectAna reads those files and never imports the model itself.

The force field is not a dependency of this project and is not named in it:
`--model-package` says which Python package to load it from. The adapter contract
is in the script's docstring — the package must expose model classes under
`<pkg>.modules.models` and `atoms_to_graph` under `<pkg>.data.utils`, and the
model must accept `return_descriptors=True` and return invariant per-atom
features. `--model-class` and `--feature-key` override the class name and the
output key when they cannot be inferred.

### Regenerating the demo data

```bash
python scripts/make_demo_data.py
```

Rebuilds `data/smoke/` from `data/smoke/initial.xyz`. Deterministic given the
seeds in the script.

## Tests

```bash
pytest tests/ -v                                    # 176 tests, about 11 s
pytest tests/ --cov=detectana --cov-fail-under=85
```

| File | Covers |
|---|---|
| `test_smoke.py` | The [scientific-rules](scientific-rules.md) checklist, executable |
| `test_units.py` | Modules one at a time: chemistry, serialisation, aggregation, onset |
| `test_io.py` | XYZ and HDF5 loaders, frame validation, step numbering |
| `test_pipeline.py` | End-to-end demo run, output schemas, manifest |
| `test_pipeline_embedding.py` | The embedding track through the pipeline |
| `test_scorer_tracks.py` | Both scorers and their threshold calibration |
| `test_molecule_generality.py` | Nothing is tied to aspirin |
| `test_onset_design.py` | False-alarm arithmetic |
| `test_evaluation.py` | Detection metrics and distortion invariants |

Everything runs on `data/smoke/`, so nothing external is needed. CI runs the suite
on Python 3.10, 3.11 and 3.12, plus the demo pipeline and the benchmark end to end.

## Layout

```
src/detectana/
  io.py               XYZ and HDF5 loaders, MoleculeSpec frame validation
  xyz_reader.py       Byte-offset-indexed reader for multi-GB XYZ files
  topology.py         Bond graph, angles, dihedrals, ring, hard-chemistry checks
  descriptors.py      Internal-coordinate fingerprint + StandardScaler + PCA
  scorer.py           Mahalanobis scorer and threshold calibration
  embedding_scorer.py Per-atom Mahalanobis scorer in MLFF embedding space
  aggregator.py       Bead-score aggregation per timestep
  onset.py            Windowed fraction onset detector + false-alarm arithmetic
  evaluation.py       Detection metrics, error correlation, controlled distortions
  pipeline.py         The orchestrator
scripts/
  run_pipeline.py             Main pipeline, XYZ or HDF5 input
  run_pipeline_hdf5.py        Deprecated wrapper, forwards to the library
  extract_onset_frames.py     Frames around the onset
  select_configurations.py    Training-set selection in descriptor space
  benchmark_detector.py       Detector benchmark against known distortions
  score_vs_error.py           OOD score against force-field error
  extract_embeddings.py       MLFF per-atom embeddings to HDF5
  make_demo_data.py           Regenerates data/smoke/
config/
  demo.yaml       Runs on data/smoke/ as-is
  example.yaml    Annotated template for real data
data/smoke/       Synthetic demo dataset, generated by make_demo_data.py
docs/
  methodology.md        Problem → assumptions → method → evaluation → interpretation
  reproducibility.md    Environment, data requirements, commands, determinism
  usage.md              This file
  scientific-rules.md   The constraints the pipeline is built around
  glossary.md           Terms, units, array shapes
  adr/                  Decision records
```
