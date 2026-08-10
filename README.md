# DetectAna

[![CI](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonCh-G/DetectAna/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

DetectAna finds the point in an MD or PIMD trajectory where a machine-learned
force field stops being supported by its training data.

## Why

A machine-learned force field (MLFF) is a neural network trained to predict
energies and forces. It is reliable only for structures similar to the ones it
was trained on. Long simulations drift away from those structures, and the model
keeps returning numbers anyway. Nothing crashes. You notice much later, when a
bond has visibly stretched or the molecule has fallen apart, and by then a large
part of the trajectory is already unusable.

I wrote this to get a specific number out of a run: the frame where it left the
training distribution. That is where you cut the trajectory, and it is also a
good place to look for new structures worth computing at a higher level of
theory and adding to the training set.

## Method

The pipeline does the following:

1. Read the reference train/valid/test sets (extended XYZ, MD17 style).
2. Build the molecular topology from `initial.xyz`: bonds, angles, dihedrals,
   benzene ring.
3. Turn every reference frame into a fingerprint of internal coordinates. Bond
   lengths, angles, torsions as sin/cos pairs, ring planarity.
4. Standardize and run PCA, fit on the training frames only. Fit a Mahalanobis
   distance on the same frames. Pick the threshold as the 99th percentile of the
   validation-set distances.
5. Score the trajectory frame by frame. For PIMD that means every bead plus the
   centroid; for classical MD, the single trajectory.
6. Collapse the bead scores at each timestep into max, 95th percentile, and the
   fraction of beads above threshold.
7. Slide a window over those and report onset at the first window where the
   fraction stays above the limit.
8. Write out the score arrays, an onset table, plots, and a manifest recording
   the config and the threshold.

Mahalanobis distance in a PCA space is a deliberately boring choice. It is
cheap, it has no hyperparameters worth tuning, and its output is one number per
frame that is easy to plot against time. The reasoning is written up in
[docs/adr/0001-ood-scoring-method.md](docs/adr/0001-ood-scoring-method.md).

There is a second, optional scoring track that works in the MLFF's own learned
embedding space (MlffModel `inv_features`) instead of in geometric coordinates.
It answers a slightly different question: not "is this structure unusual" but
"has the model seen anything like this". Both onsets end up in the same table.
See [docs/adr/0002-embedding-ood-track.md](docs/adr/0002-embedding-ood-track.md).

One limitation worth knowing before you clone: the descriptors are generic, but
the frame validation is not. Every frame is checked against aspirin, 21 atoms in
a fixed order. Another molecule means editing the atom-count and atom-type check
in [src/detectana/io.py](src/detectana/io.py) and rebuilding the topology.

## An example run

![Mahalanobis OOD score against time for a 500 ns PIMD run](docs/images/example_score_vs_time.png)

500 ns of aspirin, 16 beads. For the first 340 ns or so the bead scores sit
around the threshold (top panel). Then they jump by a factor of five and stay
there. The dotted line in the top panel is the detected persistent bead onset.
The one in the bottom panel is the centroid onset, which fires much earlier on a
brief excursion that the run recovers from.

The middle panel is the reason bead scores are never averaged. Individual beads
cross the threshold on and off throughout the first half of the run, long before
anything happens collectively. Average them and that signal disappears.

## Getting started

```bash
git clone https://github.com/AntonCh-G/DetectAna.git
cd DetectAna

uv venv .venv --python 3.11      # or python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

pytest tests/ -v
python scripts/run_pipeline.py --config config/demo.yaml
```

The demo runs in a few seconds on the small dataset in `data/smoke/` and writes
its results to `outputs/demo/`. It is there so you can check the installation
works and see the shape of the output. It is not a result: 64 training frames,
32 validation frames and a 24-frame trajectory are nowhere near enough, and
every frame comes out flagged.

## Your own data

```bash
cp config/example.yaml config/local.yaml   # config/local.yaml is git-ignored
```

Edit the paths in `config/local.yaml`, then:

```bash
python scripts/run_pipeline.py --config config/local.yaml

# more logging
python scripts/run_pipeline.py --config config/local.yaml --verbose

# ignore the cached descriptors and recompute from the XYZ files
python scripts/run_pipeline.py --config config/local.yaml --force-recompute
```

Three input layouts work:

- i-PI XYZ, one file per bead (`bead_glob`) plus a centroid file (`centroid_xyz`)
- one HDF5 file holding all beads and the centroid (`hdf5`)
- classical MD: point `bead_glob` at the single trajectory and `centroid_xyz` at
  the same file. With one replica, that replica is the centroid.

Only `config/demo.yaml` and `config/example.yaml` are tracked by git. Anything
else you put in `config/` is ignored, so local paths stay local.

### Pulling out frames around the onset

Once the pipeline has run, this grabs N frames before and M frames after the
onset, taken from the bead that crossed the threshold first:

```bash
python scripts/extract_onset_frames.py \
    --config config/local.yaml \
    --run run_name \
    --n-before 100 \
    --n-after 100
```

It writes `outputs/<run>/extraction_bead<NN>_frame<FFFF>_N<N>_M<M>.xyz`.

`--onset-type` picks which frame to centre on:
- `persistent` (default): the first window where the fraction of OOD beads
  exceeds the threshold. This is the onset the rest of the pipeline means.
- `first`: the first single frame any bead went over.

### Picking a diverse set of configurations

`select_configurations.py` selects N frames from a trajectory that are far from a
reference structure in descriptor space, while staying inside a radius you set.
The point is to get a spread of geometries rather than N nearly identical
snapshots, which is what you want for a candidate set to label and train on.

It has two modes, chosen by whether you pass `--primary-dihedrals`.

#### Primary-dihedral mode

Here `--radius` is a constraint, not a target. Only frames whose chosen
dihedrals are close to the reference value survive it. Among those, the N frames
are picked by farthest-point sampling in everything *except* those dihedrals:
bonds, angles, the remaining torsions, ring planarity.

So you hold one or two dihedrals roughly fixed and let everything else vary as
much as possible. For aspirin the carboxyl and ester dihedrals are the two
angles usually worth constraining.

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

One radius per dihedral (an ellipse). Pass the values in the same order as the
`--primary-dihedrals` flags. A frame passes when `sum((dᵢ/Rᵢ)²) ≤ 1`, where `dᵢ`
is the chord distance in sin/cos space for dihedral `i`. The
`descriptor_distance` written to the output is `sqrt(sum((dᵢ/Rᵢ)²))`, so
anything at or below 1 is inside the ellipse.

```bash
# carbonyl held tight (±6°), ester left loose (±23°)
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

`0.10` goes with the carboxyl dihedral (atoms 6 5 10 7), `0.40` with the ester
one (atoms 5 6 12 11).

##### What a radius means in degrees

Two angles that differ by Δθ are `2·sin(Δθ/2)` apart in (sin θ, cos θ) space.
That holds per dihedral in both modes.

```python
import numpy as np
radius = 2 * np.sin(np.radians(delta_deg) / 2)   # angle → radius
delta_deg = np.degrees(2 * np.arcsin(radius / 2)) # radius → angle
```

| radius per dihedral | ±Δθ |
|--------------------|------|
| 0.10 | ±5.7° |
| 0.20 | ±11.5° |
| 0.35 | ±20° |
| 0.52 | ±30° |
| 1.00 | ±60° |
| 2.00 | ±180° (everything) |

#### Full-descriptor mode (the default)

Distances are measured in PCA-reduced internal-coordinate space over all bonds,
angles, dihedrals and ring planarity, with the `DescriptorPipeline` fit on the
trajectory itself. Note that this PCA is not the one the main pipeline fits on
the reference set, so these distances are not comparable to the OOD scores.

```bash
python scripts/select_configurations.py \
    --reference path/to/initial.xyz \
    --trajectory path/to/aspirin.xc.xyz \
    --radius 5.0 \
    --n-configs 50 \
    --output outputs/selected.xyz \
    --pimd
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--reference` | yes | Single-frame XYZ used as the origin of the distance measurement |
| `--trajectory` | yes | MD or PIMD centroid trajectory file |
| `--radius` | yes | One value (circle), or one per `--primary-dihedrals` (ellipse). Full-descriptor mode takes one value only. |
| `--n-configs` | yes | How many configurations to select |
| `--output` | yes | Output extxyz path |
| `--pimd` | no | Pass for i-PI format files such as `aspirin.xc.xyz`; leave it off for extended-XYZ MD files |
| `--primary-dihedrals I J K L` | no | Constrain this dihedral to within `--radius` of its reference value, and maximise diversity in the remaining coordinates. Repeat for more dihedrals. |
| `--pca-variance` | no | Full-descriptor mode only. Variance kept by the PCA (default `0.95`) |

The output is one extxyz file. Frame one is the reference (`source=reference`).
The rest are the selected frames, ordered by decreasing descriptor distance,
each carrying:

```
source_frame=<int>  source_step=<int>  descriptor_distance=<float>
```

What `descriptor_distance` holds depends on the mode:
- full-descriptor: Euclidean distance in PCA space
- primary-dihedral, one radius: chord distance in the joint sin/cos space
- primary-dihedral, several radii: the ellipse distance `sqrt(sum((dᵢ/Rᵢ)²))`

If fewer frames than requested pass the constraint, the script warns and writes
the ones that did rather than failing.

## Configuration

Everything lives in one YAML file. [config/example.yaml](config/example.yaml) is
the annotated template; the main sections are:

```yaml
reference:
  train: path/to/train.xyz
  valid: path/to/valid.xyz
  test:  path/to/test.xyz

runs:
  - name: run_name
    initial_xyz: path/to/initial.xyz
    bead_glob:   "path/to/aspirin.pos_*.xyz"   # sorted → bead 00..15
    centroid_xyz: path/to/aspirin.xc.xyz
    timestep_fs: 0.2
    stride: 50

descriptor:
  pca_variance: 0.95   # fraction of variance retained
  random_seed: 42

threshold:
  percentile: 99.0     # of the validation-set Mahalanobis scores

onset:
  window_frames: 500
  step_frames: 50
  fraction_threshold: 0.20   # how much of the window must be OOD
```

Add another entry under `runs:` for a second run.

## Parallelism

The per-bead loop is parallelised with `joblib`:

```yaml
pipeline:
  n_jobs: -1   # all cores, the default
  # n_jobs: 1  # serial, easier to debug on a laptop
```

On SLURM, `-1` follows `--cpus-per-task` on its own. Do not set it higher than
the cores you were allocated; oversubscribing makes it slower, not faster.

```bash
#SBATCH --cpus-per-task=16   # joblib spawns 16 workers, one per bead
```

## What comes out

```
outputs/
  manifest.json                    # config, version, threshold
  onset_table.csv                  # run-level onset summary
  models/
    descriptor_pipeline.pkl        # fitted scaler + PCA
    scorer.pkl                     # fitted Mahalanobis scorer
  <run_name>/
    bead_scores.npy                # (n_beads, n_frames)
    centroid_scores.npy            # (n_frames,)
    frame_aggregate.csv            # aggregated scores per timestep
    chemistry_flags_bead00.csv     # hard-chemistry flags, bead 00
    descriptor_cache/              # raw descriptor NPZ caches, per bead
    plots/
      score_vs_time.png            # bead max/p95, fraction OOD, centroid
```

## Tests

```bash
pytest tests/ -v          # 33 tests, about 3 seconds
```

`tests/test_smoke.py` is the checklist from
[docs/scientific-rules.md](docs/scientific-rules.md), in executable form:

- topology: bond count, benzene ring found, feature names consistent
- descriptor shape, and no NaNs or infinities
- a frame with a broken bond scores higher than a normal one
- torsions survive the 0 ↔ 2π wrap thanks to the sin/cos encoding
- bead-stack shapes and the expected aggregate columns
- nothing outside the training set touches the PCA or the scorer fit

`tests/test_units.py` covers the modules one at a time: chemistry checks,
saving and loading the scorer and the descriptor pipeline, aggregation, onset
detection across a range of window scenarios, and the embedding pipeline.

Both use the small dataset in `data/smoke/`, so nothing external is needed. CI
runs them on Python 3.10, 3.11 and 3.12, and runs the demo pipeline end to end.

## Layout

```
src/detectana/
  io.py               XYZ and HDF5 loaders (chunked bead, reference, full trajectory)
  topology.py         Bond graph, angles, dihedrals, ring, hard-chemistry checks
  descriptors.py      Internal-coordinate fingerprint + StandardScaler + PCA
  scorer.py           Mahalanobis scorer and threshold calibration
  embedding_scorer.py Per-atom Mahalanobis scorer in MLFF embedding space
  aggregator.py       Bead-score aggregation per timestep
  onset.py            Windowed fraction onset detector
  pipeline.py         The orchestrator
scripts/
  run_pipeline.py             Main pipeline, XYZ input
  run_pipeline_hdf5.py        Same pipeline, HDF5 input
  extract_onset_frames.py     Frames around the onset
  select_configurations.py    Diverse configuration selection
  extract_embeddings.py       MlffModel inv_features to HDF5
config/
  demo.yaml       Runs on data/smoke/ as-is
  example.yaml    Annotated template for real data
data/smoke/       Small aspirin dataset for the tests and the demo
tests/
  test_smoke.py   Validation checklist
  test_units.py   Per-module unit tests
docs/
  scientific-rules.md   The constraints the pipeline is built around
  adr/                  Decision records
```

## Further reading

- [docs/scientific-rules.md](docs/scientific-rules.md) is the one to read if you
  plan to change anything. The constraints the pipeline is built around, why
  each exists, and the checklist a change has to pass.
- [CONTEXT.md](CONTEXT.md) is the glossary. What a bead, centroid, descriptor,
  threshold or onset means here, with units and array shapes.
- [docs/adr/0001-ood-scoring-method.md](docs/adr/0001-ood-scoring-method.md):
  why Mahalanobis distance on PCA-reduced internal coordinates.
- [docs/adr/0002-embedding-ood-track.md](docs/adr/0002-embedding-ood-track.md):
  why there is a second track in embedding space.

## Caveats worth repeating

Beads are not independent replicas. They are path-integral images of the same
molecule, so their scores are early warnings, not independent measurements. The
run is the unit you compare across models.

Out of distribution does not mean unphysical. A flagged frame says the MLFF is
extrapolating past what it was trained on. It does not say the structure is
chemically impossible, and the two are separate questions: the hard-chemistry
flags answer the second one.

The threshold comes from the validation set and never from the trajectory being
scored. That is deliberate. Calibrating on the trajectory would quietly hide
exactly the behaviour the pipeline is looking for.

Only positions go into the descriptor. Forces are read and validated but not
used for scoring yet.

## License

MIT, see [LICENSE](LICENSE).
