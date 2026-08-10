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

The pipeline was developed on aspirin but is not tied to it. The molecule comes
from the run's `initial_xyz`: its atom count, element order and bond graph define
everything downstream, and every reference and trajectory frame is then checked
against that file. Point the config at another single molecule and it runs. A
molecule with no benzene-like ring works too — the ring-planarity feature and
flag are dropped, which makes the descriptor one column shorter, so the fit, the
threshold and the scored frames all have to come from the same topology.

Two limitations are still real. There is no periodic-boundary or multi-molecule
support: internal coordinates are computed on raw coordinates, so this is a
gas-phase single-molecule tool (a disconnected bond graph only earns a warning).
And with several candidate rings the auto-detection picks the lowest-indexed one,
so set `chemistry.ring_atoms` explicitly when the choice matters.

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

## How well does the detector work?

A score that flags frames is not evidence until you know what it catches and what
it misses, so [scripts/benchmark_detector.py](scripts/benchmark_detector.py)
measures it against distortions of a known size. Held-out reference frames are the
negatives; distorted copies of those frames are the positives. Nothing about the
distorted frames reaches the fit or the threshold.

![Detection rate against distortion size, and against training coverage](docs/images/detection_benchmark.png)

The right panel is the whole argument in one picture: grey bars are how much
training data sits at each torsion angle, the green line is how often the detector
flags a frame placed there. Where the training data is, it stays silent. Where the
training data is absent, it fires every time.

Measured on the MD17-derived aspirin reference set (2500 training frames, 600
held-out split into calibration and evaluation, 71 PCA components, α = 1 %):

| distortion | magnitude | detected | also caught by the chemistry flags |
|---|---|---|---|
| Gaussian rattle | σ = 0.05 Å | 44 % | 0 % |
| Gaussian rattle | σ = 0.10 Å | 100 % | 0 % |
| bond stretch | δ = 0.30 Å | 39 % | 0 % |
| bond stretch | δ = 0.50 Å | 100 % | 0 % |

The bond-stretch rows are the useful ones: at 0.3–0.5 Å the OOD score fires while
the hard-chemistry bond check is still silent, because its cutoff is 2.0 Å. The
statistical track sees a strained bond well before it looks broken.

**The result worth reporting is the torsion scan.** A rotatable torsion is driven
right around its circle and each target angle is labelled by how often the training
set visits it. Bond lengths and angles are untouched by construction — asserted in
[tests/test_evaluation.py](tests/test_evaluation.py), not assumed — so this
isolates conformational novelty:

| target angle for C4-C11-O12-C6 | training frames in that 30° slice | flagged |
|---|---|---|
| ±165° | ~1200 | 0.3 % |
| ±135° | 29–43 | 6–10 % |
| −105° to +75° | 0 | 100 % |

Spearman correlation between training density and flag rate: **−0.93**. The
detector flags a conformer in proportion to how little training data supports it,
which is the behaviour the definition of OOD demands. Note the +105° slice, which
holds 5 frames out of 2500 and is flagged 100 % of the time: 0.2 % coverage is not
coverage. That frame is chemically unremarkable, which is exactly why this
repository insists that out of distribution and unphysical are different claims.

An earlier version of this benchmark rotated whichever torsion split the molecule
most evenly and reported AUROC ≈ 0.52, which looked like a blind spot. It was not:
that torsion is fully sampled in the training set, so the rotated frames were
genuinely in distribution and flagging them would have been an error. Labelling
distorted frames as anomalies without checking coverage is a mistake worth
avoiding.

### Does a high score mean the force field is wrong there?

The question that decides whether the score is a usable reliability estimate.
[scripts/score_vs_error.py](scripts/score_vs_error.py) answers it given the same
frames twice — once with reference forces, once with the force field's — and
reports Spearman correlation, force error by score decile, and the top-to-bottom
decile ratio. It needs recomputed reference forces, so it is not part of the demo;
the machinery is validated on synthetic controls (no relationship → Spearman
−0.01; injected relationship → +0.23 with a 2.4× decile ratio).

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
- `first`: the first single frame any bead went over. On a long run this is
  almost always a false flag produced by the threshold itself — see
  [Choosing the window rule](#choosing-the-window-rule).

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
  frame_autocorrelation: "auto"
  # false_alarm_budget: 0.01  # recommended instead of fraction_threshold
```

Add another entry under `runs:` for a second run.

### Choosing the window rule

The `onset` block, not `threshold.percentile`, is what controls false alarms. A
threshold calibrated to flag 1 % of in-distribution frames flags about 1 % of any
long run as well: ~2000 frames out of 200,000, the first of them after ~100
frames. `first_bead_anomaly` in the output is therefore a property of the
threshold rather than of the trajectory — read it as a diagnostic, not an onset.
Only the windowed criteria carry evidence.

Every run now logs and records what its rule costs, so this is visible rather
than implicit:

```
Onset rule: 13/230 effective flags per 500-frame window (fraction 0.0565);
false-alarm bound 0.00306 per run (0.000306 counting disjoint windows only)
```

Two numbers because the first counts every window start as a separate test, even
though windows that overlap by 90 % are nearly the same test. The truth is between
them; the pipeline works from the pessimistic one.

The 500-frame window is not 500 independent chances to flag, because consecutive
frames are correlated; `frame_autocorrelation: "auto"` measures the lag-1
correlation on the first `stable_fraction` of the run and converts the window to
an effective sample size. Setting it to 0 assumes independence and makes the
bound optimistic.

Set `false_alarm_budget` and the fraction is derived from it — the loosest, most
sensitive rule that keeps the run-level false-alarm probability inside the
budget. This is the recommended direction: you state the false-alarm rate you can
live with, rather than guessing a fraction. For 200,000 frames, a 1 % threshold
and a 1 % budget:

| autocorrelation | derived fraction | flags needed | bound per run |
|---|---|---|---|
| 0 (assumed independent) | 0.038 | 19 of 500 | 0.005 |
| 0.37 (measured) | 0.057 | 13 of 230 | 0.003 |

Compare the default `fraction_threshold: 0.20`, which needs 100 of 500 frames
flagged for a bound near 10⁻⁹¹. That is safe to the point of being insensitive: it
will miss a real but partial excursion. The defaults are unchanged, so nothing
moves unless you set the budget, but the logged bound tells you where you stand.

The arithmetic assumes flags inside a window are Bernoulli(α), corrects for
frame correlation with an effective sample size, and unions over overlapping
windows, so the reported probability is an upper bound. Beads count as one
observation per timestep by default (`n_effective_beads: 1`), because beads are
path-integral images of the same molecule and far from independent.

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
  io.py               XYZ and HDF5 loaders, MoleculeSpec frame validation
  topology.py         Bond graph, angles, dihedrals, ring, hard-chemistry checks
  descriptors.py      Internal-coordinate fingerprint + StandardScaler + PCA
  scorer.py           Mahalanobis scorer and threshold calibration
  embedding_scorer.py Per-atom Mahalanobis scorer in MLFF embedding space
  aggregator.py       Bead-score aggregation per timestep
  onset.py            Windowed fraction onset detector + false-alarm arithmetic
  evaluation.py       Detection metrics, error correlation, controlled distortions
  pipeline.py         The orchestrator
scripts/
  run_pipeline.py             Main pipeline, XYZ input
  run_pipeline_hdf5.py        Same pipeline, HDF5 input
  extract_onset_frames.py     Frames around the onset
  select_configurations.py    Training-set selection in descriptor space
  benchmark_detector.py       Detector benchmark against known distortions
  score_vs_error.py           OOD score against force-field error
  extract_embeddings.py       MlffModel inv_features to HDF5
config/
  demo.yaml       Runs on data/smoke/ as-is
  example.yaml    Annotated template for real data
data/smoke/       Small aspirin dataset for the tests and the demo
tests/
  test_smoke.py               Validation checklist
  test_units.py               Per-module unit tests
  test_molecule_generality.py Nothing is tied to aspirin
  test_onset_design.py        False-alarm arithmetic
  test_evaluation.py          Metrics and distortion invariants
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
- [docs/adr/0003-molecule-agnostic-topology.md](docs/adr/0003-molecule-agnostic-topology.md):
  why the molecule comes from `initial.xyz` and how the optional ring works.
- [docs/adr/0004-detector-evaluation.md](docs/adr/0004-detector-evaluation.md):
  how the detector is benchmarked, and why coverage-labelled torsions rather than
  arbitrary distortions.

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
