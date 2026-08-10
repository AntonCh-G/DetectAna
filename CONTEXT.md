# DetectAna — Domain Glossary

## Molecule

**Aspirin** (C9H8O4) — the molecule this project studies, 21 atoms. It is the
data, not a code constant: nothing in the package assumes it.

**MoleculeSpec** — the expected atom count and element order for a run, built
from the first run's `initial.xyz` (`io.MoleculeSpec`). Every reference frame,
every other run's initial geometry and every trajectory frame is validated
against it before any descriptor is computed. Trajectory formats without element
symbols (binary XYZ reader, HDF5) can only be checked on atom count.

**Ring** — the carbon ring used for the planarity feature and flag. Auto-detected
as a 6-membered all-carbon ring, or set explicitly via `chemistry.ring_atoms`
when there is more than one candidate. A molecule without one drops the feature,
which shortens the descriptor by one column, so fit, threshold and scored frames
must share a topology.

**Scope** — one gas-phase molecule. No periodic boundaries, no second molecule in
the frame; a disconnected bond graph is warned about, not handled.

## Simulation

**PIMD run** — a single ring-polymer molecular dynamics simulation. Produces one
trajectory per bead (positions + forces) and one centroid trajectory. The
independent statistical unit for inter-model comparison. Multiple runs
(potentially from different MLFFs) will be compared against the same reference
dataset.

**Bead** — one replica in the ring polymer. 16 beads per PIMD run in the current
dataset. Bead trajectories are *not* independent replicas; they are quantum
path-integral images of the same molecule. Bead-level anomaly scores are
early-warning signals, not standalone measurements.

**Centroid** — the mean position of all beads at each timestep. One centroid
trajectory per PIMD run. Files: `aspirin.xc.xyz` (positions),
`aspirin.fc.xyz` (forces).

**Stride** — output interval. Current dataset: every 50 timesteps (timestep =
0.2 fs → 10 fs between saved frames).

**Frame** — one saved snapshot. A single (step, bead) pair in the trajectory
files.

## Onset

**False-flag rate (α)** — the fraction of in-distribution frames the OOD
threshold flags by construction, `1 − threshold.percentile/100`. Not an error
rate: at α = 1 % a clean 200,000-frame run still gets ~2000 flagged frames.

**Window rule** — the real detector: onset is the first window in which the
flagged fraction reaches `fraction_threshold`. Bounds on its false-alarm rate come
from `onset.window_false_alarm_probability`; `onset.choose_fraction_threshold`
inverts it, turning a `false_alarm_budget` into the loosest fraction that fits.

**Effective trials** — independent flag opportunities in a window, `window ·
(1−ρ)/(1+ρ)` for lag-1 autocorrelation ρ, times `n_effective_beads` (default 1,
since beads are not independent). A 500-frame window at ρ = 0.37 carries ~230.

**Onset design report** — the block written to `manifest.json` recording α, the
window, the fraction actually used, ρ, effective trials, flags needed and the
false-alarm bounds per window and per run.

## Evaluation

**Detection rate** — fraction of known-abnormal frames whose score exceeds the
threshold, where the threshold is the conformal order statistic of the calibration
scores. Answers "what would a real run have flagged?", unlike AUROC, which is
threshold-free.

**Training coverage** — how many training frames occupy a slice of some coordinate,
usually a torsion's 30° bin. What makes a synthetic distortion a *labelled*
anomaly: a distortion into a well-covered slice is in-distribution, into an empty
slice it is not.

**Torsion scan** — driving one torsion around its full circle with `set_dihedral`
and scoring every target angle, each labelled by training coverage. Isolates
conformational novelty, since bond lengths and angles are preserved exactly.

**Score-error correlation** — Spearman correlation between OOD score and per-frame
force error, plus error by score decile. Decides whether the score is a usable
reliability estimate. Needs forces from the reference method, so it lives in
`scripts/score_vs_error.py` rather than in the pipeline.

## Data Files

**PIMD trajectory data (iPI XYZ format)** — `CCSD_newdata/`:
- `aspirin.pos_NN.xyz` — bead NN positions (Å), 16 files (NN = 00–15)
- `aspirin.for_NN.xyz` — bead NN forces (atomic units), 16 files
- `aspirin.xc.xyz` — centroid positions (Å)
- `aspirin.fc.xyz` — centroid forces (atomic units)
- `*.frameindex.npz` — byte-offset index for O(1) frame access
- ~200,000 frames per bead file (10,000,000 steps ÷ stride 50)

**PIMD trajectory data (HDF5 format)** — `*/hdf5/nvt_trajectory.hdf5`:
- Single file per run containing all beads and centroid together.
- `bead_positions` : `(n_frames, n_beads, n_atoms, 3)` float64 Å
- `positions`      : `(n_frames, n_atoms, 3)` float64 Å — centroid (= mean of beads)
- `potential`      : `(n_frames,)` float64 eV
- `bead_momenta`, `momenta` : momenta arrays (not used by pipeline currently)
- No forces in current files; `bead_forces` / `forces` datasets reserved for future versions.
- Steps are frame indices `0…n_frames−1`; no physical time metadata stored.
- Read via `load_pimd_trajectory_hdf5` → `PIMDTrajectory` namedtuple.

**Reference dataset** — `aspirin_md17_pimd_pbe_mbd_tight/`:
- `aspirin_md_pimd_pbe0_mbd_train2500.xyz` — 2500 training frames
- `aspirin_md_pimd_pbe0_mbd_valid600.xyz` — 600 validation frames
- `aspirin_md_pimd_pbe0_mbd_test400.xyz` — 400 test frames
- Format: extended XYZ (positions Å, forces eV/Å, energy eV)

**Unit mismatch** — PIMD forces in atomic units (Hartree/Bohr); reference forces
in eV/Å. Conversion required: 1 Hartree/Bohr ≈ 51.4221 eV/Å.

## MLFF

**MLFF (machine learning force field)** — a neural network trained to predict
energies and forces from atomic positions. Produces the PIMD trajectories
DetectAna analyses. The MLFF is an external system; DetectAna does not load or
call it during the main pipeline. One trained MLFF per quantum-chemical method
(PBE0, CCSD, RPA, VMC, VD); multiple PIMD runs with the same method share one
checkpoint.

**MlffModel** — the specific MLFF architecture used in this project. A message-
passing equivariant graph neural network with SO(3)-symmetric transformer blocks.
Relevant output: `inv_features`, the invariant per-atom embedding after the last
transformer layer. Shape `(n_atoms, n_features)`, e.g. `(21, 128)`. Extracted
via `return_descriptors=True` in the forward pass. Lives in a separate codebase
(`mlff_torch`); DetectAna never imports it.

**Atomic embedding** — the `inv_features` tensor produced by MlffModel for one
frame. Rotation-invariant; encodes the local chemical environment of each atom
as learned by the MLFF during training. Used as the basis for the embedding OOD
track.

**Pre-computed embeddings** — MlffModel inference results written to HDF5 files
before DetectAna runs. One HDF5 file per bead trajectory (`embedding_glob`),
one for the centroid (`centroid_embedding_h5`), one for the reference training
set. DetectAna reads these files; it does not trigger MLFF inference.

## Descriptors

**Internal coordinate fingerprint** — the primary descriptor for the geometric
OOD track.
Composed of: all pairwise bond lengths, bond angles, dihedral torsions (encoded
as sin/cos pairs to handle periodicity), and ring planarity deviation for the
benzene ring. Computed for every frame (bead or centroid). Feature vector is
standardized (zero mean, unit variance) using statistics from the training set
only.

**PCA projection** — dimensionality reduction applied to the standardized
internal coordinate fingerprint. Fit on training reference frames only. Used to
project bead/centroid frames for Mahalanobis OOD scoring.

**Carboxyl dihedral** — dihedral C6–C5–C10–O7 (0-based atom indices [6, 5, 10, 7]).
Describes the out-of-plane rotation of the carboxyl group relative to the benzene
ring. Encoded as `sin_dih_C6-C5-C10-O7` and `cos_dih_C6-C5-C10-O7` in the
internal-coordinate fingerprint (dihedral index 32).

**Ester dihedral** — dihedral C5–C6–O12–C11 (0-based atom indices [5, 6, 12, 11]).
Describes the rotation around the ester oxygen bond. Encoded as
`sin_dih_C5-C6-O12-C11` and `cos_dih_C5-C6-O12-C11` in the
internal-coordinate fingerprint (dihedral index 35).

**Constrained diverse sampling** — configuration selection mode in
`select_configurations.py`. One or more dihedrals are designated *primary
dihedrals* (constraint); `--radius` is the maximum allowed distance in the
sin/cos space of those dihedrals from the reference value. `--radius` accepts
either one scalar (isotropic, circular constraint: same radius for all primary
dihedrals) or one scalar per primary dihedral (anisotropic, elliptic
constraint). Among the frames that pass the constraint, N configurations are
chosen by Farthest-Point Sampling in the *complementary* raw
internal-coordinate space (all bonds, angles, remaining dihedrals, ring
planarity — the primary dihedral sin/cos columns are excluded). Result:
structures with the constrained dihedral held near the reference value but
maximally diverse in all other conformational degrees of freedom.

**Elliptic dihedral constraint** — anisotropic variant of constrained diverse
sampling. Each primary dihedral i is assigned its own radius Rᵢ. A frame
passes the filter when `sum((dᵢ/Rᵢ)²) ≤ 1`, where dᵢ is the chord distance
in the sin/cos space of dihedral i. With a single shared radius this reduces
exactly to the isotropic (circular) constraint. The output `descriptor_distance`
field stores the normalized ellipse distance `sqrt(sum((dᵢ/Rᵢ)²))`; values
≤ 1 lie inside the ellipse.

**Descriptor space** — the PCA-reduced internal-coordinate space. The
coordinate system in which inter-frame distances are computed for configuration
selection and OOD scoring. A point in descriptor space is the PCA projection of
a single frame's standardized internal-coordinate fingerprint. When used for
configuration selection from a trajectory, the PCA is fit on the trajectory
itself; this space is distinct from the training-set-fitted descriptor space
used by the main OOD pipeline, and distances between the two are not
directly comparable.

**Mahalanobis distance** — OOD score used by both tracks. Computed from a fitted
mean and inverse covariance. Higher = more out-of-distribution. In the geometric
track: computed in PCA-reduced internal-coordinate space. In the embedding track:
computed per atom in the 128-dimensional `inv_features` space.

## Anomaly Detection

**OOD threshold** — 99th percentile of scores computed on the validation set.
Fit on training set only; calibrated on held-out validation set. No PIMD
trajectory frames leak into threshold calibration. One threshold per track
(geometric and embedding).

**OOD (out-of-distribution)** — a frame whose descriptor falls outside the
distribution of the reference training set. OOD does *not* automatically imply
chemically impossible; it means the MLFF is being asked to extrapolate.

**Chemical validity flag** — hard-threshold check on bond lengths, angles,
torsions, ring planarity, and close contacts. Separate from OOD score.
A frame can be OOD without failing chemical validity and vice versa.

**Geometric OOD score** — Mahalanobis distance in PCA-reduced internal-coordinate
space. Flags structural outliers: unusual bonds, angles, dihedrals, ring
planarity. One scalar per frame (bead or centroid).

**Embedding OOD score** — per-atom Mahalanobis distance in MlffModel `inv_features`
space. Flags model-reliability outliers: frames the MLFF has not encountered
during training, where energy/force predictions are likely extrapolating. One
scalar per atom per frame; aggregated to one scalar per frame via max over all 21
atoms. Computed at a configurable stride and optional frame range.

**Geometric onset** — anomaly onset derived from the geometric OOD track.

**Embedding onset** — anomaly onset derived from the embedding OOD track.
Comparable to geometric onset; the two are always shown side-by-side in the
output table. Represents the first moment the MLFF enters an extrapolation
regime, independent of whether the geometry is yet visibly anomalous.

**Anomaly onset** — generic term for either geometric or embedding onset. Detected
by a sliding window: onset = first window where fraction of frames above OOD
threshold exceeds a configured fraction threshold. Window size, step size, and
fraction threshold are versioned config parameters.

**Embedding stride** — frames between successive MlffModel inference calls.
Independent of the geometric pipeline stride. Configured per-run to manage GPU
inference cost.

**Frame range** — optional `[frame_start, frame_end)` window that restricts
embedding OOD computation to a subrange of the trajectory. Frames outside the
range have no embedding score. Useful for focusing expensive inference around a
region of interest (e.g., near the geometric onset).

## Aggregation Levels

| Level | Unit | Purpose |
|-------|------|---------|
| Bead | (run, bead, time) | Early-warning; quantum fluctuations |
| Centroid | (run, time) | Classical-limit anomaly signal |
| Run | scalar per run | Cross-model / cross-replica comparison |

Bead scores aggregated per timestep via: max, 95th-percentile, fraction above
threshold, and centroid score — never averaged away.
