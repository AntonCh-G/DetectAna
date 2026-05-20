# DetectAna — Domain Glossary

## Molecule

**Aspirin** (C9H8O4) — the molecule under study. Always 21 atoms. Atom ordering
must be validated against `initial.xyz` before any descriptor calculation.

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

## Data Files

**PIMD trajectory data** — `CCSD_newdata/`:
- `aspirin.pos_NN.xyz` — bead NN positions (Å), 16 files (NN = 00–15)
- `aspirin.for_NN.xyz` — bead NN forces (atomic units), 16 files
- `aspirin.xc.xyz` — centroid positions (Å)
- `aspirin.fc.xyz` — centroid forces (atomic units)
- `*.frameindex.npz` — byte-offset index for O(1) frame access
- ~200,000 frames per bead file (10,000,000 steps ÷ stride 50)

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
