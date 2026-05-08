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

## Descriptors

**Internal coordinate fingerprint** — the primary descriptor for OOD scoring.
Composed of: all pairwise bond lengths, bond angles, dihedral torsions (encoded
as sin/cos pairs to handle periodicity), and ring planarity deviation for the
benzene ring. Computed for every frame (bead or centroid). Feature vector is
standardized (zero mean, unit variance) using statistics from the training set
only.

**PCA projection** — dimensionality reduction applied to the standardized
internal coordinate fingerprint. Fit on training reference frames only. Used to
project bead/centroid frames for Mahalanobis OOD scoring.

**Mahalanobis distance** — primary OOD score. Computed in the PCA-reduced space
using the covariance of training projections. Higher = more out-of-distribution.

## Anomaly Detection

**OOD threshold** — 99th percentile of Mahalanobis distances computed on the
validation set frames. Fit on training set only; calibrated on held-out
validation set. No PIMD trajectory frames leak into threshold calibration unless
explicitly configured.

**OOD (out-of-distribution)** — a frame whose descriptor falls outside the
distribution of the reference training set. OOD does *not* automatically imply
chemically impossible; it means the MLFF is being asked to extrapolate.

**Chemical validity flag** — hard-threshold check on bond lengths, angles,
torsions, ring planarity, and close contacts. Separate from OOD score.
A frame can be OOD without failing chemical validity and vice versa.

**Anomaly onset** — the first time in a run where anomaly signals become
persistent (not a single-frame spike). Detected by a sliding window over bead-
aggregated scores: onset = first window where fraction of frames above OOD
threshold exceeds a configured fraction threshold. Window size, step size, and
fraction threshold are versioned config parameters.

## Aggregation Levels

| Level | Unit | Purpose |
|-------|------|---------|
| Bead | (run, bead, time) | Early-warning; quantum fluctuations |
| Centroid | (run, time) | Classical-limit anomaly signal |
| Run | scalar per run | Cross-model / cross-replica comparison |

Bead scores aggregated per timestep via: max, 95th-percentile, fraction above
threshold, and centroid score — never averaged away.
