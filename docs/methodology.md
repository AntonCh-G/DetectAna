# Methodology

How DetectAna decides that a trajectory has left its force field's training
distribution, and how that decision was tested. The chain is:

```
problem formulation → assumptions → inputs → descriptor → OOD statistic
    → threshold calibration → aggregation → onset rule → evaluation → interpretation
```

Each step below states what is computed, why that choice was made, and what the
result does *not* license you to claim. Decision records for the four
non-obvious choices are in [adr/](adr/); the constraints that must not be broken
silently are in [scientific-rules.md](scientific-rules.md); term definitions,
units and array shapes are in [glossary.md](glossary.md).

## 1. Problem formulation

A machine-learned force field (MLFF) predicts energies and forces from atomic
positions. It is fitted on a finite reference set, and it is only reliable in the
region of configuration space that set covers. Outside it the model does not
fail loudly: it returns smooth, plausible forces for structures it has never
seen, so the simulation keeps running and the trajectory keeps looking normal.
By the time an artefact is visible — a stretched bond, a broken ring — an
unknown fraction of the run is already unusable.

**Research question.** Given a trajectory and the reference set the force field
was fitted on, can we identify the point in the trajectory beyond which the
force field is extrapolating, with a false-alarm rate that is stated in advance
rather than discovered afterwards?

Formally: let $R$ be the reference training set and $x_t$ the configuration at
frame $t$. Define a scalar novelty score $s(x_t \mid R)$ and a decision rule
$D$ that maps the score series to an onset index $t^\*$. The requirements are:

1. $s$ is fitted on $R$ only; no frame of the trajectory being scored may
   influence the fit or the threshold.
2. $D$ has a computable false-alarm probability under the null hypothesis that
   the whole run is in distribution.
3. $t^\*$ is reported per bead, per centroid and per run separately, because
   these are different physical statements.

Two things this deliberately does *not* attempt: predicting the force error
directly (that needs reference-quality forces, see §9), and deciding whether a
structure is chemically possible (a separate, non-statistical check, §4.5).

## 2. Assumptions

| # | Assumption | Consequence if violated |
|---|---|---|
| A1 | One gas-phase molecule per frame, non-periodic | Internal coordinates are computed on raw positions with no minimum-image convention; a periodic cell or a second molecule gives meaningless distances |
| A2 | Constant atom count and element order within a run, matching `initial.xyz` | Internal coordinates are index tuples, so a reordered file produces wrong descriptors that nothing downstream can detect |
| A3 | Bond topology is fixed, taken from the initial geometry | A reaction (bond formed or broken permanently) is scored against the wrong coordinate set; it will score high, but the *coordinates* stop meaning what their names say |
| A4 | The reference set is the distribution of interest | The score measures distance from *this* set. Swapping the reference set changes every score; nothing about chemistry changes |
| A5 | Reference frames are exchangeable with in-distribution trajectory frames | Threshold calibration assumes this. A reference set sampled at a different temperature makes the calibrated α wrong |
| A6 | Under the null, per-frame flags are Bernoulli(α) with AR(1) correlation | The false-alarm bound in §7 rests on this; it is an approximation, made in the conservative direction |
| A7 | PIMD beads are *not* independent samples | They are path-integral images of one molecule. Treating 16 beads as 16 draws understates the false-alarm rate by orders of magnitude |

A1–A3 are enforced in code and fail loudly. A4–A7 are modelling assumptions and
are the ones to challenge when reading a result.

## 3. Inputs

Three separate inputs, and keeping them separate is the point:

| Input | Role | Format |
|---|---|---|
| `initial.xyz` | Defines the molecule: atom count, element order, bond graph | Single-frame XYZ |
| Reference set, split `train` / `valid` / `test` | `train` fits the statistics; `valid` calibrates the threshold; `test` checks the calibration held | Extended XYZ (positions Å, optional forces eV/Å, energy eV) |
| Trajectory | Scored, never fitted on | i-PI XYZ (one file per bead + a centroid file), or one HDF5 file with all beads, or a single extended-XYZ MD trajectory |

The reference set must be the one the force field was trained on, or a sample of
the same distribution. The quantum-chemical method behind it is a property of
the dataset, not of DetectAna: the pipeline never reads or records it, and
swapping reference sets is a configuration change.

The reference set used during development (2500 train / 600 validation / 400
test aspirin frames) is **not redistributable and is not in this repository**.
What ships instead is `data/smoke/` — 64 training, 32 validation and 24
trajectory frames generated synthetically from one equilibrium geometry by
[../scripts/make_demo_data.py](../scripts/make_demo_data.py). It exercises every
code path and is far too small to measure anything: 64 frames against a
134-column descriptor puts the covariance estimate deep in the noise-dominated
regime, and every trajectory frame comes out flagged.

### Preprocessing

There is deliberately very little, because every transformation is a chance to
leak information:

- **Frame validation.** Atom count and element order are checked against
  `initial.xyz` on every frame of every file, including the reference set. A
  mismatch is a hard failure, not a warning (`io.MoleculeSpec`). Formats that
  carry no element symbols — the fast binary XYZ reader and HDF5 — can only be
  checked on atom count, and files with mixed atom counts across frames are
  rejected.
- **Striding.** Trajectories are read at their stored output stride; the
  configured `timestep_fs × stride` converts frame index to picoseconds. Output
  tables carry both the raw step index and the time, because mixing the two
  produces a plot that looks entirely plausible.
- **No alignment, no centering, no rotation.** Internal coordinates are
  invariant to translation and rotation by construction, so there is nothing to
  align.
- **Descriptor caching.** Computed descriptors are written as NPZ per bead. XYZ
  parsing dominates the cost (a real run is ~9 GB of text); scoring, threshold
  changes and onset-rule changes then re-run in seconds against the cache.
  `--force-recompute` rebuilds it.

## 4. The geometric track

### 4.1 Topology

`topology.build_topology` reads the initial geometry and derives:

- **bonds** — from an ASE neighbour list with covalent-radius cutoffs scaled by
  `chemistry.nl_mult` (default 1.1);
- **angles** — every $i$–$j$–$k$ with $i,k$ both bonded to $j$, deduplicated;
- **dihedrals** — every $i$–$j$–$k$–$l$ across each bond $j$–$k$, canonicalised
  so $(i,j,k,l)$ and $(l,k,j,i)$ are one entry;
- **ring** — a 6-membered all-carbon ring, auto-detected, selectable via
  `chemistry.ring_atoms`, or absent.

A disconnected bond graph produces a warning, not support (A1). The molecule
comes from the input file rather than from constants in the code, which is what
makes the pipeline transferable to other molecules
([ADR 0003](adr/0003-molecule-agnostic-topology.md)).

### 4.2 Descriptor

Each frame becomes one fixed-length vector of internal coordinates, in this
column order — `topology.feature_names` is the authority:

| Block | Length | Unit |
|---|---|---|
| Bond lengths | $n_\text{bonds}$ | Å |
| Bond angles | $n_\text{angles}$ | rad |
| $\sin$ of every dihedral | $n_\text{dih}$ | — |
| $\cos$ of every dihedral | $n_\text{dih}$ | — |
| Ring planarity RMSD | 1, or 0 with no ring | Å |

For aspirin: $21 + 32 + 40 + 40 + 1 = 134$ columns.

Two design points matter:

**Torsions are encoded as $(\sin\theta, \cos\theta)$, not as $\theta$.** A
torsion is periodic on $[0, 2\pi)$, so $359°$ and $1°$ are neighbours. Any
distance computed on the raw angle makes them maximally far apart, and a
standardised angle is worse still. The sin/cos pair is continuous across the
wrap. The cost is that a distance in this space is a chord, not an arc: two
angles $\Delta\theta$ apart sit $2\sin(\Delta\theta/2)$ apart, which is the
conversion used by
[`select_configurations.py`](../scripts/select_configurations.py).

**Internal coordinates rather than Cartesians or SOAP.** Cartesian coordinates
are rotation-sensitive and would need alignment; SOAP or ACSF descriptors are
more expressive but add a heavy dependency, and were deferred
([ADR 0001](adr/0001-ood-scoring-method.md)). Internal coordinates are also
diagnosable: a high score can be traced back to the bond, angle or torsion that
produced it, which a kernel or tree-ensemble detector does not offer.

### 4.3 Standardisation and PCA

Both are fitted on the **training** reference frames only:

1. `StandardScaler` — zero mean, unit variance per column. Necessary because
   the blocks have incomparable units and wildly different variances (a bond
   length varies by ~0.05 Å, a torsion sine by ~1).
2. `PCA` — retain `descriptor.pca_variance` of the variance (default 0.95),
   with a fixed `random_seed` (default 42).

PCA is here for conditioning, not compression: the raw 134-column covariance is
close to singular because bonded internal coordinates are strongly correlated,
and inverting it directly gives a Mahalanobis distance dominated by noise
directions.

### 4.4 OOD statistic and threshold

The score is the Mahalanobis distance in PCA space, with mean $\mu$ and
covariance $\Sigma$ estimated on the training projections:

$$s(x) = \sqrt{(z(x) - \mu)^\top \Sigma^{-1} (z(x) - \mu)}, \qquad
\Sigma \leftarrow \Sigma + 10^{-8} I$$

where $z(\cdot)$ is standardise-then-project. The ridge term is numerical
stabilisation, not regularisation of a fit.

The threshold is a **percentile of the scores on the held-out validation set**
(`threshold.percentile`, default 99). This gives it an operational meaning: a
frame is flagged when it scores above 99 % of reference frames the fit never
saw. Define

$$\alpha = 1 - \frac{\text{percentile}}{100}$$

as the *false-flag rate* — the fraction of in-distribution frames the threshold
flags by construction. `evaluation.conformal_threshold` implements the same rule
as an order statistic, taking the $k$-th largest calibration score with
$k = \lceil (n+1)\alpha \rceil$, which bounds the false-flag probability by
$\alpha$ for a new exchangeable frame.

Beads and the centroid may take different percentiles
(`threshold.bead_percentile` / `centroid_percentile`), because a sensitive rule
suits an early-warning signal and a strict one suits the classical-limit signal.
Thresholds are stored per named track so calibrating one cannot overwrite
another.

**Nothing from the trajectory reaches either step.** The scaler, the PCA, the
mean and the covariance come from `train`; the threshold from `valid`;
`test` exists to check the calibration held. Calibrating on the trajectory would
define anomalies relative to the anomalous run itself and suppress exactly the
signal being looked for. There is a test asserting this and it is meant to stay.

### 4.5 Hard-chemistry checks, kept separate

Independently of the score, every frame is checked against fixed physical
cutoffs (`topology.check_chemistry`): a bonded pair further apart than
`bond_break_cutoff` (2.0 Å) is a broken bond; a non-bonded pair closer than
`close_contact_cutoff` (1.2 Å) is a close contact; ring planarity RMSD is
reported.

These answer a different question from the score and are reported in a different
file. "Out of distribution" is a statement about the training set; "broken" is a
statement about chemistry. Conflating them is the single most likely
misinterpretation of this pipeline, which is why the outputs are kept apart.

## 5. The embedding track (optional second measurement)

The geometric score asks *is this structure unusual?* The embedding score asks
*has the model seen anything like this?* — a closer proxy for reliability
([ADR 0002](adr/0002-embedding-ood-track.md)).

The force field's **invariant** per-atom embeddings (shape
`(n_atoms, n_features)`, e.g. `(21, 128)`, stored under the HDF5 dataset name
`inv_features`) are extracted **outside** DetectAna by
[`scripts/extract_embeddings.py`](../scripts/extract_embeddings.py) on a GPU node
and written to HDF5. Invariance is required, not incidental: an equivariant
tensor would make the score depend on molecular orientation. The architecture
used during development is part of unpublished work and is deliberately not named
in this repository — the extraction script takes the model's Python package as a
`--model-package` argument, so any model exposing invariant per-atom features
works. `embedding_scorer.EmbeddingPipeline` then fits one
Mahalanobis scorer per atom index on the reference training embeddings, scores
each atom of each frame, and aggregates to one score per frame by the **maximum
over atoms** — the same reasoning as for beads: a single atom in a novel
environment is the signal, and averaging removes it.

The split is deliberate: DetectAna never imports the force field, so the
analysis install stays CPU-only and no model dependency can break it. The cost
is that embeddings must be pre-computed, and the two tracks have not yet been
compared on a common benchmark.

## 6. Aggregation across beads

A PIMD run gives one trajectory per bead plus the centroid. At each timestep
`aggregator.aggregate_bead_scores` reduces the bead scores to:

- `bead_max` — the maximum,
- `bead_p95` — the 95th percentile,
- `bead_frac_ood` — the fraction above the threshold,

alongside `centroid_score` and `centroid_ood`. **Never the mean.** A single bead
can leave the training region long before the centroid notices, and the mean
erases that. The example figure in the README shows individual beads crossing
the threshold on and off through the entire first half of a run whose mean looks
flat.

Classical MD is the same code path with one replica: that replica is the
centroid.

## 7. The onset rule, and its false-alarm arithmetic

This is where most of the statistical care sits, because the obvious rule is
wrong.

### 7.1 Why the first flagged frame is not an onset

A threshold calibrated to flag a fraction $\alpha$ of in-distribution frames
flags about $\alpha$ of the frames of *any* long run, by construction. At
$\alpha = 1\%$ over 200 000 frames that is ~2000 flagged frames per bead, the
first arriving after ~100 frames — a few picoseconds. `first_bead_anomaly` is
therefore a restatement of the threshold, not a property of the trajectory. It
stays in the output as a diagnostic and must not be reported as an onset.

### 7.2 The window rule

Onset is the first window in which the flagged fraction reaches
`fraction_threshold`. Four criteria are reported separately:

| Criterion | Definition |
|---|---|
| `first_bead_anomaly` | first frame any bead exceeds the threshold (diagnostic only) |
| `persistent_bead_anomaly` | first window with `bead_frac_ood > fraction_threshold` |
| `centroid_anomaly` | first window with the centroid criterion met |
| `collective_anomaly` | first window where both hold |

A centroid onset much earlier than the persistent bead onset usually means a
brief excursion the run recovered from. That is information, so it is kept
rather than reconciled into one number.

### 7.3 Effective sample size

A 500-frame window is not 500 independent chances to flag, because consecutive
frames are correlated. Under an AR(1) approximation with lag-1 autocorrelation
$\rho$:

$$n_\text{eff} = W \cdot \frac{1 - \rho}{1 + \rho} \cdot n_\text{beads}^\text{eff}$$

- $\rho$ is *measured*, not assumed: `frame_autocorrelation: "auto"` estimates
  it on the opening `stable_fraction` of the run (default 0.1). Negative
  estimates are clipped to 0 because they would raise $n_\text{eff}$, and
  assuming independence is the conservative direction there.
- $n_\text{beads}^\text{eff}$ defaults to **1** (A7). Beads are path-integral
  images of one molecule; treating 16 as 16 independent draws would understate
  the false-alarm rate by orders of magnitude. Raise it only with evidence.

Measured on the development reference scores, $\rho \approx 0.3$–$0.37$, so a
500-frame window carries roughly 230 effective trials, not 500.

### 7.4 False-alarm bound

Under the null (the whole run is in distribution), the flag count in a window is
$\mathrm{Binomial}(n_\text{eff}, \alpha)$ and the rule fires when it reaches
$m = \lceil f \cdot n_\text{eff}\rceil$ flags. Per window:

$$P_\text{window} = P(X \ge m), \quad X \sim \mathrm{Binomial}(n_\text{eff}, \alpha)$$

and over a run of $N_W$ tested windows, by the union bound:

$$P_\text{run} \le \min(1, N_W \cdot P_\text{window})$$

Both approximations run the same way, so the reported number is an **upper
bound**, not an estimate. Overlapping windows sharing 90 % of their frames are
nearly the same test, so counting each as a separate trial is loose; the
pipeline therefore reports a second number over disjoint windows only. The truth
is between them and both are logged, so the width of the conservatism is visible
instead of assumed.

### 7.5 Inverting the bound: state a budget, not a fraction

`onset.choose_fraction_threshold` runs the arithmetic backwards. Given $\alpha$,
the window, the number of windows and a run-level `false_alarm_budget`, it
returns the **smallest** $m$ — hence the loosest fraction — whose bound stays
inside the budget. Loosest is best: sensitivity falls as the fraction rises, so
this is the rule that declares an onset on the weakest evidence the budget
allows.

This is the recommended direction, because it makes you state the false-alarm
rate you can live with instead of guessing a fraction. For 200 000 frames at
$\alpha = 1\%$ and a 1 % budget:

| assumption on $\rho$ | derived fraction | flags needed | bound per run |
|---|---|---|---|
| 0 (independence assumed) | 0.038 | 19 of 500 | 0.005 |
| 0.37 (measured) | 0.057 | 13 of 230 | 0.003 |

Compare the shipped default `fraction_threshold: 0.20`, which needs 100 of 500
frames flagged and has a bound near $10^{-91}$: safe to the point of being
insensitive, and capable of missing a real but partial excursion. Defaults were
left unchanged so no existing result moves, but the bound is logged for every
run either way, and the whole design report goes into `manifest.json`.

## 8. Experimental design: how the detector was tested

No trajectory exists that is annotated frame-by-frame with "the force field is
unreliable here", so labelled positives have to be manufactured — and the
obvious way to do that is wrong.

### 8.1 The mistake that shaped the design

The first version of the benchmark rotated whichever torsion divided the
molecule most evenly, called the rotated frames anomalies, and reported
AUROC ≈ 0.52 from 5° to 180°. Read naively: a detector blind to conformational
change. It was not. The chosen torsion (the ester C1–C6–O12–C11) is fully
sampled in the reference set — all twelve 30° slices occupied, σ = 92° — so the
rotated frames were *in distribution*, and flagging them would have been the
error. The benchmark was measuring its own mislabelling
([ADR 0004](adr/0004-detector-evaluation.md)).

**"Distorted" and "out of distribution" are different properties.** A large
displacement that lands inside the training distribution is not an anomaly.

### 8.2 Coverage-labelled positives

Every synthetic positive is therefore labelled by **training coverage**, not by
distortion size. In [`evaluation.py`](../src/detectana/evaluation.py):

- `torsion_coverage` histograms a torsion over the training set in 30° slices;
- `most_gap_rich_dihedral` picks the torsion with the most unvisited slices;
- `set_dihedral` drives a frame to an exact target angle;
- `perturb_gaussian`, `perturb_bond_stretch`, `perturb_dihedral` are the
  distortion ladders.

[`scripts/benchmark_detector.py`](../scripts/benchmark_detector.py) then scans a
torsion around its full circle and reports flag rate per slice against the
training count in that slice. Two numbers come out with opposite meanings:

| Regime | Flag rate is a measurement of | Should be |
|---|---|---|
| well-sampled slices | false alarms | near $\alpha$ |
| never-visited slices | sensitivity | high |

Pooling them averages a specificity result with a sensitivity result and
describes nothing — which is what the first version did.

Torsion rotation is the right probe because it is a *pure* conformational
change: bond lengths and bond angles are preserved exactly, which
`tests/test_evaluation.py` asserts to 1e-9 rather than assuming. Cartesian
rattle is kept as a floor and bond stretch because it can be compared directly
against the hard-chemistry bond check.

### 8.3 Metrics

- **Detection rate at $\alpha$** — the headline number, computed against the
  conformal order statistic of the calibration scores, i.e. the same threshold
  rule the pipeline deploys. It answers "what fraction of these would a real run
  have flagged?"
- **AUROC and average precision** — threshold-free, reported for context. They
  describe an idealised detector and are not what gets deployed.
- **Spearman correlation between training density and flag rate** — the summary
  statistic for the whole coverage argument. A strongly negative value is the
  result being claimed.
- **Score–error correlation** (Spearman, plus force error by score decile and
  the top-to-bottom decile ratio) — the validation that actually matters, and
  the one still open; see §9.

### 8.4 Guard rails in the benchmark

The benchmark warns when the training set holds fewer than ten frames per PCA
component, since the covariance is then noise-dominated. The demo configuration
is deliberately in that regime (64 synthetic frames, 134 columns), so CI's
benchmark step is a smoke test of the machinery, not a measurement — and it says
so.

## 9. Interpretation: what a flag means and what it does not

**Supported.** "This frame's internal-coordinate fingerprint is further from the
reference training distribution than 99 % of held-out reference frames." "The
window rule fired here, with a run-level false-alarm bound of $p$." "Training
coverage and flag rate are strongly anti-correlated on the development reference
set." "Bond stretches of 0.3–0.5 Å are flagged while the 2.0 Å hard-chemistry
cutoff is still silent."

**Not supported.** "This frame is unphysical." A torsion slice holding 5 of 2500
training frames is flagged every time, and those conformers are chemically
unremarkable — the benchmark demonstrates this rather than asserting it. "The
force field's error is large here": plausible, and the reason the tool exists,
but **not yet measured**. `scripts/score_vs_error.py` implements the test —
Spearman correlation between score and per-frame force error, error by score
decile — and it needs the same frames' forces recomputed with the reference
method, which has not been done. Until then the score is a statement about
training coverage only, which is all this repository claims.

## 10. Reproducibility and provenance

Every run writes `manifest.json` next to its results: the full resolved config,
the code version, the calibrated threshold, the topology (atom count, element
order, ring indices) and the complete onset design report ($\alpha$, window,
fraction used, measured $\rho$, effective trials, flags needed, both
false-alarm bounds). A score array without the threshold that produced it is not
interpretable six months later. Seeds are fixed for PCA and anything else
stochastic. Descriptor parameters and thresholds live in the config, not the
code.

Setup, commands, data placement and output layout:
[reproducibility.md](reproducibility.md).

## 11. Limitations

- **Positions only.** Forces are loaded and unit-checked (i-PI writes
  Hartree/Bohr, the reference set eV/Å, factor 51.4221) but never enter the
  descriptor. A force-based descriptor would likely catch some failures earlier.
- **The reliability claim is unvalidated.** See §9. This is the single most
  important open item.
- **The two tracks are not compared.** Geometric and embedding scores are
  reported side by side but have not been benchmarked against each other.
- **Single gas-phase molecule (A1).** No periodic boundaries, no second
  molecule, no reactions.
- **Global descriptor.** Local atomic-environment descriptors (SOAP, ACSF) would
  be more expressive; left out to avoid a heavy dependency
  ([ADR 0001](adr/0001-ood-scoring-method.md)).
- **Covariance needs data.** Roughly ten training frames per retained PCA
  component is the floor; below it the Mahalanobis distance measures noise. The
  shipped demo data is intentionally below it.
- **Fixed topology (A3).** A permanent bond change is detected but the
  coordinates stop meaning what their names say.
- **The headline benchmark numbers are not reproducible from a clone**, because
  the reference set behind them cannot be redistributed. What is reproducible is
  the machinery and every test.
