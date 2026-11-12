# Scientific rules

These are the constraints the pipeline is built around. Most of them exist
because breaking them produces a result that still looks fine on a plot, which
is the dangerous kind of mistake. If you change the code, keep them.

## Beads are not independent replicas

A PIMD run with 16 beads is not 16 simulations. The beads are path-integral
images of the same molecule at the same instant, coupled by the ring polymer.
Treating them as independent samples inflates your statistics by a factor you
cannot justify.

So a bead score is an early warning, not a measurement. The independent unit is
the run. Comparing two force fields means comparing runs, ideally several per
force field, not comparing beads.

## Never average bead scores away

The reason bead scores matter is that a single bead can wander out of the
training distribution long before the centroid notices. Take the mean over
beads and that signal is gone.

At each timestep the pipeline keeps the maximum, the 95th percentile, and the
fraction of beads above the threshold. Add other aggregates if they help, but
those three stay. Production runs show why: individual beads cross the threshold
on and off for a long stretch before anything happens collectively, while the mean
over beads looks flat throughout.

## Out of distribution is not the same as unphysical

A high score means the model is being asked about a structure unlike anything
in its training set. That is a statement about the training set, not about
chemistry. The structure may be perfectly reasonable and simply
under-represented in the reference data.

This matters for how results are worded. "Frame 679095 is out of distribution"
is a claim the pipeline can support. "Frame 679095 is unphysical" is not.
The hard-chemistry flags (broken bonds, close contacts, ring planarity) are a
separate check and are reported separately, because they answer the second
question and the OOD score does not.

## Nothing from the trajectory touches the fit or the threshold

The scaler, the PCA and the Mahalanobis statistics are fitted on the training
reference set. The threshold is the 99th percentile of the scores on the
validation set, which the fit never saw. Frames from the trajectory being
scored are never involved in either step.

Calibrating on the trajectory would define anomalies relative to the anomalous
run itself, which quietly suppresses exactly what you are looking for. There is
a test for this, and it should stay.

## The window rule is the detector, not the threshold

A threshold calibrated to flag a fraction α of in-distribution frames flags about
α of any long run by construction. At α = 1 % over 200,000 frames that is ~2000
flagged frames per bead, and the first arrives after ~100 frames, which is a few
picoseconds. So `first_bead_anomaly` says nothing about the trajectory; it is a
restatement of the threshold. It stays in the output because it is a useful
diagnostic, but it is not an onset and must not be reported as one.

What controls false alarms is the windowed criterion: the fraction of a window
that has to be flagged. The pipeline computes and records that trade-off for
every run rather than leaving it implicit. The null model is that flags inside a
window are Bernoulli(α); frame-to-frame correlation is handled by converting the
window to an effective sample size (AR(1), from the lag-1 autocorrelation measured
on the opening `stable_fraction` of the run, default 0.1), and overlapping windows
by a union bound. Both approximations run the same way, so the reported
probability is an upper bound on the false-alarm rate, not an estimate of it. A
second number counting only disjoint windows is reported alongside it, so the
width of that conservatism is visible instead of assumed.

Two consequences worth keeping:

- Prefer stating a false-alarm budget over guessing a fraction. Given α, the
  window and the budget, there is a unique loosest fraction that stays inside it,
  and that is the most sensitive rule available. `fraction_threshold: 0.20` with
  α = 1 % has a bound near 10⁻⁹¹ — it needs 100 of 500 frames flagged, so it is
  safe and insensitive at once, and a real partial excursion can pass under it.
- Do not assume frames are independent. Reference scores in this project show a
  clearly positive lag-1 autocorrelation, and a trajectory is worse. Assuming
  independence inflates the effective sample size and understates false alarms.

Beads count as one observation per timestep unless you have measured otherwise.
They are path-integral images of one molecule, so treating 16 beads as 16
independent draws would understate the false-alarm rate by orders of magnitude.

## A distortion is not an anomaly

Synthetic anomalies are the only labelled positives available, and the tempting
shortcut — distort a frame by a lot, call it out of distribution — is wrong. If the
training set already samples the region the distortion lands in, the frame *is* in
distribution and flagging it would be the error.

This is not hypothetical. Rotating aspirin's ester torsion by 180° produces a
perfectly ordinary frame, because the reference set covers that torsion's whole
circle, and an early version of the benchmark scored the detector near chance on
exactly that basis before the mislabelling was found.

So every synthetic positive is labelled by training coverage, and the two regimes
are reported separately: flag rate where training data is dense is a false-alarm
measurement, flag rate where there is none is a sensitivity measurement. Averaging
them describes nothing. See [adr/0004-detector-evaluation.md](adr/0004-detector-evaluation.md).

The same benchmark is where "out of distribution is not unphysical" stops being a
slogan: a torsion slice holding a handful of training frames is flagged reliably,
and those conformers are chemically unremarkable.

## Report the three levels separately

Bead, centroid and run are different things and get their own rows:

| Level | Unit | What it tells you |
|-------|------|-------------------|
| Bead | (run, bead, time) | Earliest warning, sensitive to quantum fluctuations |
| Centroid | (run, time) | The classical-limit signal |
| Run | one number per run | What you compare across force fields |

A centroid onset that fires much earlier than the persistent bead onset usually
means a brief excursion the run recovered from. That is information, so it is
kept rather than reconciled into a single number.

## Validate the geometry before computing anything

Atom count and atom order are checked against `initial.xyz` on every frame, and
a mismatch is a hard failure rather than a warning. Internal coordinates are
computed from index triples and quadruples, so a reordered file produces
descriptors that are wrong in a way nothing downstream can detect.

`initial.xyz` is the reference, not a hard-coded molecule: the first run's file
defines the expected atom count and element order (`io.MoleculeSpec`), and the
reference set, the other runs' initial geometries and every trajectory frame are
validated against it. The binary XYZ reader and the HDF5 files carry no element
symbols, so on those paths only the atom count can be checked.

Frame counts across bead files are treated differently, because a run killed
mid-write is ordinary rather than pathological. The analysis uses the frame range
every bead covers, and records the trim: `n_frames_used`, the per-bead counts
found on disk and the number of frames dropped go into the `frame_alignment`
block of `manifest.json`, and the trim is logged as a warning. The danger was
never the trimming — it was trimming quietly, which would let an onset measured
on half a trajectory read like one measured on all of it.

Trimming is defensible only because it drops frames from the tail, which leaves
frame *i* the same timestep in every bead. That is checked rather than assumed: if
the step numbers disagree over the common range, the beads are on different
strides or frame ranges, per-timestep aggregates would mix different times, and
the run fails. No trim repairs that.

## Be explicit about units

Positions are in ångström throughout. Reference forces are in eV/Å. Forces
written by i-PI are in atomic units, Hartree/Bohr, and need a factor of
51.4221 eV/Å per Hartree/Bohr to compare. Energies are in eV.

Where a quantity could be read two ways, the unit goes in the variable name or
the file metadata. The output tables carry both the raw step index and a time in
picoseconds, because `stride` and `timestep_fs` are easy to mix up and a plot
against the wrong one looks entirely plausible.

## Make runs reproducible

Seeds are fixed for PCA and anything else stochastic, and the config, the code
version and the calibrated threshold are written to `manifest.json` next to the
results. A score array without the threshold that produced it is not
interpretable six months later.

Descriptor parameters and thresholds live in the config file, not in the code.

## What a run has to produce

- a frame-level table: run, bead, time, score, flags
- a run-level onset table: first bead anomaly, persistent bead anomaly,
  centroid anomaly, collective anomaly
- the manifest: input files, parameters, code version, seeds, threshold
- score-against-time plots with the threshold and the onsets drawn on

## Checklist before calling a change done

The test suite covers all of this, so in practice this means running `pytest`.
It is written out because the list is the point, not the tests:

- descriptors compute on a small trajectory slice without NaNs or infinities
- a frame with a deliberately broken bond scores higher than a normal one
- torsions survive the 0 to 2π wrap, which is what the sin/cos encoding is for
- array shapes are right for bead, centroid, atom and time dimensions
- no trajectory frame has leaked into the fit or the calibration
- the first detected anomalies have been looked at, not just counted

That last one is not automatable. Plot the onset, pull the frames around it
with `extract_onset_frames.py`, and look at the structures before you believe
the number.

## Known limitations

Only positions go into the descriptor. Forces are loaded and unit-checked but not
scored on, even though a force-based descriptor would likely catch some failures
earlier. Forces are used in one place, outside the pipeline:
`scripts/score_vs_error.py` compares the score against force error.

Whether a high score actually predicts force-field error is not yet answered. The
machinery exists and is validated on synthetic controls, but it needs forces
recomputed with the reference method for the frames being analysed, which has not
been run. Until it has, the score is a statement about training coverage only —
which is all the repository claims for it.

Only a single gas-phase molecule is supported. Internal coordinates are computed
on raw coordinates with no periodic images, so a periodic cell or a second
molecule in the file is not handled; a disconnected bond graph produces a warning
and nothing more.

Frame validation is no longer tied to aspirin: the reference atom count and
element order come from `initial.xyz`. The molecule still has to be the same one
throughout a run, which is the point of the check.

Local atomic-environment descriptors of the SOAP or ACSF kind would be more
expressive than internal coordinates. They were left out to avoid a heavy
dependency, and the embedding track covers some of the same ground. See
[adr/0001-ood-scoring-method.md](adr/0001-ood-scoring-method.md).
