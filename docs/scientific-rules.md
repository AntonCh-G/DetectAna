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
those three stay. The example plot in the README shows why: individual beads
cross the threshold on and off for the entire first half of a 500 ns run, while
the mean would have looked flat.

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

The same applies to bead counts and frame counts across bead files. A truncated
bead file gives a shorter score array, and silently trimming it would misalign
every timestep after that point.

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

Only positions go into the descriptor. Forces are loaded and unit-checked but
not scored on, even though a force-based descriptor would likely catch some
failures earlier.

Frame validation is hard-coded to aspirin, 21 atoms in a fixed order. The
descriptors themselves are general.

Local atomic-environment descriptors of the SOAP or ACSF kind would be more
expressive than internal coordinates. They were left out to avoid a heavy
dependency, and the embedding track covers some of the same ground. See
[adr/0001-ood-scoring-method.md](adr/0001-ood-scoring-method.md).
