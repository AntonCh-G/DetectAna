# ADR 0003 — Molecule-agnostic Topology: the molecule comes from `initial.xyz`

**Status:** Accepted
**Date:** 2026-08-10

## Context

The descriptor and scoring code was always general — bonds, angles, dihedrals and
the PCA/Mahalanobis stack work for any molecule — but three things pinned the
package to aspirin:

1. `io.ASPIRIN_N_ATOMS = 21` and `io.ASPIRIN_ATOM_TYPES`, a fixed element list.
   Every loader validated against these module constants, so any other molecule
   (or aspirin in a different atom order) failed at load time.
2. `topology._find_benzene_ring` raised if no 6-membered all-carbon ring existed,
   and the ring-planarity RMSD was an unconditional descriptor column
   (`n_features` always ended in `+ 1`).
3. Reference data was loaded *before* the topology was built, so `initial.xyz`
   could not be the source of truth for validation even in principle.

The validation itself is not the problem — internal coordinates are index tuples,
so a reordered file gives silently wrong descriptors and the check has to stay.
What was wrong is *where the expected ordering came from*.

## Decision

**`initial.xyz` defines the molecule.** The first run's initial geometry is loaded
first, the topology is built from it, and its atom count and element order become
a `MoleculeSpec` that the reference set, the remaining runs' initial geometries
and every trajectory frame are validated against. The loaders take that spec as
an argument; called without one — as the defining frame itself is — they skip the
comparison. `ASPIRIN_SPEC` is kept so the old strict check can still be requested
by name.

**The ring is optional and selectable.** Auto-detection now returns nothing
instead of raising, and a molecule with no all-carbon ring simply has no
planarity feature and no planarity flag (`ChemistryFlags.ring_planarity_rmsd`
becomes `None`). `chemistry.ring_atoms` in the config selects the ring explicitly,
which matters because auto-detection picks the lowest-indexed candidate when there
is more than one; `ring_atoms: []` switches the feature off.

**Aspirin behaviour is unchanged.** The demo run reproduces bit-identical
descriptors, scores, aggregates and onsets, because for aspirin `initial.xyz`
holds exactly the old hard-coded ordering and the ring is found as before.

## Consequences

- Descriptor length now depends on the topology: a ring-less molecule is one
  column shorter. The fit, the calibrated threshold and the scored frames must
  come from one topology. The manifest records atom count, element order and ring
  indices for this reason.
- Symbol-free formats limit the check. The binary XYZ reader skips element
  symbols and the HDF5 files store none, so on those paths only the atom count is
  verified. The reader now also rejects files with mixed atom counts across frames.
- Still one gas-phase molecule. No periodic images, no second molecule; a
  disconnected bond graph produces a warning, not support.
- `AspirinTopology` is now an alias of `MoleculeTopology`, kept for external code.

## Alternatives considered

**Keep the constants, add a config override.** Rejected: two sources of truth for
atom ordering, and the config copy would drift from `initial.xyz`, which is the
file the topology is actually built from.

**Derive the spec from the reference training set instead.** Rejected: the bond
graph comes from `initial.xyz`, so making a different file the ordering authority
would allow a mismatch between the graph and the validation.

**Keep planarity mandatory and require a ring.** Rejected: it excludes whole
classes of molecules for one descriptor column out of a hundred or more, and the
hard-chemistry ring check is reported separately anyway.
