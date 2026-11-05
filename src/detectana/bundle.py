"""Reference bundle: the saved fit for one force field (ADR 0005).

A bundle is fitted once from training and validation frames and then scores any
number of frames, in batch or one at a time. It is a directory::

    <name>.bundle/
        manifest.json   what the fit is: molecule, units, features, thresholds,
                        onset rule, versions, descriptor code hash
        state.npz       fitted arrays only, read with allow_pickle=False

Callers use the public properties and methods; they never open the fitted state.

The statistic is Hotelling's T², the squared Mahalanobis distance in the kept PCA
space. "Distance" (√T²) is only a display label. Thresholds are percentiles of the
validation T² values, so they can differ marginally from the square of the
distance percentile the batch pipeline reports (both interpolate linearly between
order statistics); flags agree exactly only at the order statistics themselves.

Today there is one track, ``"geometric"``: the internal-coordinate descriptor,
standardised, reduced by PCA and scored by T². ADR 0006 adds more.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from detectana import __version__
from detectana.descriptors import (
    DescriptorPipeline,
    _compute_descriptor_batch_vectorised,
    _dihedral_rad,
    compute_descriptor,
    compute_descriptor_batch,
)
from detectana.onset import WindowRule
from detectana.scorer import MahalanobisScorer
from detectana.topology import MoleculeTopology, _ring_planarity_rmsd

BUNDLE_FORMAT = "detectana-reference-bundle"
#: Bumped whenever the manifest layout or the meaning of a stored value changes.
#: Version 1 stores T² (not the distance √T²).
BUNDLE_FORMAT_VERSION = 1

GEOMETRIC_TRACK = "geometric"
DESCRIPTOR_NAME = "internal_coordinates"
SUPPORTED_LENGTH_UNITS = ("angstrom",)
UNITS = {"length": "angstrom", "angle": "radian"}

MANIFEST_FILE = "manifest.json"
STATE_FILE = "state.npz"

REFERENCE_KEY = "molecule/reference_positions"
#: A first-frame bond longer than this multiple of its reference length counts as broken.
BOND_STRETCH_LIMIT = 1.5

_STATE_ARRAYS = ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "t2_mean", "t2_inv_cov")

# The functions that compute the internal-coordinate descriptor. Their source text
# is hashed into the manifest so a bundle refuses to score with changed code.
_DESCRIPTOR_FUNCTIONS = (
    compute_descriptor_batch,
    _compute_descriptor_batch_vectorised,
    compute_descriptor,
    _dihedral_rad,
    _ring_planarity_rmsd,
)


class BundleError(ValueError):
    """A bundle, or frames given to it, do not match; the message says why."""


@dataclass(frozen=True)
class FrameCheck:
    """Diagnostics for a first frame that passed ``validate_frame``'s hard checks.

    Neither check proves atom identity; they catch common mistakes.

    broken_bonds : (i, j, length, reference_length) for each bundle bond longer
        than ``BOND_STRETCH_LIMIT`` × its length in the reference geometry. Catches
        swaps between atoms bonded to different partners.
    possible_swaps : (i, j) pairs of same-element atoms whose positions match the
        reference better exchanged than as numbered (rigid alignment, then
        per-element Hungarian matching). Catches swaps the bond graph cannot see,
        such as two hydrogens on one carbon, but only while the frame is close to
        the reference: a methyl group rotated by 120° looks the same as a swap.
    rmsd_to_reference : Å, after rigid alignment, atoms as numbered.
    rmsd_after_matching : Å, after alignment with the best same-element matching.
    """

    broken_bonds: tuple[tuple[int, int, float, float], ...]
    possible_swaps: tuple[tuple[int, int], ...]
    rmsd_to_reference: float
    rmsd_after_matching: float

    @property
    def ok(self) -> bool:
        return not self.broken_bonds and not self.possible_swaps

    @property
    def summary(self) -> str:
        if self.ok:
            return f"ok (RMSD to reference {self.rmsd_to_reference:.3f} Å)"
        parts = []
        if self.broken_bonds:
            bonds = ", ".join(f"{i}-{j} {d:.2f} Å (ref {r:.2f})" for i, j, d, r in self.broken_bonds[:5])
            parts.append(f"{len(self.broken_bonds)} stretched bond(s): {bonds}")
        if self.possible_swaps:
            pairs = ", ".join(f"{i}<->{j}" for i, j in self.possible_swaps)
            parts.append(
                f"possible atom swap(s) {pairs}: RMSD {self.rmsd_to_reference:.3f} Å as numbered, "
                f"{self.rmsd_after_matching:.3f} Å matched"
            )
        return "; ".join(parts)


def descriptor_code_hash() -> str:
    """SHA-256 of the source of the functions that compute the descriptor.

    A formatting edit changes it too (ADR 0006): such a bundle is refused and must
    be refitted. The DetectAna version in the manifest narrows down which release
    wrote it.
    """
    h = hashlib.sha256()
    for fn in _DESCRIPTOR_FUNCTIONS:
        h.update(f"{fn.__module__}.{fn.__qualname__}\n".encode())
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(
    train: np.ndarray,
    valid: np.ndarray,
    topology: MoleculeTopology,
    *,
    pca_variance: float = 0.95,
    random_seed: int = 42,
    percentiles: Mapping[str, float] | None = None,
    window_frames: int = 500,
    step_frames: int = 50,
    fraction_threshold: float = 0.20,
    length_unit: str = "angstrom",
) -> ReferenceBundle:
    """Fit a reference bundle on ``train`` and calibrate its thresholds on ``valid``.

    Parameters
    ----------
    train, valid : (n_frames, n_atoms, 3) positions in ``length_unit``, atoms in
        the topology's order. Disjoint sets: the thresholds are only honest when
        no validation frame was used to fit. Overlap is not checked.
    topology : from ``build_topology``; fixes atom order, elements and features.
    pca_variance : fraction of standardised-descriptor variance PCA keeps.
    random_seed : PCA seed.
    percentiles : threshold name → percentile of validation T², default
        ``{"bead": 99.0, "centroid": 99.0}``.
    window_frames, step_frames, fraction_threshold : the window onset rule.
    length_unit : only ``"angstrom"`` is supported.
    """
    _check_length_unit(length_unit)
    if percentiles is None:
        percentiles = {"bead": 99.0, "centroid": 99.0}
    rule = WindowRule(int(window_frames), int(step_frames), float(fraction_threshold))
    train = _as_frames(train, topology.n_atoms, "train")
    valid = _as_frames(valid, topology.n_atoms, "valid")
    if topology.reference_positions is not None:
        reference = _as_frames(topology.reference_positions[None], topology.n_atoms, "reference")[0]
        reference_source = "topology"
    else:
        reference = train[0]
        reference_source = "first training frame"

    X_train = compute_descriptor_batch(train, topology)
    pipe = DescriptorPipeline(pca_variance=pca_variance, random_seed=random_seed).fit(X_train)
    scorer = MahalanobisScorer().fit(pipe.transform(X_train))
    scaler, pca = pipe._scaler, pipe._pca
    assert scaler is not None and pca is not None
    mean, inv_cov = scorer._fitted()

    state = {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "pca_mean": np.asarray(pca.mean_, dtype=np.float64),
        "pca_components": np.asarray(pca.components_, dtype=np.float64),
        "t2_mean": np.asarray(mean, dtype=np.float64),
        "t2_inv_cov": np.asarray(inv_cov, dtype=np.float64),
    }

    valid_t2 = _t2(compute_descriptor_batch(valid, topology), state)
    thresholds = {
        str(name): {"percentile": float(p), "value": float(np.percentile(valid_t2, p))}
        for name, p in percentiles.items()
    }

    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "statistic": "T2",
        "detectana_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "molecule": {
            "n_atoms": int(topology.n_atoms),
            "elements": list(topology.atom_types),
            "index_base": 0,
            "reference_geometry": {"source": reference_source, "state_key": REFERENCE_KEY},
            "topology": {
                "bonds": [list(map(int, b)) for b in topology.bonds],
                "angles": [list(map(int, a)) for a in topology.angles],
                "dihedrals": [list(map(int, d)) for d in topology.dihedrals],
                "ring_atoms": [int(i) for i in topology.ring_atoms],
            },
        },
        "units": dict(UNITS),
        "tracks": {
            GEOMETRIC_TRACK: {
                "descriptor": {"name": DESCRIPTOR_NAME, "code_sha256": descriptor_code_hash()},
                "feature_names": list(topology.feature_names),
                "n_features": int(topology.n_features),
                "n_components": int(state["pca_components"].shape[0]),
                "parameters": {"pca_variance": float(pca_variance), "random_seed": int(random_seed)},
                "thresholds": thresholds,
                "state_keys": [f"{GEOMETRIC_TRACK}/{k}" for k in _STATE_ARRAYS],
            }
        },
        "onset_rule": rule.to_dict(),
        "fit_data": {"n_train": int(len(train)), "n_valid": int(len(valid))},
    }
    arrays = {f"{GEOMETRIC_TRACK}/{k}": v for k, v in state.items()}
    arrays[REFERENCE_KEY] = np.array(reference, dtype=np.float64)
    return ReferenceBundle(manifest, arrays)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class ReferenceBundle:
    """A fitted reference: metadata, scoring and the onset rule.

    Build one with ``fit`` or ``load``; do not construct it directly.
    """

    def __init__(self, manifest: dict[str, Any], state: Mapping[str, np.ndarray]) -> None:
        self._manifest = manifest
        self._state = {k: np.asarray(v) for k, v in state.items()}
        self._topology = _topology_from_manifest(manifest)

    # ── metadata ─────────────────────────────────────────────────────────────
    @property
    def manifest(self) -> dict[str, Any]:
        """A copy of the manifest, as written to ``manifest.json``."""
        return copy.deepcopy(self._manifest)

    @property
    def format_version(self) -> int:
        return int(self._manifest["format_version"])

    @property
    def detectana_version(self) -> str:
        return str(self._manifest["detectana_version"])

    @property
    def elements(self) -> tuple[str, ...]:
        """Element symbol of each atom, in the order frames must use."""
        return tuple(self._manifest["molecule"]["elements"])

    @property
    def n_atoms(self) -> int:
        return int(self._manifest["molecule"]["n_atoms"])

    @property
    def length_unit(self) -> str:
        return str(self._manifest["units"]["length"])

    @property
    def topology(self) -> dict[str, list]:
        """Bonds, angles, dihedrals and ring atoms, 0-based atom indices."""
        return copy.deepcopy(self._manifest["molecule"]["topology"])

    @property
    def reference_positions(self) -> np.ndarray:
        """The reference geometry, (n_atoms, 3) Å, used by ``validate_frame``."""
        return self._state[REFERENCE_KEY].copy()

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(self._manifest["tracks"])

    def feature_names(self, track: str) -> tuple[str, ...]:
        """Descriptor column names of ``track``, in column order."""
        return tuple(self._track(track)["feature_names"])

    def thresholds(self, track: str) -> dict[str, float]:
        """Calibrated T² thresholds of ``track``, by name (``bead``, ``centroid``)."""
        return {name: float(t["value"]) for name, t in self._track(track)["thresholds"].items()}

    @property
    def onset_rule(self) -> WindowRule:
        return WindowRule(**self._manifest["onset_rule"])

    @property
    def history_frames(self) -> int:
        """Frames a streaming caller must keep for the onset rule."""
        return self.onset_rule.history_frames

    # ── frames ───────────────────────────────────────────────────────────────
    def validate_frame(
        self,
        elements: Sequence[str],
        positions: np.ndarray,
        *,
        length_unit: str = "angstrom",
    ) -> FrameCheck:
        """Check a first frame against the bundle.

        Raises ``BundleError`` when the unit, the atom count, the element of any
        atom (in order), or the positions' shape and finiteness do not match.

        Otherwise returns a ``FrameCheck`` with two diagnostics against the
        reference geometry: stretched bonds and possible same-element swaps. They
        are not proof either way: matching elements and a clean check do not
        prove atom identity. Only starting the run from the bundle's reference
        frame, in its atom order, does.
        """
        self._check_unit(length_unit)
        elements = [str(e) for e in elements]
        if len(elements) != self.n_atoms:
            raise BundleError(f"frame has {len(elements)} atoms; bundle expects {self.n_atoms}")
        diffs = [
            f"{i}: {got} (expected {want})"
            for i, (got, want) in enumerate(zip(elements, self.elements))
            if got != want
        ]
        if diffs:
            shown = ", ".join(diffs[:5]) + (" …" if len(diffs) > 5 else "")
            raise BundleError(f"atom order differs from the bundle at atom {shown}")
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (self.n_atoms, 3):
            raise BundleError(f"positions have shape {pos.shape}; expected ({self.n_atoms}, 3)")
        if not np.all(np.isfinite(pos)):
            raise BundleError("positions contain NaN or infinity")
        return _frame_check(pos, self.reference_positions, self.elements, self._topology.bonds)

    def score(self, positions: np.ndarray, *, length_unit: str = "angstrom") -> dict[str, np.ndarray]:
        """T² per frame for each track.

        Parameters
        ----------
        positions : (n_frames, n_atoms, 3), or (n_atoms, 3) for one frame, in the
            bundle's atom order.

        Returns
        -------
        {track: (n_frames,) T²}
        """
        self._check_unit(length_unit)
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 2:
            pos = pos[None]
        try:
            pos = _as_frames(pos, self.n_atoms, "positions", allow_empty=True)
        except ValueError as exc:
            raise BundleError(str(exc)) from exc
        return self.score_features(compute_descriptor_batch(pos, self._topology))

    def score_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """T² per frame from descriptors already computed with this bundle's topology.

        For callers that cache descriptors (the batch pipeline does). The columns
        must be in ``feature_names(track)`` order; only the width is checked.

        Parameters
        ----------
        features : (n_frames, n_features)
        """
        X = np.asarray(features, dtype=np.float64)
        n_features = len(self.feature_names(GEOMETRIC_TRACK))
        if X.ndim != 2 or X.shape[1] != n_features:
            raise BundleError(f"features have shape {X.shape}; expected (n_frames, {n_features})")
        return {GEOMETRIC_TRACK: _t2(X, self._track_state(GEOMETRIC_TRACK))}

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Write the bundle directory (created if needed; its two files replaced)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = dict(self._state)
        with open(path / STATE_FILE, "wb") as fh:
            np.savez(fh, **arrays)
        (path / MANIFEST_FILE).write_text(json.dumps(self._manifest, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ReferenceBundle:
        """Read a bundle directory written by ``save``.

        Raises ``BundleError`` when the bundle cannot be trusted with this
        installation: another format version or statistic, descriptor code that
        changed since the fit, unsupported units, or a manifest and fitted
        arrays that do not agree with each other.
        """
        path = Path(path)
        for name in (MANIFEST_FILE, STATE_FILE):
            if not (path / name).is_file():
                raise BundleError(f"{path}: missing {name}; not a saved reference bundle")
        try:
            manifest = json.loads((path / MANIFEST_FILE).read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BundleError(f"{path / MANIFEST_FILE}: unreadable manifest ({exc})") from exc
        try:
            with np.load(path / STATE_FILE, allow_pickle=False) as npz:
                state = {k: npz[k] for k in npz.files}
        except (OSError, ValueError) as exc:
            raise BundleError(f"{path / STATE_FILE}: unreadable fitted state ({exc})") from exc
        _check_bundle(manifest, state)
        return cls(manifest, state)

    # ── internals ────────────────────────────────────────────────────────────
    def _track(self, track: str) -> dict[str, Any]:
        tracks = self._manifest["tracks"]
        if track not in tracks:
            raise KeyError(f"unknown track {track!r}; bundle has {', '.join(tracks)}")
        return tracks[track]

    def _track_state(self, track: str) -> dict[str, np.ndarray]:
        prefix = f"{track}/"
        return {k[len(prefix):]: v for k, v in self._state.items() if k.startswith(prefix)}

    def _check_unit(self, length_unit: str) -> None:
        if length_unit != self.length_unit:
            raise BundleError(
                f"frames are in {length_unit!r}; bundle expects {self.length_unit!r}"
            )


def load(path: str | Path) -> ReferenceBundle:
    """Load a reference bundle directory."""
    return ReferenceBundle.load(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t2(X: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standardise, project onto the kept PCA directions, return T² per row."""
    Xs = (X - state["scaler_mean"]) / state["scaler_scale"]
    Z = (Xs - state["pca_mean"]) @ state["pca_components"].T
    diff = Z - state["t2_mean"]
    return np.maximum(np.einsum("ij,jk,ik->i", diff, state["t2_inv_cov"], diff), 0.0)


def _as_frames(pos: np.ndarray, n_atoms: int, name: str, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (n_atoms, 3):
        raise ValueError(f"{name}: expected (n_frames, {n_atoms} atoms, 3), got shape {arr.shape}")
    if len(arr) == 0 and not allow_empty:
        raise ValueError(f"{name}: no frames")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: positions contain NaN or infinity")
    return arr


def _check_length_unit(length_unit: str) -> None:
    if length_unit not in SUPPORTED_LENGTH_UNITS:
        raise ValueError(
            f"length unit {length_unit!r} is not supported; use one of {SUPPORTED_LENGTH_UNITS}"
        )


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigidly align P onto Q (rotation + translation, no reflection); return (P aligned, RMSD)."""
    p0, q0 = P.mean(axis=0), Q.mean(axis=0)
    H = (P - p0).T @ (Q - q0)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    aligned = (P - p0) @ R.T + q0
    return aligned, float(np.sqrt(np.mean(np.sum((aligned - Q) ** 2, axis=1))))


def _match_to_reference(
    frame: np.ndarray, ref: np.ndarray, elements: Sequence[str], max_rounds: int = 10,
) -> tuple[np.ndarray, float]:
    """Best same-element matching of ``frame`` atoms to ``ref`` atoms.

    Alternates rigid alignment and per-element Hungarian assignment until the
    matching stops changing. Returns ``perm`` (frame atom ``perm[k]`` matches
    reference atom ``k``) and the RMSD after aligning with it. Not guaranteed to
    find the global optimum; good enough near the reference.
    """
    elems = np.asarray(elements)
    perm = np.arange(len(ref))
    rmsd = _kabsch(frame, ref)[1]
    for _ in range(max_rounds):
        aligned, rmsd = _kabsch(frame[perm], ref)
        by_frame_atom = np.empty_like(aligned)
        by_frame_atom[perm] = aligned
        new = perm.copy()
        for e in np.unique(elems):
            idx = np.flatnonzero(elems == e)
            rows, cols = linear_sum_assignment(cdist(ref[idx], by_frame_atom[idx], "sqeuclidean"))
            new[idx[rows]] = idx[cols]
        if np.array_equal(new, perm):
            break
        perm = new
    return perm, _kabsch(frame[perm], ref)[1]


def _frame_check(
    frame: np.ndarray, ref: np.ndarray, elements: Sequence[str], bonds: Sequence[tuple[int, int]],
) -> FrameCheck:
    broken = []
    for i, j in bonds:
        d = float(np.linalg.norm(frame[i] - frame[j]))
        r = float(np.linalg.norm(ref[i] - ref[j]))
        if d > BOND_STRETCH_LIMIT * r:
            broken.append((int(i), int(j), d, r))

    rmsd_identity = _kabsch(frame, ref)[1]
    perm, rmsd_matched = _match_to_reference(frame, ref, elements)
    swaps: tuple[tuple[int, int], ...] = ()
    # Report a different matching only when it fits strictly better.
    if rmsd_matched < rmsd_identity - 1e-9:
        swaps = tuple(sorted({
            (int(min(k, perm[k])), int(max(k, perm[k]))) for k in range(len(perm)) if perm[k] != k
        }))
    else:
        rmsd_matched = rmsd_identity
    return FrameCheck(tuple(broken), swaps, rmsd_identity, rmsd_matched)


def _check_bundle(manifest: Any, state: Mapping[str, np.ndarray]) -> None:
    """Refuse a manifest + state this installation cannot score faithfully."""
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError("manifest: not a DetectAna reference bundle")
    version = manifest.get("format_version")
    if version != BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"bundle format version {version!r}; this DetectAna "
            f"({__version__}) reads format version {BUNDLE_FORMAT_VERSION} only. Refit the bundle."
        )
    if manifest.get("statistic") != "T2":
        raise BundleError(f"bundle statistic {manifest.get('statistic')!r}; expected 'T2'")

    units = manifest.get("units")
    if units != UNITS:
        raise BundleError(f"bundle units {units!r}; supported unit set is {UNITS!r}")

    try:
        mol = manifest["molecule"]
        n_atoms = int(mol["n_atoms"])
        elements = list(mol["elements"])
        topo_dict = mol["topology"]
        tracks = manifest["tracks"]
        WindowRule(**manifest["onset_rule"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(f"manifest: missing or invalid field ({exc})") from exc

    if len(elements) != n_atoms:
        raise BundleError(f"manifest: {len(elements)} elements for {n_atoms} atoms")
    for key in ("bonds", "angles", "dihedrals"):
        for group in topo_dict.get(key, []):
            if any(not 0 <= int(i) < n_atoms for i in group):
                raise BundleError(f"manifest: {key} entry {group} has an atom index outside 0..{n_atoms - 1}")
    if any(not 0 <= int(i) < n_atoms for i in topo_dict.get("ring_atoms", [])):
        raise BundleError(f"manifest: ring_atoms has an atom index outside 0..{n_atoms - 1}")

    if set(tracks) != {GEOMETRIC_TRACK}:
        raise BundleError(f"manifest: tracks {sorted(tracks)}; this DetectAna knows only {GEOMETRIC_TRACK!r}")
    track = tracks[GEOMETRIC_TRACK]
    descriptor = track.get("descriptor", {})
    if descriptor.get("name") != DESCRIPTOR_NAME:
        raise BundleError(f"track {GEOMETRIC_TRACK!r}: unknown descriptor {descriptor.get('name')!r}")
    if descriptor.get("code_sha256") != descriptor_code_hash():
        raise BundleError(
            f"track {GEOMETRIC_TRACK!r}: descriptor code differs from the code that fitted "
            f"this bundle (fitted with DetectAna {manifest.get('detectana_version')}, "
            f"installed {__version__}). Refit the bundle."
        )

    try:
        topo = _topology_from_manifest(manifest)
    except BundleError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BundleError(f"manifest: invalid topology ({exc})") from exc
    n_features = len(track.get("feature_names", []))
    if n_features != topo.n_features or track.get("n_features") != topo.n_features:
        raise BundleError(
            f"track {GEOMETRIC_TRACK!r}: {n_features} feature names, the topology gives {topo.n_features}"
        )

    expected_keys = [f"{GEOMETRIC_TRACK}/{k}" for k in _STATE_ARRAYS]
    missing = [k for k in expected_keys if k not in state]
    if missing or track.get("state_keys") != expected_keys:
        raise BundleError(f"fitted state: missing arrays {missing or track.get('state_keys')}")
    n_comp = int(track.get("n_components", -1))
    shapes = {
        "scaler_mean": (n_features,),
        "scaler_scale": (n_features,),
        "pca_mean": (n_features,),
        "pca_components": (n_comp, n_features),
        "t2_mean": (n_comp,),
        "t2_inv_cov": (n_comp, n_comp),
    }
    for key, shape in shapes.items():
        arr = state[f"{GEOMETRIC_TRACK}/{key}"]
        if arr.shape != shape:
            raise BundleError(f"fitted state: {key} has shape {arr.shape}, manifest implies {shape}")
        if arr.dtype.kind != "f" or not np.all(np.isfinite(arr)):
            raise BundleError(f"fitted state: {key} is not a finite float array")
    ref_key = mol.get("reference_geometry", {}).get("state_key")
    if ref_key != REFERENCE_KEY or REFERENCE_KEY not in state:
        raise BundleError("fitted state: missing the reference geometry")
    ref = state[REFERENCE_KEY]
    if ref.shape != (n_atoms, 3) or ref.dtype.kind != "f" or not np.all(np.isfinite(ref)):
        raise BundleError(f"fitted state: reference geometry has shape {ref.shape}, expected ({n_atoms}, 3)")
    for name, t in track.get("thresholds", {}).items():
        if not np.isfinite(float(t.get("value", np.nan))):
            raise BundleError(f"track {GEOMETRIC_TRACK!r}: threshold {name!r} is not a finite number")


def _topology_from_manifest(manifest: Mapping[str, Any]) -> MoleculeTopology:
    """Rebuild the topology the descriptor needs from the manifest alone."""
    mol = manifest["molecule"]
    t = mol["topology"]
    topo = MoleculeTopology(
        bonds=[(int(i), int(j)) for i, j in t["bonds"]],
        angles=[(int(i), int(j), int(k)) for i, j, k in t["angles"]],
        dihedrals=[(int(i), int(j), int(k), int(l)) for i, j, k, l in t["dihedrals"]],
        ring_atoms=[int(i) for i in t["ring_atoms"]],
        n_atoms=int(mol["n_atoms"]),
        atom_types=list(mol["elements"]),
    )
    names = list(manifest["tracks"][GEOMETRIC_TRACK]["feature_names"])
    if len(names) != topo.n_features:
        raise BundleError(
            f"track {GEOMETRIC_TRACK!r}: {len(names)} feature names, the topology gives {topo.n_features}"
        )
    nb, na, nd = topo.n_bond_features, topo.n_angle_features, topo.n_dihedral_features
    topo.bond_names = names[:nb]
    topo.angle_names = names[nb:nb + na]
    topo.dihedral_names = names[nb + na:nb + na + nd]
    if topo.has_ring:
        topo.planarity_name = names[nb + na + nd]
    topo.bond_idx = np.array(topo.bonds, dtype=np.int32).reshape(-1, 2)
    topo.angle_idx = np.array(topo.angles, dtype=np.int32).reshape(-1, 3)
    topo.dihedral_idx = np.array(topo.dihedrals, dtype=np.int32).reshape(-1, 4)
    topo.ring_idx = np.array(topo.ring_atoms, dtype=np.int32) if topo.ring_atoms else None
    return topo
