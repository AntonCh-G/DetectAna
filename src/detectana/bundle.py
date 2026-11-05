"""Reference bundle: the saved fit for one force field (ADR 0005, 0006, 0007).

A bundle is fitted once from training and validation frames and then scores any
number of frames, in batch or one at a time. It is a directory::

    <name>.bundle/
        manifest.json   what the fit is: molecule, units, descriptors and their
                        code hashes, tracks, thresholds, budget, onset rule, versions
        state.npz       fitted arrays only, read with allow_pickle=False

Callers use the public properties and methods; they never open the fitted state.

**Tracks.** A track is a set of descriptor numbers fitted, scored and
thresholded together: standardised, reduced by PCA, then scored twice —

- T², Hotelling's statistic, the squared Mahalanobis distance in the kept PCA
  space: unusual *within* the motions the training data showed;
- SPE, the squared PCA residual: a new kind of motion. Inactive when PCA keeps
  every direction or the validation residual is numerically zero.

The default is four tracks — bonds, angles, dihedrals, ring — and is
provisional until benchmarked (ADR 0006). ``JOINED_TRACKS`` is the single
joined track that reproduces the step-1 descriptor.

**Tests and budget.** Each active (track, score) pair is a *test*. For each
threshold role (``bead``, ``centroid``) the total false-alarm rate
α = 1 − percentile/100 is split equally over the K active tests (Bonferroni):
each test's threshold is the 100·(1 − α/K) percentile of its validation values,
and a frame is marked when any test exceeds its threshold. ``exceedance`` turns
that into one number per frame.

Thresholds are percentiles (linear interpolation) of validation values.
"""

from __future__ import annotations

import copy
import datetime as _dt
import importlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from detectana import __version__
from detectana.descriptors import DescriptorPipeline
from detectana.onset import WindowRule
from detectana.registry import (
    BUILTIN_DESCRIPTORS,
    DescriptorValues,
    RegisteredDescriptor,
    compute,
    resolve_descriptor,
)
from detectana.scorer import MahalanobisScorer
from detectana.topology import MoleculeTopology, complete_topology

BUNDLE_FORMAT = "detectana-reference-bundle"
#: Bumped whenever the manifest layout or the meaning of a stored value changes.
#: 1: one track, T². 2: descriptor registry, tracks, T² and SPE, budget split.
BUNDLE_FORMAT_VERSION = 2

SCORES = ("T2", "SPE")
#: ADR 0006 default, provisional until the benchmark decides.
DEFAULT_TRACKS: dict[str, tuple[str, ...]] = {name: (name,) for name in BUILTIN_DESCRIPTORS}
#: One joined track: the step-1 internal-coordinate descriptor.
JOINED_TRACKS: dict[str, tuple[str, ...]] = {"geometric": BUILTIN_DESCRIPTORS}

SUPPORTED_LENGTH_UNITS = ("angstrom",)
UNITS = {"length": "angstrom", "angle": "radian"}
BUDGET_RULE = "bonferroni"
#: Largest validation SPE below this (standardised units²) means no residual.
SPE_FLOOR = 1e-10

MANIFEST_FILE = "manifest.json"
STATE_FILE = "state.npz"

REFERENCE_KEY = "molecule/reference_positions"
#: A first-frame bond longer than this multiple of its reference length counts as broken.
BOND_STRETCH_LIMIT = 1.5

_STATE_ARRAYS = ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "t2_mean", "t2_inv_cov")
_TRACK_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


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


@dataclass(frozen=True)
class Exceedance:
    """For one threshold role: per frame, the largest score ÷ threshold over the
    active tests, and the test that produced it.

    Above 1 = the frame is marked. Its size is not comparable across tests; only
    "above 1" has a statistical meaning.

    values : (n_frames,)
    tests : per frame, the (track, score) of the largest ratio
    """

    role: str
    values: np.ndarray
    tests: tuple[tuple[str, str], ...]

    @property
    def marked(self) -> np.ndarray:
        return self.values > 1.0


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(
    train: np.ndarray,
    valid: np.ndarray,
    topology: MoleculeTopology,
    *,
    tracks: Mapping[str, Sequence[str]] | None = None,
    scores: Sequence[str] = SCORES,
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
    topology : from ``build_topology``; fixes atom order, elements and the
        built-in descriptors' numbers.
    tracks : track name → descriptor references (a registered name such as
        ``"bonds"`` or ``"module.path:function"``; the module is imported).
        Default ``DEFAULT_TRACKS`` (provisional). A track whose descriptors give
        no numbers (ring on a ring-less molecule) is dropped and recorded.
    scores : which of ``"T2"``, ``"SPE"`` to compute for every track.
    pca_variance, random_seed : PCA settings, shared by all tracks.
    percentiles : threshold role → percentile, the *total* budget of that role
        over all active tests; default ``{"bead": 99.0, "centroid": 99.0}``.
    window_frames, step_frames, fraction_threshold : the window onset rule.
    length_unit : only ``"angstrom"`` is supported.
    """
    _check_length_unit(length_unit)
    if percentiles is None:
        percentiles = {"bead": 99.0, "centroid": 99.0}
    for role, p in percentiles.items():
        if not 0.0 < float(p) < 100.0:
            raise ValueError(f"percentile of role {role!r} must be in (0, 100), got {p}")
    scores = tuple(scores)
    if not scores or any(s not in SCORES for s in scores) or len(set(scores)) != len(scores):
        raise ValueError(f"scores must be a non-empty subset of {SCORES}, got {scores}")
    track_spec = _track_spec(DEFAULT_TRACKS if tracks is None else tracks)
    rule = WindowRule(int(window_frames), int(step_frames), float(fraction_threshold))
    if len(topology.bond_names) != len(topology.bonds):
        topology = complete_topology(copy.deepcopy(topology))
    train = _as_frames(train, topology.n_atoms, "train")
    valid = _as_frames(valid, topology.n_atoms, "valid")
    if topology.reference_positions is not None:
        reference = _as_frames(topology.reference_positions[None], topology.n_atoms, "reference")[0]
        reference_source = "topology"
    else:
        reference = train[0]
        reference_source = "first training frame"

    # ── Descriptors: each computed once, however many tracks use it ──────────
    regs: dict[str, RegisteredDescriptor] = {}
    for refs in track_spec.values():
        for ref in refs:
            reg = resolve_descriptor(ref, allow_import=True)
            if reg.name in regs and regs[reg.name].reference != reg.reference:
                raise ValueError(f"two descriptors are called {reg.name!r}")
            regs[reg.name] = reg
    d_train = {n: compute(r, train, topology) for n, r in regs.items()}
    d_valid = {n: compute(r, valid, topology) for n, r in regs.items()}

    # ── Tracks: fit, score validation ────────────────────────────────────────
    track_manifests: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    valid_scores: dict[tuple[str, str], np.ndarray] = {}
    dropped: dict[str, str] = {}
    for track, refs in track_spec.items():
        names = [resolve_descriptor(r, allow_import=True).name for r in refs]
        layout = _track_layout(track, names, d_train)
        if not layout["feature_names"]:
            dropped[track] = "its descriptors give no numbers for this molecule"
            continue
        X_train = np.concatenate([d_train[n].values for n in names], axis=1)
        X_valid = np.concatenate([d_valid[n].values for n in names], axis=1)
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
        n_features, n_components = X_train.shape[1], state["pca_components"].shape[0]
        score_info: dict[str, dict[str, Any]] = {}
        if "T2" in scores:
            valid_scores[(track, "T2")] = _t2(X_valid, state)
            score_info["T2"] = {"active": True}
        if "SPE" in scores:
            spe = _spe(X_valid, state)
            if n_components >= n_features:
                score_info["SPE"] = {"active": False, "reason": "PCA keeps every direction"}
            elif float(spe.max()) < SPE_FLOOR:
                score_info["SPE"] = {"active": False, "reason": "no residual variance in validation"}
            else:
                valid_scores[(track, "SPE")] = spe
                score_info["SPE"] = {"active": True}
        track_manifests[track] = {
            **layout,
            "n_features": int(n_features),
            "n_components": int(n_components),
            "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
            "parameters": {"pca_variance": float(pca_variance), "random_seed": int(random_seed)},
            "scores": score_info,
            "state_keys": [f"{track}/{k}" for k in _STATE_ARRAYS],
        }
        arrays.update({f"{track}/{k}": v for k, v in state.items()})
    if not track_manifests:
        raise ValueError(f"every track is empty for this molecule: {dropped}")

    # ── Budget: Bonferroni over the active tests, per role ───────────────────
    tests = list(valid_scores)
    if not tests:
        raise ValueError("no active test: every requested score is inactive")
    k = len(tests)
    n_valid = len(valid)
    budget: dict[str, Any] = {"rule": BUDGET_RULE, "tests": [list(t) for t in tests], "n_tests": k, "roles": {}}
    for role, p in percentiles.items():
        alpha = 1.0 - float(p) / 100.0
        p_test = 100.0 * (1.0 - alpha / k)
        expected = n_valid * alpha / k
        budget["roles"][str(role)] = {
            "percentile": float(p), "alpha": alpha, "alpha_per_test": alpha / k,
            "percentile_per_test": p_test, "expected_valid_exceedances_per_test": expected,
        }
        if expected < 1.0:
            warnings.warn(
                f"role {role!r}: {n_valid} validation frames × α/K = {alpha:.3g}/{k} gives "
                f"{expected:.2f} expected exceedances per test (< 1); each threshold is "
                "close to the largest validation value",
                UserWarning, stacklevel=2,
            )
        for track, score in tests:
            thresholds = track_manifests[track]["scores"][score].setdefault("thresholds", {})
            thresholds[str(role)] = {
                "percentile_total": float(p), "percentile_test": p_test,
                "value": float(np.percentile(valid_scores[(track, score)], p_test)),
            }

    used = {n for t in track_manifests.values() for n in t["descriptors"]}
    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
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
        "descriptors": {
            n: {
                "reference": r.reference,
                "builtin": r.builtin,
                "code_sha256": r.code_hash(),
                "feature_names": list(d_train[n].names),
            }
            for n, r in regs.items() if n in used
        },
        "scores": list(scores),
        "tracks": track_manifests,
        "dropped_tracks": dropped,
        "budget": budget,
        "onset_rule": rule.to_dict(),
        "fit_data": {"n_train": int(len(train)), "n_valid": int(n_valid)},
    }
    arrays[REFERENCE_KEY] = np.array(reference, dtype=np.float64)
    return ReferenceBundle(manifest, arrays)


def _track_spec(tracks: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not tracks:
        raise ValueError("no tracks given")
    spec: dict[str, tuple[str, ...]] = {}
    for name, refs in tracks.items():
        if not _TRACK_NAME.match(str(name)):
            raise ValueError(f"track name {name!r}: use letters, digits, '_', '.', '-'")
        if isinstance(refs, str) or not refs:
            raise ValueError(f"track {name!r}: give a non-empty list of descriptor references")
        spec[str(name)] = tuple(str(r) for r in refs)
    return spec


def _track_layout(track: str, names: Sequence[str], values: Mapping[str, DescriptorValues]) -> dict[str, Any]:
    """Column names, atoms and group IDs (scoped ``descriptor:group``) of a track."""
    feature_names: list[str] = []
    atoms: list[list[int]] = []
    groups: list[str] = []
    for n in names:
        v = values[n]
        feature_names += v.names
        atoms += [list(a) for a in v.atoms]
        groups += [f"{n}:{g}" for g in v.groups]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"track {track!r}: two of its descriptors use the same number name")
    return {"descriptors": list(names), "feature_names": feature_names,
            "feature_atoms": atoms, "feature_groups": groups}


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
        self._descriptors = {
            n: resolve_descriptor(d["reference"], allow_import=False)
            for n, d in manifest["descriptors"].items()
        }

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

    @property
    def tests(self) -> tuple[tuple[str, str], ...]:
        """The active (track, score) pairs; the false-alarm budget is split over these."""
        return tuple((t, s) for t, s in self._manifest["budget"]["tests"])

    @property
    def roles(self) -> tuple[str, ...]:
        """Threshold roles, e.g. ``("bead", "centroid")``."""
        return tuple(self._manifest["budget"]["roles"])

    def descriptors(self, track: str) -> tuple[str, ...]:
        """Names of the descriptors ``track`` is built from."""
        return tuple(self._track(track)["descriptors"])

    def feature_names(self, track: str) -> tuple[str, ...]:
        """Descriptor number names of ``track``, in column order."""
        return tuple(self._track(track)["feature_names"])

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        """Column order of ``compute_features``: every descriptor's numbers, in turn."""
        return tuple(n for d in self._manifest["descriptors"].values() for n in d["feature_names"])

    def thresholds(self, track: str) -> dict[str, dict[str, float]]:
        """Calibrated thresholds of ``track``'s active scores: {score: {role: value}}."""
        return {
            score: {role: float(t["value"]) for role, t in info["thresholds"].items()}
            for score, info in self._track(track)["scores"].items() if info["active"]
        }

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

    def compute_features(self, positions: np.ndarray, *, length_unit: str = "angstrom") -> np.ndarray:
        """Every descriptor's numbers, (n_frames, len(all_feature_names)).

        Raises ``BundleError`` if a descriptor now returns other numbers than at
        fit time.
        """
        pos = self._frames(positions, length_unit)
        blocks = []
        for name, reg in self._descriptors.items():
            out = compute(reg, pos, self._topology)
            expected = tuple(self._manifest["descriptors"][name]["feature_names"])
            if out.names != expected:
                raise BundleError(
                    f"descriptor {name!r} returns numbers {list(out.names)[:3]}…; "
                    f"the bundle was fitted on {list(expected)[:3]}…. Refit the bundle."
                )
            blocks.append(out.values)
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(pos), 0))

    def score(self, positions: np.ndarray, *, length_unit: str = "angstrom") -> dict[str, dict[str, np.ndarray]]:
        """Scores per frame: ``{track: {"T2": (n,), "SPE": (n,)}}``.

        An inactive score's key is absent.

        Parameters
        ----------
        positions : (n_frames, n_atoms, 3), or (n_atoms, 3) for one frame, in the
            bundle's atom order.
        """
        return self.score_features(self.compute_features(positions, length_unit=length_unit))

    def score_features(self, features: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """As ``score``, from ``compute_features`` output (for callers that cache it).

        The columns must be in ``all_feature_names`` order; only the width is checked.
        """
        X = np.asarray(features, dtype=np.float64)
        width = len(self.all_feature_names)
        if X.ndim != 2 or X.shape[1] != width:
            raise BundleError(f"features have shape {X.shape}; expected (n_frames, {width})")
        out: dict[str, dict[str, np.ndarray]] = {}
        for track in self.tracks:
            Xt = X[:, self._columns(track)]
            state = self._track_state(track)
            info = self._track(track)["scores"]
            out[track] = {}
            if info.get("T2", {}).get("active"):
                out[track]["T2"] = _t2(Xt, state)
            if info.get("SPE", {}).get("active"):
                out[track]["SPE"] = _spe(Xt, state)
        return out

    def exceedance(self, scores: Mapping[str, Mapping[str, np.ndarray]], role: str = "bead") -> Exceedance:
        """Largest score ÷ threshold over the active tests, per frame, for ``role``.

        ``scores`` is what ``score`` or ``score_features`` returned. Above 1 =
        the frame is marked for ``role``; the window rule reads this.
        """
        if role not in self.roles:
            raise KeyError(f"unknown threshold role {role!r}; bundle has {', '.join(self.roles)}")
        tests = self.tests
        ratios = np.stack([
            np.asarray(scores[t][s], dtype=np.float64) / self.thresholds(t)[s][role] for t, s in tests
        ])
        best = np.argmax(ratios, axis=0)
        return Exceedance(role, ratios[best, np.arange(ratios.shape[1])], tuple(tests[i] for i in best))

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
    def load(cls, path: str | Path, *, descriptor_modules: Sequence[str] = ()) -> ReferenceBundle:
        """Read a bundle directory written by ``save``.

        A user descriptor's module is never imported because a bundle names it:
        import it first, or list it in ``descriptor_modules`` to have it imported
        here.

        Raises ``BundleError`` when the bundle cannot be trusted with this
        installation: another format version, a descriptor that is not imported
        or whose code changed since the fit, unsupported units, or a manifest and
        fitted arrays that do not agree with each other.
        """
        path = Path(path)
        for module in descriptor_modules:
            try:
                importlib.import_module(module)
            except ImportError as exc:
                raise BundleError(f"cannot import descriptor module {module!r}: {exc}") from exc
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

    def _columns(self, track: str) -> np.ndarray:
        """Indices of ``track``'s numbers within ``all_feature_names``."""
        start, offsets = 0, {}
        for name, d in self._manifest["descriptors"].items():
            offsets[name] = start
            start += len(d["feature_names"])
        return np.concatenate([
            np.arange(offsets[n], offsets[n] + len(self._manifest["descriptors"][n]["feature_names"]))
            for n in self.descriptors(track)
        ]).astype(np.intp)

    def _frames(self, positions: np.ndarray, length_unit: str) -> np.ndarray:
        self._check_unit(length_unit)
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 2:
            pos = pos[None]
        try:
            return _as_frames(pos, self.n_atoms, "positions", allow_empty=True)
        except ValueError as exc:
            raise BundleError(str(exc)) from exc

    def _check_unit(self, length_unit: str) -> None:
        if length_unit != self.length_unit:
            raise BundleError(
                f"frames are in {length_unit!r}; bundle expects {self.length_unit!r}"
            )


def load(path: str | Path, *, descriptor_modules: Sequence[str] = ()) -> ReferenceBundle:
    """Load a reference bundle directory; see ``ReferenceBundle.load``."""
    return ReferenceBundle.load(path, descriptor_modules=descriptor_modules)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardised(X: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standardised descriptor, centred on the PCA mean (z in attribution.py)."""
    return (X - state["scaler_mean"]) / state["scaler_scale"] - state["pca_mean"]


def _t2(X: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standardise, project onto the kept PCA directions, return T² per row."""
    Z = _standardised(X, state) @ state["pca_components"].T
    diff = Z - state["t2_mean"]
    return np.maximum(np.einsum("ij,jk,ik->i", diff, state["t2_inv_cov"], diff), 0.0)


def _spe(X: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Squared PCA residual per row, in standardised units."""
    z = _standardised(X, state)
    P = state["pca_components"]
    r = z - (z @ P.T) @ P
    return np.einsum("ij,ij->i", r, r)


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
    units = manifest.get("units")
    if units != UNITS:
        raise BundleError(f"bundle units {units!r}; supported unit set is {UNITS!r}")

    try:
        mol = manifest["molecule"]
        n_atoms = int(mol["n_atoms"])
        elements = list(mol["elements"])
        topo_dict = mol["topology"]
        tracks = manifest["tracks"]
        descriptors = manifest["descriptors"]
        budget = manifest["budget"]
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
    try:
        topo = _topology_from_manifest(manifest)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BundleError(f"manifest: invalid topology ({exc})") from exc

    ref_key = mol.get("reference_geometry", {}).get("state_key")
    if ref_key != REFERENCE_KEY or REFERENCE_KEY not in state:
        raise BundleError("fitted state: missing the reference geometry")
    ref = state[REFERENCE_KEY]
    if ref.shape != (n_atoms, 3) or ref.dtype.kind != "f" or not np.all(np.isfinite(ref)):
        raise BundleError(f"fitted state: reference geometry has shape {ref.shape}, expected ({n_atoms}, 3)")

    # ── Descriptors: importable, same code, same numbers ─────────────────────
    for name, d in descriptors.items():
        try:
            reg = resolve_descriptor(str(d["reference"]), allow_import=False)
        except (LookupError, ValueError) as exc:
            raise BundleError(f"descriptor {name!r}: {exc}") from exc
        if reg.name != name:
            raise BundleError(f"descriptor {name!r}: {d['reference']} is registered as {reg.name!r}")
        if d.get("code_sha256") != reg.code_hash():
            raise BundleError(
                f"descriptor {name!r}: descriptor code differs from the code that fitted "
                f"this bundle (fitted with DetectAna {manifest.get('detectana_version')}, "
                f"installed {__version__}). Refit the bundle."
            )
        if reg.builtin:
            # Built-ins can be checked now: the topology fixes their numbers.
            probe = compute(reg, ref[None], topo)
            if list(probe.names) != list(d.get("feature_names", [])):
                raise BundleError(f"descriptor {name!r}: feature names do not match the topology")

    # ── Tracks: layout, scores, state arrays ─────────────────────────────────
    if not tracks:
        raise BundleError("manifest: no tracks")
    active: list[tuple[str, str]] = []
    for track, t in tracks.items():
        if not _TRACK_NAME.match(str(track)):
            raise BundleError(f"manifest: invalid track name {track!r}")
        names = t.get("descriptors", [])
        if not names or any(n not in descriptors for n in names):
            raise BundleError(f"track {track!r}: unknown descriptors {names}")
        expected_names = [f for n in names for f in descriptors[n]["feature_names"]]
        n_features = len(expected_names)
        if (t.get("feature_names") != expected_names or t.get("n_features") != n_features
                or len(t.get("feature_atoms", [])) != n_features or len(t.get("feature_groups", [])) != n_features):
            raise BundleError(f"track {track!r}: {len(t.get('feature_names', []))} feature names, "
                              f"its descriptors give {n_features}")
        for atoms in t["feature_atoms"]:
            if not atoms or any(not 0 <= int(i) < n_atoms for i in atoms):
                raise BundleError(f"track {track!r}: feature atoms {atoms} outside 0..{n_atoms - 1}")
        expected_keys = [f"{track}/{k}" for k in _STATE_ARRAYS]
        missing = [k for k in expected_keys if k not in state]
        if missing or t.get("state_keys") != expected_keys:
            raise BundleError(f"fitted state: missing arrays {missing or t.get('state_keys')}")
        n_comp = int(t.get("n_components", -1))
        shapes = {
            "scaler_mean": (n_features,), "scaler_scale": (n_features,), "pca_mean": (n_features,),
            "pca_components": (n_comp, n_features), "t2_mean": (n_comp,), "t2_inv_cov": (n_comp, n_comp),
        }
        for key, shape in shapes.items():
            arr = state[f"{track}/{key}"]
            if arr.shape != shape:
                raise BundleError(f"fitted state: {track}/{key} has shape {arr.shape}, manifest implies {shape}")
            if arr.dtype.kind != "f" or not np.all(np.isfinite(arr)):
                raise BundleError(f"fitted state: {track}/{key} is not a finite float array")
        for score, info in t.get("scores", {}).items():
            if score not in SCORES:
                raise BundleError(f"track {track!r}: unknown score {score!r}")
            if not info.get("active"):
                continue
            active.append((track, score))
            for role, thr in info.get("thresholds", {}).items():
                if not np.isfinite(float(thr.get("value", np.nan))):
                    raise BundleError(f"track {track!r}: {score} threshold {role!r} is not a finite number")
            if set(info.get("thresholds", {})) != set(budget.get("roles", {})):
                raise BundleError(f"track {track!r}: {score} thresholds do not cover every role")

    if [tuple(x) for x in budget.get("tests", [])] != active or budget.get("n_tests") != len(active):
        raise BundleError("manifest: the budget's tests are not the active scores")
    if budget.get("rule") != BUDGET_RULE:
        raise BundleError(f"manifest: unknown budget rule {budget.get('rule')!r}")



def _topology_from_manifest(manifest: Mapping[str, Any]) -> MoleculeTopology:
    """Rebuild the topology the built-in descriptors need from the manifest alone."""
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
    return complete_topology(topo)
