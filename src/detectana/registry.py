"""Descriptor registry (ADR 0006).

A descriptor is a function marked ``@descriptor("name")`` that maps a batch of
frames and the molecule's topology to named numbers::

    @descriptor("bonds")
    def bonds(positions, topology) -> DescriptorValues:
        # positions: (n_frames, n_atoms, 3) Å, the bundle's atom order
        ...

Each number declares the atoms it involves and a *descriptor group*: the numbers
explained together by attribution (one bond; one dihedral's sin and cos). The
names, atoms and groups come back with every call; a reference bundle records
them at fit time and checks them at score time.

Parameters (a SOAP cutoff, a reference geometry) live in the descriptor's own
module, not in DetectAna: a different setting is a different function.

Built-ins — ``bonds``, ``angles``, ``dihedrals``, ``ring`` — are registered by
``detectana.descriptors``; side by side, in that order, they are the joined
internal-coordinate descriptor.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from detectana.topology import MoleculeTopology

#: Built-in descriptors, in the order that reproduces the joined descriptor.
BUILTIN_DESCRIPTORS = ("bonds", "angles", "dihedrals", "ring")

DescriptorFunction = Callable[[np.ndarray, "MoleculeTopology"], "DescriptorValues"]


@dataclass(frozen=True)
class DescriptorValues:
    """What a descriptor returns for a batch of frames.

    values : (n_frames, k) float64
    names : k unique names, in column order
    atoms : k tuples of 0-based atom indices, one per number (at least one atom)
    groups : k group IDs; numbers sharing an ID are explained together
    """

    values: np.ndarray
    names: tuple[str, ...]
    atoms: tuple[tuple[int, ...], ...]
    groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64))
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        object.__setattr__(self, "atoms", tuple(tuple(int(i) for i in a) for a in self.atoms))
        object.__setattr__(self, "groups", tuple(str(g) for g in self.groups))


@dataclass(frozen=True)
class RegisteredDescriptor:
    """A registered descriptor function and what its code hash covers."""

    name: str
    function: DescriptorFunction
    helpers: tuple[Callable[..., Any], ...] = ()

    @property
    def reference(self) -> str:
        """``module:function``, the form a config or manifest names it by."""
        return f"{self.function.__module__}:{self.function.__qualname__}"

    @property
    def builtin(self) -> bool:
        return self.function.__module__.startswith("detectana.")

    def code_hash(self) -> str:
        """SHA-256 of the descriptor's code.

        Built-ins: the source of the function and of the helpers listed when it
        was registered; the bundle records the DetectAna version beside it.

        User descriptors: the whole source file of the module defining the
        function, read from disk now, plus the installed version of the package
        it belongs to, when it has one. Any edit to that file changes the hash,
        so helpers in the same file are covered; code imported from *other* files
        of an editable install is not.

        A formatting edit changes the hash too (ADR 0006): such a bundle is
        refused and must be refitted.
        """
        h = hashlib.sha256()
        if self.builtin:
            for fn in (self.function, *self.helpers):
                h.update(f"{fn.__module__}.{fn.__qualname__}\n".encode())
                h.update(inspect.getsource(fn).encode())
            return h.hexdigest()
        module = sys.modules[self.function.__module__]
        source_file = inspect.getsourcefile(module)
        if source_file is None:
            raise ValueError(f"descriptor {self.name!r}: cannot find the source file of {module.__name__}")
        h.update(f"{self.reference}\n".encode())
        with open(source_file, "rb") as fh:
            h.update(fh.read())
        h.update(f"\npackage-version={_package_version(module.__name__)}\n".encode())
        return h.hexdigest()


_REGISTRY: dict[str, RegisteredDescriptor] = {}


def descriptor(
    name: str, *, helpers: Sequence[Callable[..., Any]] = ()
) -> Callable[[DescriptorFunction], DescriptorFunction]:
    """Register ``fn(positions, topology) -> DescriptorValues`` under ``name``.

    ``helpers`` are functions whose source is hashed with the descriptor's own.
    Registering a different function under a taken name raises ``ValueError``;
    re-registering the same function (a module reload) replaces it.
    """
    if not name or ":" in name:
        raise ValueError(f"descriptor name {name!r} must be non-empty and contain no ':'")

    def mark(fn: DescriptorFunction) -> DescriptorFunction:
        entry = RegisteredDescriptor(str(name), fn, tuple(helpers))
        old = _REGISTRY.get(name)
        if old is not None and old.reference != entry.reference:
            raise ValueError(
                f"descriptor {name!r} is already registered by {old.reference}; "
                f"cannot register {entry.reference} under the same name"
            )
        _REGISTRY[name] = entry
        fn.__detectana_descriptor__ = entry  # type: ignore[attr-defined]
        return fn

    return mark


def get_descriptor(ref: str) -> RegisteredDescriptor:
    """The registered descriptor called ``ref``."""
    _ensure_builtins()
    try:
        return _REGISTRY[ref]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"no descriptor {ref!r} is registered (registered: {known})") from None


def resolve_descriptor(ref: str, *, allow_import: bool) -> RegisteredDescriptor:
    """The descriptor a config or manifest names: a registered name such as
    ``"bonds"``, or ``"module.path:function"`` for a user descriptor.

    ``allow_import`` says whether the module may be imported here. Fitting from a
    config allows it (the user named the module); loading a bundle does not
    (a bundle is data, and an import runs code), so its modules must already be
    imported. A ``LookupError`` names the module either way.
    """
    if ":" not in ref:
        return get_descriptor(ref)
    module_name, _, attr = ref.partition(":")
    if module_name in sys.modules:
        module = sys.modules[module_name]
    elif allow_import:
        module = importlib.import_module(module_name)
    else:
        raise LookupError(
            f"descriptor {ref!r} needs module {module_name!r}, which is not imported. "
            "Import it first, or pass it in descriptor_modules."
        )
    obj: Any = module
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise LookupError(f"descriptor {ref!r}: {module_name} has no attribute {attr!r}")
        obj = getattr(obj, part)
    entry = getattr(obj, "__detectana_descriptor__", None)
    if not isinstance(entry, RegisteredDescriptor):
        raise ValueError(f"{ref!r} is not marked with @descriptor")
    return entry


def compute(reg: RegisteredDescriptor, positions: np.ndarray, topology: MoleculeTopology) -> DescriptorValues:
    """Run a descriptor on ``positions`` (n_frames, n_atoms, 3) and check its output."""
    out = reg.function(positions, topology)
    if not isinstance(out, DescriptorValues):
        raise ValueError(f"descriptor {reg.name!r} returned {type(out).__name__}, not DescriptorValues")
    n_frames = len(positions)
    v = out.values
    if v.ndim != 2 or v.shape[0] != n_frames:
        raise ValueError(f"descriptor {reg.name!r}: values have shape {v.shape}; expected ({n_frames} frames, k)")
    k = v.shape[1]
    for field_name in ("names", "atoms", "groups"):
        if len(getattr(out, field_name)) != k:
            raise ValueError(
                f"descriptor {reg.name!r}: {len(getattr(out, field_name))} {field_name} for {k} numbers"
            )
    if len(set(out.names)) != k:
        raise ValueError(f"descriptor {reg.name!r}: number names are not unique")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"descriptor {reg.name!r}: values are not all finite")
    n_atoms = topology.n_atoms
    for name, atoms in zip(out.names, out.atoms):
        if not atoms or any(not 0 <= i < n_atoms for i in atoms):
            raise ValueError(
                f"descriptor {reg.name!r}: number {name!r} names atom(s) {atoms}; "
                f"need at least one atom index in 0..{n_atoms - 1}"
            )
    return out


def _package_version(module_name: str) -> str | None:
    """Installed version of the distribution providing ``module_name``, if any."""
    from importlib import metadata

    top = module_name.partition(".")[0]
    try:
        dists = metadata.packages_distributions().get(top, [])
    except Exception:  # metadata scanning must never break scoring
        return None
    for dist in dists:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return None


def _ensure_builtins() -> None:
    import detectana.descriptors  # noqa: F401  (registers the built-ins on import)
