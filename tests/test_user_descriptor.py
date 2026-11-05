"""User descriptor modules named as ``module:function`` (ADR 0006)."""

from __future__ import annotations

import sys
import textwrap
import uuid

import numpy as np
import pytest

from detectana.registry import compute, resolve_descriptor

USER_MODULE = '''
import numpy as np
from detectana.registry import DescriptorValues, descriptor

SCALE = {scale}


@descriptor("{name}")
def pair_distance(positions, topology):
    """Distance between atoms 0 and 1, times SCALE."""
    d = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1) * SCALE
    return DescriptorValues(d[:, None], ("d01",), ((0, 1),), ("d01",))


def not_a_descriptor(positions, topology):
    return None
'''


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    """Write a fresh user module to disk and put it on sys.path. Not imported yet."""
    tag = uuid.uuid4().hex[:8]
    module, name = f"userdesc_{tag}", f"pair_{tag}"
    path = tmp_path / f"{module}.py"
    path.write_text(textwrap.dedent(USER_MODULE.format(name=name, scale=1.0)))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module, name, path
    sys.modules.pop(module, None)


def test_a_reference_imports_and_registers_the_function(user_module, topo, train_positions):
    module, name, _ = user_module
    reg = resolve_descriptor(f"{module}:pair_distance", allow_import=True)
    assert reg.name == name
    assert reg.reference == f"{module}:pair_distance"
    assert not reg.builtin
    out = compute(reg, train_positions[:4], topo)
    expected = np.linalg.norm(train_positions[:4, 0] - train_positions[:4, 1], axis=-1)
    np.testing.assert_allclose(out.values[:, 0], expected)


def test_without_import_permission_an_unimported_module_is_refused(user_module):
    module, _, _ = user_module
    with pytest.raises(LookupError, match=module):
        resolve_descriptor(f"{module}:pair_distance", allow_import=False)
    assert module not in sys.modules


def test_an_already_imported_module_needs_no_permission(user_module):
    module, name, _ = user_module
    __import__(module)
    assert resolve_descriptor(f"{module}:pair_distance", allow_import=False).name == name


def test_a_plain_name_resolves_through_the_registry():
    assert resolve_descriptor("bonds", allow_import=False).name == "bonds"


def test_an_unmarked_function_is_refused(user_module):
    module, _, _ = user_module
    with pytest.raises(ValueError, match="@descriptor"):
        resolve_descriptor(f"{module}:not_a_descriptor", allow_import=True)


def test_a_missing_function_is_refused(user_module):
    module, _, _ = user_module
    with pytest.raises(LookupError, match="no_such"):
        resolve_descriptor(f"{module}:no_such", allow_import=True)


def test_user_hash_covers_the_whole_module_file(user_module):
    """Editing anything in the module file — here a constant the function reads —
    changes the hash, though the function's own source is unchanged."""
    module, name, path = user_module
    reg = resolve_descriptor(f"{module}:pair_distance", allow_import=True)
    before = reg.code_hash()
    assert before == reg.code_hash()
    path.write_text(textwrap.dedent(USER_MODULE.format(name=name, scale=2.0)))
    assert reg.code_hash() != before


# ---------------------------------------------------------------------------
# A bundle that uses a user descriptor
# ---------------------------------------------------------------------------

@pytest.fixture
def user_bundle(user_module, topo, train_positions, tmp_path):
    from detectana.bundle import fit

    module, name, path = user_module
    b = fit(train_positions[:64], train_positions[64:], topo,
            tracks={"bonds": ["bonds"], "pair": [f"{module}:pair_distance"]},
            percentiles={"bead": 80.0})
    saved = tmp_path / "user.bundle"
    b.save(saved)
    return b, saved, module, path


def test_bundle_records_the_user_descriptor(user_bundle):
    b, _, module, _ = user_bundle
    d = b.manifest["descriptors"][b.descriptors("pair")[0]]
    assert d["reference"] == f"{module}:pair_distance"
    assert d["builtin"] is False
    assert b.manifest["tracks"]["pair"]["feature_groups"] == [f"{b.descriptors('pair')[0]}:d01"]


def test_user_bundle_loads_and_scores_in_a_fresh_process(user_bundle, train_positions, tmp_path):
    """Nothing carried over from this process: the module is named explicitly."""
    import subprocess

    b, saved, module, path = user_bundle
    np.save(tmp_path / "frames.npy", train_positions[:5])
    expected = b.score(train_positions[:5])["pair"]["T2"]
    script = textwrap.dedent(f"""
        import sys, numpy as np, detectana
        sys.path.insert(0, {str(path.parent)!r})
        b = detectana.load({str(saved)!r}, descriptor_modules=[{module!r}])
        np.save({str(tmp_path / "out.npy")!r}, b.score(np.load({str(tmp_path / "frames.npy")!r}))["pair"]["T2"])
    """)
    src = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", script], env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)


def test_load_does_not_import_a_user_module_itself(user_bundle):
    from detectana.bundle import BundleError, load

    _, saved, module, _ = user_bundle
    sys.modules.pop(module)
    with pytest.raises(BundleError, match=module):
        load(saved)
    assert module not in sys.modules
    assert load(saved, descriptor_modules=[module]).tracks == ("bonds", "pair")


def test_load_refuses_when_the_user_module_changed(user_bundle):
    """Editing the descriptor's module file after the fit refuses the bundle."""
    from detectana.bundle import BundleError, load

    _, saved, module, path = user_bundle
    path.write_text(path.read_text().replace("SCALE = 1.0", "SCALE = 2.0"))
    with pytest.raises(BundleError, match="descriptor code differs"):
        load(saved)


def test_load_refuses_an_unimportable_descriptor_module(user_bundle):
    from detectana.bundle import BundleError, load

    _, saved, _, _ = user_bundle
    with pytest.raises(BundleError, match="no_such_module"):
        load(saved, descriptor_modules=["no_such_module"])
