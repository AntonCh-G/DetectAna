"""The core library imports and works without the [pipeline] extra (ADR 0005).

A fresh interpreter blocks pandas, matplotlib, h5py and pyyaml as if they were
not installed, then fits, saves, loads and scores a bundle and streams the onset
rule. No uninstall is needed. joblib is not blocked: it moved to the extra as a
direct dependency of DetectAna, but scikit-learn requires it, so every core
install has it anyway.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_ONLY = ("pandas", "matplotlib", "h5py", "yaml")

SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {blocked!r}

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}} (blocked)", name=name)
            return None

    sys.meta_path.insert(0, Block())
    for mod in list(sys.modules):
        if mod.split(".")[0] in BLOCKED:
            del sys.modules[mod]

    import tempfile
    import numpy as np
    from ase.io import read

    import detectana
    import detectana.bundle, detectana.descriptors, detectana.io, detectana.onset
    import detectana.scorer, detectana.topology, detectana.xyz_reader
    from detectana.topology import build_topology

    smoke = {smoke!r}
    train = np.array([a.positions for a in read(smoke + "/reference_train.xyz", index=":")])
    valid = np.array([a.positions for a in read(smoke + "/reference_valid.xyz", index=":")])
    topo = build_topology(smoke + "/initial.xyz")
    bundle = detectana.fit(train, valid, topo, window_frames=4, step_frames=1)
    with tempfile.TemporaryDirectory() as d:
        bundle.save(d + "/b")
        loaded = detectana.load(d + "/b")
    t2 = loaded.score(valid)["geometric"]
    assert t2.shape == (len(valid),)
    stream = loaded.onset_rule.stream()
    for value in t2 > loaded.thresholds("geometric")["bead"]:
        stream.push(float(value))

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    print("ok")
    """
)


def test_core_works_without_pipeline_dependencies():
    script = SCRIPT.format(blocked=PIPELINE_ONLY, smoke=str(ROOT / "data" / "smoke"))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
