"""The threshold override in scripts/extract_onset_frames.py reads the run's units."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_onset_frames.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("extract_onset_frames", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("statistic", ["exceedance", "T2"])
def test_current_and_t2_runs_are_accepted(script, tmp_path, statistic):
    (tmp_path / "manifest.json").write_text(json.dumps({"statistic": statistic}))
    assert script.saved_score_units(tmp_path / "manifest.json") == statistic


@pytest.mark.parametrize("manifest", [{"statistic": "distance"}, {}, None])
def test_distance_or_unknown_runs_are_refused(script, tmp_path, manifest):
    if manifest is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rerun run_pipeline"):
        script.saved_score_units(tmp_path / "manifest.json")
