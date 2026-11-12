"""The window onset rule, fed one frame at a time, must match batch ``detect_onset``.

A live caller sees frames one by one; the batch pipeline sees the whole run. Both
must declare the same onset frame on the same data, exactly, not approximately.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from detectana.onset import WindowRule, _windowed_onset, detect_onset


def _stream_onset(rule: WindowRule, signal: np.ndarray) -> int | None:
    stream = rule.stream()
    result = None
    for value in signal:
        result = stream.push(float(value))
    return result


SIGNAL_LENGTHS = [0, 1, 7, 20, 63, 200]
RULES = [
    (1, 1, 0.5),
    (5, 1, 0.2),
    (5, 3, 0.4),
    (10, 10, 0.3),
    (10, 15, 0.3),   # step longer than the window: gaps between windows
    (20, 7, 0.25),
    (50, 5, 0.1),
]


@pytest.mark.parametrize("window,step,frac", RULES)
@pytest.mark.parametrize("n", SIGNAL_LENGTHS)
@pytest.mark.parametrize("kind", ["flags", "fractions"])
def test_stream_equals_batch(window, step, frac, n, kind):
    for seed in range(12):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.0, 0.6)
        if kind == "flags":
            signal = (rng.random(n) < p).astype(np.float64)
        else:
            # bead_frac_ood is a multiple of 1/n_beads
            signal = rng.integers(0, 17, size=n) / 16.0 * (rng.random(n) < p)
        rule = WindowRule(window, step, frac)
        assert _stream_onset(rule, signal) == _windowed_onset(signal, window, step, frac)


def test_tie_exactly_at_the_fraction_fires_in_both():
    # 2 of 10 flagged = 0.2 exactly, and the rule uses >=.
    signal = np.zeros(10)
    signal[[3, 7]] = 1.0
    rule = WindowRule(10, 1, 0.2)
    assert _windowed_onset(signal, 10, 1, 0.2) == 0
    assert _stream_onset(rule, signal) == 0


def test_fewer_frames_than_a_window_never_fires():
    rule = WindowRule(10, 1, 0.1)
    assert _stream_onset(rule, np.ones(9)) is None


def test_onset_latches_after_first_fire():
    rule = WindowRule(4, 2, 0.5)
    stream = rule.stream()
    signal = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    results = [stream.push(v) for v in signal]
    first = next(r for r in results if r is not None)
    assert first == 0  # window [0, 4) holds two flags out of four
    assert all(r == first for r in results[results.index(first):])
    assert stream.onset_frame == first
    assert stream.n_seen == len(signal)


def test_stream_returns_none_until_the_rule_fires():
    stream = WindowRule(3, 1, 1.0).stream()
    assert [stream.push(v) for v in (1, 1, 0, 1, 1, 1)] == [None, None, None, None, None, 3]


def test_history_frames_is_the_window_length():
    assert WindowRule(500, 50, 0.2).history_frames == 500


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_frames": 0, "step_frames": 1, "fraction_threshold": 0.2},
        {"window_frames": 5, "step_frames": 0, "fraction_threshold": 0.2},
        {"window_frames": 5, "step_frames": 1, "fraction_threshold": 0.0},
        {"window_frames": 5, "step_frames": 1, "fraction_threshold": 1.5},
    ],
)
def test_invalid_rule_is_refused(kwargs):
    with pytest.raises(ValueError):
        WindowRule(**kwargs)


@pytest.mark.parametrize("window,step,frac", [(20, 5, 0.2), (50, 10, 0.3), (8, 3, 0.5)])
def test_stream_matches_detect_onset_on_an_aggregate_table(window, step, frac):
    """Same frames through detect_onset (batch) and two streams (bead, centroid)."""
    for seed, (lo, hi) in itertools.product(range(5), [(0.02, 0.8), (0.0, 0.3)]):
        rng = np.random.default_rng(100 + seed)
        n = 300
        # Flag probability ramps up, so onsets fall somewhere inside the run.
        p = np.linspace(lo, hi, n)
        bead_frac = (rng.random((n, 16)) < p[:, None]).mean(axis=1)
        centroid_ood = rng.random(n) < p
        agg = pd.DataFrame({
            "step": np.arange(n) * 50,
            "bead_frac_ood": bead_frac,
            "centroid_ood": centroid_ood,
        })
        batch = detect_onset(agg, threshold=1.0, window_frames=window,
                             step_frames=step, fraction_threshold=frac)
        rule = WindowRule(window, step, frac)
        assert _stream_onset(rule, bead_frac) == batch.persistent_bead_anomaly_frame
        assert _stream_onset(rule, centroid_ood.astype(float)) == batch.centroid_anomaly_frame
