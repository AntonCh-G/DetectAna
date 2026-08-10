"""Tests for the window-rule false-alarm arithmetic in onset.py.

The window rule, not the OOD threshold, controls false alarms, so these tests
pin down the arithmetic that makes the trade-off explicit:
- effective sample size under autocorrelation
- the false-alarm bound for a fixed rule
- deriving the loosest rule that fits a budget
- config resolution, including that the default path changes nothing
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

# ---------------------------------------------------------------------------
# 1. Effective trials
# ---------------------------------------------------------------------------

def test_independent_frames_give_full_window():
    from detectana.onset import effective_window_trials

    assert effective_window_trials(500, frame_autocorrelation=0.0) == 500


def test_autocorrelation_shrinks_effective_trials():
    from detectana.onset import effective_window_trials

    # AR(1): n * (1-rho)/(1+rho) → 500 * 0.63/1.37 ≈ 230
    assert effective_window_trials(500, frame_autocorrelation=0.37) == 230
    assert effective_window_trials(500, 0.9) < effective_window_trials(500, 0.5)


def test_negative_autocorrelation_is_clipped_to_independent():
    """Negative rho would raise n_eff; assuming independence is conservative."""
    from detectana.onset import effective_window_trials

    assert effective_window_trials(500, frame_autocorrelation=-0.5) == 500


def test_effective_trials_never_below_one():
    from detectana.onset import effective_window_trials

    assert effective_window_trials(1, frame_autocorrelation=0.99) == 1


def test_beads_multiply_trials():
    from detectana.onset import effective_window_trials

    assert effective_window_trials(100, 0.0, n_effective_beads=4) == 400


def test_effective_trials_rejects_bad_input():
    from detectana.onset import effective_window_trials

    with pytest.raises(ValueError, match="window_frames"):
        effective_window_trials(0)
    with pytest.raises(ValueError, match="n_effective_beads"):
        effective_window_trials(10, n_effective_beads=0)


# ---------------------------------------------------------------------------
# 2. False-alarm probability
# ---------------------------------------------------------------------------

def test_false_alarm_matches_binomial_tail():
    from detectana.onset import window_false_alarm_probability

    alpha, window, frac = 0.01, 500, 0.05
    expected = binom.sf(int(np.ceil(frac * window)) - 1, window, alpha)
    got = window_false_alarm_probability(alpha, window, frac, n_windows_tested=1)
    assert got == pytest.approx(expected)


def test_union_bound_over_windows():
    from detectana.onset import window_false_alarm_probability

    one = window_false_alarm_probability(0.01, 500, 0.05, n_windows_tested=1)
    many = window_false_alarm_probability(0.01, 500, 0.05, n_windows_tested=4000)
    assert many == pytest.approx(min(1.0, one * 4000))


def test_probability_is_capped_at_one():
    from detectana.onset import window_false_alarm_probability

    p = window_false_alarm_probability(0.10, 500, 0.10, n_windows_tested=10**6)
    assert p == 1.0


def test_stricter_fraction_lowers_false_alarms():
    from detectana.onset import window_false_alarm_probability

    loose = window_false_alarm_probability(0.01, 500, 0.05, 4000)
    strict = window_false_alarm_probability(0.01, 500, 0.20, 4000)
    assert strict < loose


def test_autocorrelation_raises_the_bound():
    """Correlated frames mean fewer effective trials, so more false alarms."""
    from detectana.onset import window_false_alarm_probability

    naive = window_false_alarm_probability(0.01, 500, 0.05, 4000, 0.0)
    honest = window_false_alarm_probability(0.01, 500, 0.05, 4000, 0.37)
    assert honest > naive


def test_the_projects_default_rule_is_very_conservative():
    """alpha=1%, window=500, fraction=0.20 over ~4000 windows."""
    from detectana.onset import window_false_alarm_probability

    p = window_false_alarm_probability(0.01, 500, 0.20, 4000)
    assert p < 1e-50


def test_false_alarm_rejects_bad_input():
    from detectana.onset import window_false_alarm_probability

    with pytest.raises(ValueError, match="false_flag_rate"):
        window_false_alarm_probability(0.0, 500, 0.1)
    with pytest.raises(ValueError, match="fraction_threshold"):
        window_false_alarm_probability(0.01, 500, 1.5)


# ---------------------------------------------------------------------------
# 3. Choosing the fraction from a budget
# ---------------------------------------------------------------------------

def test_chosen_fraction_meets_budget_and_is_minimal():
    from detectana.onset import choose_fraction_threshold, window_false_alarm_probability

    budget, alpha, window, windows = 0.01, 0.01, 500, 4000
    frac, p = choose_fraction_threshold(budget, alpha, window, windows)
    assert p <= budget

    # One step looser breaks the budget → the choice really is the loosest one.
    looser = frac - 1.0 / window
    assert window_false_alarm_probability(alpha, window, looser, windows) > budget


def test_chosen_fraction_is_looser_than_the_hand_picked_default():
    """The point of the budget: recover sensitivity thrown away by fraction=0.20."""
    from detectana.onset import choose_fraction_threshold

    frac, _ = choose_fraction_threshold(0.01, 0.01, 500, 4000)
    assert frac < 0.20


def test_looser_budget_gives_looser_rule():
    from detectana.onset import choose_fraction_threshold

    tight, _ = choose_fraction_threshold(1e-6, 0.01, 500, 4000)
    loose, _ = choose_fraction_threshold(0.05, 0.01, 500, 4000)
    assert loose <= tight


def test_impossible_budget_raises():
    from detectana.onset import choose_fraction_threshold

    # Threshold flags half of all in-distribution frames: no rule can hold 1e-9.
    with pytest.raises(ValueError, match="No fraction_threshold meets"):
        choose_fraction_threshold(1e-9, 0.5, 4, 10**6)


# ---------------------------------------------------------------------------
# 4. Autocorrelation estimate
# ---------------------------------------------------------------------------

def test_white_noise_has_near_zero_autocorrelation():
    from detectana.onset import estimate_lag1_autocorrelation

    x = np.random.default_rng(0).normal(size=5000)
    assert estimate_lag1_autocorrelation(x) < 0.05


def test_ar1_autocorrelation_is_recovered():
    from detectana.onset import estimate_lag1_autocorrelation

    rng = np.random.default_rng(1)
    rho = 0.7
    x = np.zeros(20_000)
    for i in range(1, len(x)):
        x[i] = rho * x[i - 1] + rng.normal()
    assert estimate_lag1_autocorrelation(x) == pytest.approx(rho, abs=0.05)


def test_degenerate_series_gives_zero():
    from detectana.onset import estimate_lag1_autocorrelation

    assert estimate_lag1_autocorrelation(np.array([1.0, 2.0])) == 0.0
    assert estimate_lag1_autocorrelation(np.ones(100)) == 0.0


# ---------------------------------------------------------------------------
# 5. n_windows
# ---------------------------------------------------------------------------

def test_n_windows_counts_window_starts():
    from detectana.onset import n_windows

    assert n_windows(1000, 500, 50) == 11      # starts 0, 50, …, 500
    assert n_windows(500, 500, 50) == 1
    assert n_windows(499, 500, 50) == 0


# ---------------------------------------------------------------------------
# 6. Config resolution
# ---------------------------------------------------------------------------

def test_fixed_fraction_is_passed_through_unchanged():
    """Default config path must not alter the rule — only report on it."""
    from detectana.onset import resolve_onset_rule

    cfg = {"window_frames": 500, "step_frames": 50, "fraction_threshold": 0.20}
    frac, report = resolve_onset_rule(cfg, false_flag_rate=0.01, n_frames=10_000)
    assert frac == 0.20
    assert report["derived_from_false_alarm_budget"] is None
    assert report["false_alarm_probability_per_run"] < 1e-50


def test_budget_overrides_fraction_and_loosens_it():
    from detectana.onset import resolve_onset_rule

    cfg = {
        "window_frames": 500,
        "step_frames": 50,
        "fraction_threshold": 0.20,
        "false_alarm_budget": 0.01,
    }
    frac, report = resolve_onset_rule(cfg, false_flag_rate=0.01, n_frames=200_000)
    assert frac < 0.20
    assert report["derived_from_false_alarm_budget"] == 0.01
    assert report["false_alarm_probability_per_run"] <= 0.01


def test_auto_autocorrelation_uses_the_stable_series():
    from detectana.onset import resolve_onset_rule

    rng = np.random.default_rng(2)
    x = np.zeros(5000)
    for i in range(1, len(x)):
        x[i] = 0.8 * x[i - 1] + rng.normal()

    cfg = {
        "window_frames": 500,
        "step_frames": 50,
        "fraction_threshold": 0.20,
        "frame_autocorrelation": "auto",
    }
    _, report = resolve_onset_rule(cfg, 0.01, 10_000, stable_series=x)
    assert report["frame_autocorrelation"] == pytest.approx(0.8, abs=0.05)
    assert report["effective_trials_per_window"] < 500


def test_auto_without_a_series_falls_back_to_independent():
    from detectana.onset import resolve_onset_rule

    cfg = {
        "window_frames": 500,
        "step_frames": 50,
        "fraction_threshold": 0.20,
        "frame_autocorrelation": "auto",
    }
    _, report = resolve_onset_rule(cfg, 0.01, 10_000, stable_series=None)
    assert report["frame_autocorrelation"] == 0.0


def test_bad_autocorrelation_string_raises():
    from detectana.onset import resolve_onset_rule

    cfg = {
        "window_frames": 500,
        "step_frames": 50,
        "fraction_threshold": 0.20,
        "frame_autocorrelation": "measured",
    }
    with pytest.raises(ValueError, match="frame_autocorrelation"):
        resolve_onset_rule(cfg, 0.01, 10_000)


def test_report_records_what_the_threshold_costs():
    from detectana.onset import resolve_onset_rule

    cfg = {"window_frames": 500, "step_frames": 50, "fraction_threshold": 0.20}
    _, report = resolve_onset_rule(cfg, false_flag_rate=0.01, n_frames=200_000)
    assert report["expected_false_flags_per_frame_series"] == pytest.approx(2000)
    assert report["frames_to_first_false_flag"] == pytest.approx(100)
    assert report["n_windows_tested"] == 3991
    assert report["flags_needed_per_window"] == 100


def test_report_brackets_the_bound_with_disjoint_windows():
    """The overlapping-window bound is the conservative end of a range."""
    from detectana.onset import resolve_onset_rule

    cfg = {"window_frames": 500, "step_frames": 50, "fraction_threshold": 0.05}
    _, report = resolve_onset_rule(cfg, false_flag_rate=0.01, n_frames=200_000)

    assert report["n_disjoint_windows"] == 400
    assert (
        report["false_alarm_probability_per_run_nonoverlapping"]
        <= report["false_alarm_probability_per_run"]
    )
