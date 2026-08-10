"""Tests for evaluation.py — detection metrics, error correlation, distortions.

The distortion tests matter most: the benchmark's conclusions rest on the claim
that a dihedral rotation changes *only* the torsion, so that is asserted directly
against bond lengths and bond angles rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. Conformal threshold
# ---------------------------------------------------------------------------

def test_conformal_threshold_is_the_kth_largest():
    from detectana.evaluation import conformal_threshold

    scores = np.arange(1.0, 101.0)          # 1 … 100, n = 100
    # k = ceil(101 * 0.05) = 6 → 6th largest = 95
    assert conformal_threshold(scores, 0.05) == 95.0


def test_conformal_threshold_flags_at_most_alpha():
    from detectana.evaluation import conformal_threshold

    rng = np.random.default_rng(0)
    scores = rng.normal(size=2000)
    thr = conformal_threshold(scores, 0.01)
    assert (scores > thr).mean() <= 0.01 + 1e-9


def test_conformal_threshold_rejects_bad_input():
    from detectana.evaluation import conformal_threshold

    with pytest.raises(ValueError, match="at least one"):
        conformal_threshold(np.array([]), 0.01)
    with pytest.raises(ValueError, match="false_flag_rate"):
        conformal_threshold(np.arange(10.0), 1.5)


# ---------------------------------------------------------------------------
# 2. Detection metrics
# ---------------------------------------------------------------------------

def test_perfect_separation():
    from detectana.evaluation import detection_metrics

    m = detection_metrics(np.arange(100.0), np.arange(1000.0, 1100.0), 0.01)
    assert m["auroc"] == 1.0
    assert m["detection_rate"] == 1.0
    assert m["average_precision"] == pytest.approx(1.0)


def test_identical_distributions_score_at_chance():
    from detectana.evaluation import detection_metrics

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    m = detection_metrics(a, b, 0.01)
    assert 0.45 < m["auroc"] < 0.55
    # A detector with no signal flags abnormal frames at the false-flag rate.
    assert m["detection_rate"] < 0.03


def test_empty_group_returns_none_metrics():
    from detectana.evaluation import detection_metrics

    m = detection_metrics(np.arange(10.0), np.array([]), 0.01)
    assert m["auroc"] is None
    assert m["detection_rate"] is None
    assert m["n_abnormal"] == 0


# ---------------------------------------------------------------------------
# 3. Force error
# ---------------------------------------------------------------------------

def test_force_error_metrics_on_known_values():
    from detectana.evaluation import per_frame_force_error

    ref = np.zeros((1, 2, 3))
    pred = np.array([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])

    # components are 3, 4, 0, 0, 0, 0 → mae = 7/6, rmse = sqrt(25/6)
    assert per_frame_force_error(ref, pred, "mae")[0] == pytest.approx(7 / 6)
    assert per_frame_force_error(ref, pred, "rmse")[0] == pytest.approx(np.sqrt(25 / 6))
    # largest per-atom vector norm error is |(3,4,0)| = 5
    assert per_frame_force_error(ref, pred, "max")[0] == pytest.approx(5.0)


def test_identical_forces_give_zero_error():
    from detectana.evaluation import per_frame_force_error

    rng = np.random.default_rng(2)
    f = rng.normal(size=(5, 4, 3))
    assert np.allclose(per_frame_force_error(f, f), 0.0)


def test_force_error_rejects_mismatch_and_bad_metric():
    from detectana.evaluation import per_frame_force_error

    with pytest.raises(ValueError, match="shape mismatch"):
        per_frame_force_error(np.zeros((2, 3, 3)), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="Unknown metric"):
        per_frame_force_error(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), "median")


# ---------------------------------------------------------------------------
# 4. Score/error relationship
# ---------------------------------------------------------------------------

def test_monotone_relationship_gives_spearman_one():
    from detectana.evaluation import score_error_correlation

    scores = np.arange(50.0)
    errors = np.exp(scores / 10.0)          # monotone but far from linear
    result = score_error_correlation(scores, errors)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] < 1.0          # which is the point of using Spearman


def test_no_relationship_gives_near_zero():
    from detectana.evaluation import score_error_correlation

    rng = np.random.default_rng(3)
    result = score_error_correlation(rng.normal(size=2000), rng.normal(size=2000))
    assert abs(result["spearman"]) < 0.1


def test_deciles_increase_with_a_real_relationship():
    from detectana.evaluation import error_by_score_decile

    scores = np.arange(1000.0)
    errors = scores * 2.0
    table = error_by_score_decile(scores, errors, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 1000
    assert table["error_mean"].is_monotonic_increasing
    assert table["error_mean"].iloc[-1] > 10 * table["error_mean"].iloc[0]


def test_deciles_flat_when_score_is_uninformative():
    from detectana.evaluation import error_by_score_decile

    rng = np.random.default_rng(4)
    table = error_by_score_decile(rng.normal(size=5000), rng.normal(10, 1, 5000))
    spread = table["error_mean"].max() - table["error_mean"].min()
    assert spread < 0.5      # all bins near the common mean of 10


# ---------------------------------------------------------------------------
# 5. Distortions — the benchmark's validity rests on these
# ---------------------------------------------------------------------------

def _bond_lengths(pos, topo):
    return np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in topo.bonds])


def _bond_angles(pos, topo):
    out = []
    for i, j, k in topo.angles:
        v1, v2 = pos[i] - pos[j], pos[k] - pos[j]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        out.append(np.arccos(np.clip(cos, -1, 1)))
    return np.array(out)


def test_rotation_preserves_every_bond_and_angle(topo, train_positions):
    """The central claim: a torsion rotation is a pure conformational change.

    If this failed, the benchmark's "rotate" column would be measuring bond and
    angle distortion instead of conformational novelty.
    """
    from detectana.evaluation import perturb_dihedral

    pos = train_positions[0]
    rotated = perturb_dihedral(pos, topo, 90.0)

    np.testing.assert_allclose(_bond_lengths(rotated, topo), _bond_lengths(pos, topo), atol=1e-9)
    np.testing.assert_allclose(_bond_angles(rotated, topo), _bond_angles(pos, topo), atol=1e-9)
    assert not np.allclose(rotated, pos)     # something did move


def test_rotation_changes_the_target_torsion_by_the_requested_amount(topo, train_positions):
    from detectana.evaluation import (
        _most_balanced_dihedral,
        dihedral_angles_deg,
        perturb_dihedral,
    )

    quad = _most_balanced_dihedral(topo)
    pos = train_positions[0]
    before = dihedral_angles_deg(pos[np.newaxis], quad)[0]
    after = dihedral_angles_deg(perturb_dihedral(pos, topo, 30.0)[np.newaxis], quad)[0]
    delta = (after - before + 180) % 360 - 180
    assert abs(abs(delta) - 30.0) < 1e-6


def test_set_dihedral_hits_the_target_angle(topo, train_positions):
    from detectana.evaluation import dihedral_angles_deg, most_gap_rich_dihedral, set_dihedral

    quad, _ = most_gap_rich_dihedral(train_positions, topo)
    moved = set_dihedral(train_positions[:5], topo, quad, -105.0)
    np.testing.assert_allclose(dihedral_angles_deg(moved, quad), -105.0, atol=1e-6)


def test_bond_stretch_lengthens_its_bond(topo, train_positions):
    from detectana.evaluation import perturb_bond_stretch

    pos = train_positions[0]
    stretched = perturb_bond_stretch(pos, topo, 0.5, bond_index=0)
    before, after = _bond_lengths(pos, topo), _bond_lengths(stretched, topo)
    assert after[0] == pytest.approx(before[0] + 0.5, abs=1e-9)


def test_gaussian_rattle_is_reproducible_and_scales(topo, train_positions):
    from detectana.evaluation import perturb_gaussian

    pos = train_positions[0]
    a = perturb_gaussian(pos, 0.1, np.random.default_rng(7))
    b = perturb_gaussian(pos, 0.1, np.random.default_rng(7))
    np.testing.assert_allclose(a, b)

    small = np.abs(perturb_gaussian(pos, 0.01, np.random.default_rng(8)) - pos).mean()
    large = np.abs(perturb_gaussian(pos, 0.20, np.random.default_rng(8)) - pos).mean()
    assert large > 10 * small


# ---------------------------------------------------------------------------
# 6. Torsion coverage
# ---------------------------------------------------------------------------

def test_coverage_table_accounts_for_every_frame(topo, train_positions):
    from detectana.evaluation import _most_balanced_dihedral, torsion_coverage

    quad = _most_balanced_dihedral(topo)
    table = torsion_coverage(train_positions, topo, quad, n_bins=12)
    assert len(table) == 12
    assert table["train_count"].sum() == len(train_positions)
    assert (table["visited"] == (table["train_count"] > 0)).all()


def test_gap_rich_dihedral_wins_among_eligible_torsions(topo, train_positions):
    """The chosen torsion must be the gappiest of those it was allowed to consider.

    Eligibility mirrors the selector: no ring bonds (neither side can move) and no
    terminal bonds (rotating one hydrogen changes almost nothing).
    """
    from detectana.evaluation import (
        _fragment_beyond_bond,
        most_gap_rich_dihedral,
        torsion_coverage,
    )

    quad, table = most_gap_rich_dihedral(train_positions, topo, n_bins=12, min_fragment=3)
    gaps = int((~table["visited"]).sum())
    assert quad in topo.dihedrals

    ring = set(topo.ring_atoms)
    n_eligible = 0
    for other in topo.dihedrals:
        _, j, k, _ = other
        if j in ring and k in ring:
            continue
        moving = _fragment_beyond_bond(topo, j, k)
        if min(len(moving), topo.n_atoms - len(moving)) < 3:
            continue
        n_eligible += 1
        other_gaps = int(
            (~torsion_coverage(train_positions, topo, other, 12)["visited"]).sum()
        )
        assert gaps >= other_gaps

    assert n_eligible >= 2, "expected several rotatable torsions in aspirin"


def test_gap_rich_dihedral_raises_when_no_torsion_qualifies(ethanol_xyz):
    """No eligible torsion is an error, not a silently poor choice."""
    from detectana.evaluation import most_gap_rich_dihedral
    from detectana.io import load_single_frame
    from detectana.topology import build_topology

    ethanol_topo = build_topology(ethanol_xyz)
    positions = load_single_frame(ethanol_xyz).get_positions()[np.newaxis]

    # Ethanol has rotatable torsions, so the default succeeds …
    quad, table = most_gap_rich_dihedral(positions, ethanol_topo)
    assert quad in ethanol_topo.dihedrals
    assert len(table) == 12

    # … but demanding a fragment larger than the molecule leaves nothing eligible.
    with pytest.raises(ValueError, match="No rotatable"):
        most_gap_rich_dihedral(positions, ethanol_topo, min_fragment=99)
