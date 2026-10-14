"""Hand-built clock and estimator checks; times are ns, skill is bits."""

import numpy as np
import pytest

from sims.theory_checks import DELTA_1, DELTA_L, h1_cell, half_life, merge_events, ratio_estimate


def test_clock_keeps_deep_changes_and_trades_and_uses_last_tied_state():
    """A deep-size update counts even when the mid-price stays fixed."""
    books = np.array([
        [102, 2, 100, 3, 103, 4, 99, 5],
        [102, 2, 100, 3, 103, 4, 99, 6],  # deep size changes
        [104, 2, 100, 3, 103, 4, 99, 6],  # tied-time final book
        [104, 2, 100, 3, 103, 4, 99, 6],  # hidden trade, book unchanged
        [104, 2, 100, 3, 103, 4, 99, 6],  # unchanged nontrade
        [104, 2, 100, 3, 103, 4, 99, 6],  # halt
    ])
    times, mid, metadata = merge_events(
        np.array([10, 20, 20, 30, 40, 50]), np.array([1, 2, 1, 5, 2, 7]), books,
    )
    np.testing.assert_array_equal(times, [10, 20, 30])
    np.testing.assert_array_equal(mid, [202, 204, 204])
    assert metadata["unchanged_nontrade_rows_dropped"] == 1
    assert metadata["halt_rows_dropped"] == 1


def test_clock_rejects_misalignment_and_nonchronological_messages():
    with pytest.raises(ValueError, match="row count"):
        merge_events(np.array([10, 20]), np.array([1, 1]), np.array([[102, 1, 100, 1]]))
    with pytest.raises(ValueError, match="chronological"):
        merge_events(np.array([20, 10]), np.array([1, 1]), np.array([[102, 1, 100, 1], [103, 1, 100, 1]]))


def test_clock_marks_dummy_mid_instead_of_counting_a_price_jump():
    _, mid, metadata = merge_events(
        np.array([10, 20]), np.array([1, 1]),
        np.array([[102, 1, 100, 1], [9999999999, 0, 100, 1]]),
    )
    assert mid[1] == np.iinfo(np.int64).min
    assert metadata["invalid_mid_events"] == 1


def test_isotonic_half_life_pools_a_reversal_before_interpolating():
    result = half_life(np.array([0, 1, 2, 4, 8]), np.array([0.4, 0.3, 0.32, 0.1, -0.02]))
    np.testing.assert_allclose(result["fit_bits"], [0.4, 0.31, 0.31, 0.1, 0])
    assert result["half_life_events"] == pytest.approx(2 + 2 * 0.11 / 0.21)
    assert result["status"] == "observed"


def test_half_life_censoring_and_absent_initial_skill_are_distinct():
    censored = half_life(np.array([0, 8]), np.array([0.4, 0.3]))
    absent = half_life(np.array([0, 8]), np.array([-0.1, -0.2]))
    assert censored["status"] == "right_censored"
    assert censored["lower_bound_events"] == 8
    assert censored["half_life_events"] is None
    assert absent["status"] == "no_positive_initial_skill"
    with pytest.raises(ValueError, match="finite"):
        half_life(np.array([0, 8]), np.array([0.1, np.nan]))


def test_h1_switches_on_opposite_sides_of_analytic_threshold():
    tau_c = 20
    critical_rate = -np.log1p(-DELTA_1) / (DELTA_L * (-np.expm1(-1 / tau_c)))
    below = h1_cell("synthetic", critical_rate * 0.99, tau_c, "unit fixture")
    above = h1_cell("synthetic", critical_rate * 1.01, tau_c, "unit fixture")
    assert not below["fast_wins"]
    assert below["fast_to_slow_score_ratio"] < 1
    assert above["fast_wins"]
    assert above["fast_to_slow_score_ratio"] > 1


def test_paired_ratio_has_zero_uncertainty_for_proportional_scores():
    denominator = np.array([1.0, 0.2, 0.8, 0.1])
    ratio, influence = ratio_estimate(0.9 * denominator, denominator)
    assert ratio == pytest.approx(0.9)
    np.testing.assert_allclose(influence, 0, atol=1e-15)
