"""Hand-calculated checks for event/ns timing, causal features, and bit metrics."""

import numpy as np
import pytest

from lob_latency.events import DUMMY_PRICE, Events, merge_events
from lob_latency.features import make_features, order_flow_imbalance, standardize
from lob_latency.io_lobster import LobsterData, load_lobster
from lob_latency.labels import action_mask, delayed_labels, next_move, select_anchors
from lob_latency.latency import event_delays, interpolation_weights
from lob_latency.metrics import (block_means, entropy_bits, interval, masked_skill,
                                 moving_time_blocks, skill_bits)
from lob_latency.models import fit_model, temperature_scale


def books(n):
    """Ten-level synthetic books with prices in dollars * 10000."""
    book = np.tile([10_100, 10, 10_000, 20], (n, 10))
    return book.astype(np.int64)


def test_event_merge_and_last_execution_sign():
    book = books(6)
    book[:, 1] = np.arange(6) + 10
    data = LobsterData(np.array([1, 1, 2, 3, 3, 3]), np.array([1, 4, 1, 5, 4, 1]),
                       np.array([1, -1, 1, -1, 1, 1]), book, 6, 0)
    events = merge_events(data)
    np.testing.assert_array_equal(events.time_ns, [1, 2, 3])
    np.testing.assert_array_equal(events.message_count, [2, 1, 3])
    np.testing.assert_array_equal(events.book, book[[1, 2, 5]])
    np.testing.assert_array_equal(events.execution, [True, False, True])
    np.testing.assert_array_equal(events.trade_sign, [1, 0, -1])


def test_loading_exact_nanoseconds_and_halt_alignment(tmp_path):
    message = tmp_path / "message.csv"
    message.write_text("35100.000000001,1,1,10,10100,1\n"
                       "35100.000000002,7,0,0,-1,-1\n"
                       "35100.000000003,6,2,10,10000,-1\n")
    book = books(3)
    book[:, 1] = [10, 99, 30]
    book_path = tmp_path / "book.csv"
    np.savetxt(book_path, book, delimiter=",", fmt="%d")
    data = load_lobster(message, book_path)
    np.testing.assert_array_equal(data.time_ns, [35_100_000_000_001, 35_100_000_000_003])
    np.testing.assert_array_equal(data.event_type, [1, 6])
    np.testing.assert_array_equal(data.book[:, 1], [10, 30])
    assert data.raw_rows == 3 and data.dropped_halts == 1


def test_loading_rounds_serialization_tails_without_float_times(tmp_path):
    message = tmp_path / "message.csv"
    message.write_text("36754.716797047004,1,1,10,10100,1\n"
                       "36754.716797047996,1,2,10,10100,1\n")
    book_path = tmp_path / "book.csv"
    np.savetxt(book_path, books(2), delimiter=",", fmt="%d")
    data = load_lobster(message, book_path)
    np.testing.assert_array_equal(data.time_ns, [36_754_716_797_047, 36_754_716_797_048])
    assert data.sub_ns_timestamps_rounded == 2


def test_labels_at_action_and_invariance():
    mid = np.array([10., 10., 11., 11., 11., 9., 9., 12., 12.])
    direction, end = next_move(mid)
    np.testing.assert_array_equal(direction, [1, 1, 0, 0, 0, 1, 1, -1, -1])
    np.testing.assert_array_equal(end, [2, 2, 5, 5, 5, 7, 7, -1, -1])
    assert delayed_labels(direction, np.array([1]), 2)[0] == 0
    # Action is index 3; its first changed mid is index 5.
    before = mid.copy()
    before[:3] = [100, 1, 42]
    after = mid.copy()
    after[6:] = [100, 1, 42]
    assert delayed_labels(next_move(before)[0], np.array([1]), 2)[0] == 0
    assert delayed_labels(next_move(after)[0], np.array([1]), 2)[0] == 0


def test_missing_mids_are_neither_reference_nor_terminal():
    direction, end = next_move(np.array([10., np.nan, 10., 11., np.nan]))
    np.testing.assert_array_equal(direction, [1, -1, 1, -1, -1])
    np.testing.assert_array_equal(end, [3, -1, 3, -1, -1])
    book = books(2)
    book[1, 0] = DUMMY_PRICE
    events = Events(np.array([0, 1]), book, np.ones(2), np.zeros(2, bool), np.zeros(2))
    assert np.isnan(events.mid[1])


@pytest.mark.parametrize("bid,ask,qb,qa,expected", [
    (10_100, 10_200, 30, 15, 40),  # Both prices rise: new bid + old ask.
    (9_900, 10_000, 30, 15, -35),  # Both fall: -old bid - new ask.
    (10_000, 10_100, 30, 15, 5),  # Equal prices: delta bid - delta ask.
    (10_100, 10_000, 30, 15, 15),  # Bid rises, ask falls: new bid - new ask.
    (9_900, 10_200, 30, 15, -10),  # Bid falls, ask rises: -old bid + old ask.
])
def test_ofi_indicator_cases(bid, ask, qb, qa, expected):
    book = books(2)
    book[1, :4] = [ask, qa, bid, qb]
    np.testing.assert_array_equal(order_flow_imbalance(book), [0, expected])


def test_features_are_causal_and_trade_sign_expires():
    book = books(100)
    book[:, 3] += np.arange(100)
    execution = np.zeros(100, dtype=bool)
    execution[52] = True
    sign = np.zeros(100)
    sign[52] = 1
    events = Events(np.arange(100) * 1000, book, np.ones(100), execution, sign)
    original = make_features(events)
    changed_book = book.copy()
    changed_book[61:] *= 10
    changed_execution = execution.copy()
    changed_execution[61:] = True
    changed_sign = sign.copy()
    changed_sign[61:] = -1
    changed = Events(events.time_ns.copy(), changed_book, events.message_count,
                     changed_execution, changed_sign)
    changed.time_ns[61:] += 500
    np.testing.assert_array_equal(make_features(changed)[:61], original[:61])
    assert original[61, -1] == 1 and original[62, -1] == 0
    assert original[60, 6] == 50  # Fifty bid-size increments of one share.
    assert original[60, 7] == 1


def test_standardization_uses_only_training_anchors():
    features = np.array([[1., 4.], [3., 4.], [1000., 50.]])
    standardized, mean, scale = standardize(features, np.array([0, 1]))
    np.testing.assert_array_equal(mean, [2., 4.])
    np.testing.assert_array_equal(scale, [1., 1.])
    np.testing.assert_array_equal(standardized[:2, 0], [-1, 1])


def test_delay_ties_and_inclusive_right_endpoint():
    times = np.array([0, 0, 1, 2, 2, 3, 5], dtype=np.int64)
    np.testing.assert_array_equal(event_delays(times, np.array([0, 1, 3]), 2), [3, 3, 1])
    np.testing.assert_array_equal(event_delays(times, np.array([0, 3]), 0), [0, 0])


def test_interpolation_and_tail_hold():
    grid = np.array([0, 1, 2, 4, 64])
    weight = interpolation_weights(np.array([0, 1, 3, 64, 100]), grid)
    np.testing.assert_allclose(weight.sum(axis=1), 1)
    np.testing.assert_allclose(weight @ np.array([0., 1., 2., 4., 64.]), [0, 1, 3, 64, 64])


def test_skill_prior_and_perfect_predictor():
    labels = np.array([1, 0, 0, 0])
    prior = labels.mean()
    assert abs(skill_bits(labels, np.full(4, prior), prior)) < 1e-12
    assert skill_bits(labels, labels.astype(float), prior) == entropy_bits(prior)


def test_anchor_windows_and_label_ends_stay_in_splits():
    times = np.arange(20, dtype=np.int64) * 1_000_000_000
    end = np.arange(20) + 2
    end[-2:] = -1
    anchors = select_anchors(times, end, np.ones(20, bool), lookback=2, max_tau=2,
                             splits={"train": (0, 10), "validation": (10, 20)})
    np.testing.assert_array_equal(anchors["train"], [2, 3, 4, 5])
    assert np.all(times[end[anchors["train"] + 2]] < 10_000_000_000)
    assert np.all(times[anchors["validation"] - 2] >= 10_000_000_000)


def test_wall_clock_label_that_crosses_boundary_is_dropped():
    times = np.array([48_597, 48_598, 48_599, 48_600, 48_601]) * 1_000_000_000
    _, end = next_move(np.array([10., 10., 11., 12., 13.]))
    np.testing.assert_array_equal(action_mask(times, end, np.array([0, 1]),
                                            np.array([0, 1]), "train"), [True, False])


def test_paired_block_bootstrap_and_subgroup_denominators():
    times = np.arange(1001, dtype=np.int64) * 1_000_000_000
    left, right = moving_time_blocks(times, resamples=100, seed=0)
    left_again, right_again = moving_time_blocks(times, resamples=100, seed=0)
    np.testing.assert_array_equal(left, left_again)
    np.testing.assert_array_equal(right, right_again)
    assert np.all(right - left == 300)
    means = block_means(np.column_stack([np.arange(1001), np.arange(1001) + 5]), left, right)
    np.testing.assert_allclose(means[:, 1] - means[:, 0], 5)
    assert interval(means[:, 1] - means[:, 0]) == pytest.approx([5, 5])
    mask = np.arange(1001) % 2 == 0
    point, boot = masked_skill(np.full((1001, 1), 0.25), mask, 1., left, right)
    np.testing.assert_allclose(point, [0.75])
    np.testing.assert_allclose(boot, 0.75)


def test_temperature_reduces_validation_loss():
    from lob_latency.metrics import logit_losses_bits
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    logits = np.array([10, 10, -10, -10, -10, 10, 10, -10], dtype=float)
    temperature = temperature_scale(logits, labels)
    assert temperature > 1
    assert logit_losses_bits(labels, logits / temperature).mean() < logit_losses_bits(labels, logits).mean()


def test_hgb_selection_retains_chosen_candidate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(350, 10))
    y = (x[:, 0] + rng.normal(size=350) > 0).astype(int)
    model = fit_model("hgb", x[:200], y[:200], x[200:], y[200:])
    chosen = min(model.validation_candidates, key=model.validation_candidates.get)
    assert model.max_iter == int(chosen)
    assert model.estimator.n_iter_ == model.max_iter
    assert np.isfinite(model.logits(x[200:])).all()
