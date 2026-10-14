"""Integration checks of generated bit/ns artifacts against raw data and definitions.

Run after scripts/run_gate0.py. Without a completed run these checks are skipped;
the synthetic unit tests remain independent of research result artifacts.
"""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lob_latency.events import merge_events
from lob_latency.features import make_features
from lob_latency.io_lobster import load_lobster
from lob_latency.labels import TAU_GRID, delayed_labels, next_move, select_anchors

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/gate0"
TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
pytestmark = pytest.mark.skipif(not (RESULTS / "summary.json").exists(),
                                reason="Full Gate 0 result artifacts are not available yet")


@pytest.fixture(scope="module", params=TICKERS)
def ticker_result(request):
    """Reconstruct split anchors from local data once per ticker."""
    ticker = request.param
    result = json.loads((RESULTS / ticker / "metrics.json").read_text())
    prefix = ROOT / "data/raw/lobster" / f"{ticker}_2012-06-21_34200000_57600000"
    message, book = Path(f"{prefix}_message_10.csv"), Path(f"{prefix}_orderbook_10.csv")
    data = load_lobster(message, book)
    events = merge_events(data)
    direction, end = next_move(events.mid)
    features = make_features(events)
    anchors = select_anchors(events.time_ns, end, np.isfinite(features).all(axis=1))
    for split, ids in anchors.items():
        labels = np.column_stack([delayed_labels(direction, ids, tau) for tau in TAU_GRID])
        anchors[split] = ids[np.all(labels >= 0, axis=1)]
    return result, data, events, end, features, anchors, (message, book)


def test_generated_anchor_boundaries_and_data_provenance(ticker_result):
    result, data, events, end, features, anchors, sources = ticker_result
    assert result["data"]["raw_messages"] == data.raw_rows
    assert result["data"]["merged_events"] == len(events.time_ns)
    assert result["data"]["sub_ns_timestamps_rounded"] == data.sub_ns_timestamps_rounded
    assert np.all(np.diff(events.time_ns) > 0)
    for source in sources:
        assert result["data"]["source_sha256"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
    for split, ids in anchors.items():
        recorded = result["anchors"][split]
        start, stop = np.array(recorded["split_seconds"]) * 1_000_000_000
        assert recorded["count"] == len(ids)
        assert np.all(events.time_ns[ids - 50] >= start)
        for tau in TAU_GRID:
            assert np.all(end[ids + tau] > ids + tau)
            assert np.all(events.time_ns[end[ids + tau]] < stop)
        for direct in result["direct_checks"]:
            tau = np.searchsorted(events.time_ns, events.time_ns[ids] + direct["latency_ns"],
                                   side="right") - ids - 1
            action = ids + tau
            target_end = end[action]
            keep = (target_end >= 0) & (events.time_ns[np.maximum(target_end, 0)] < stop)
            assert direct["kept_anchors"][split] == np.sum(keep)
            assert direct["fraction_dropped"][split] == np.mean(~keep)
    direction = next_move(events.mid)[0]
    train_prior = [np.mean(direction[anchors["train"] + tau]) for tau in TAU_GRID]
    np.testing.assert_allclose(train_prior, result["train_prior_up"], atol=1e-15)
    training = features[anchors["train"]]
    np.testing.assert_allclose(training.mean(axis=0), result["training_feature_mean"], rtol=1e-12)
    draws = np.load(RESULTS / result["ticker"] / "bootstrap_blocks.npz")
    np.testing.assert_array_equal(draws["test_anchor_indices"], anchors["test"])
    np.testing.assert_array_equal(draws["test_time_ns"], events.time_ns[anchors["test"]])
    times = draws["test_time_ns"]
    left, right = draws["left"], draws["right"]
    assert left.shape[0] == 1000
    assert np.all(times[left] + 300_000_000_000 <= times[-1])
    np.testing.assert_array_equal(right, np.searchsorted(times, times[left] + 300_000_000_000))


def test_generated_latency_conversion_and_independent_mixture(ticker_result):
    result, _, events, _, _, anchors, _ = ticker_result
    test = anchors["test"]
    times = events.time_ns
    for j, entry in enumerate(result["latency_distribution"]):
        tau = np.searchsorted(times, times[test] + entry["latency_ns"], side="right") - test - 1
        values, counts = np.unique(tau, return_counts=True)
        pmf = entry["pmf"]
        np.testing.assert_array_equal(values, [int(k) for k in pmf])
        np.testing.assert_allclose(counts / len(test), list(pmf.values()), atol=1e-15)
        assert entry["p_beyond_64"] == np.mean(tau > 64)
        assert entry["p_at_least_one"] == np.mean(tau >= 1)
        np.testing.assert_allclose(np.percentile(tau, [50, 90, 99]),
                                   list(entry["percentiles_events"].values()))
        # np.interp is independent of the runner's barycentric weight implementation.
        for variant, latency_variant in (("retrained", "latency_adjusted"),
                                          ("naive", "naive_latency_adjusted")):
            for model, curve in result[variant].items():
                adjusted = np.average(np.interp(values, TAU_GRID, curve["S_bits"]), weights=counts)
                assert abs(adjusted - result[latency_variant][model]["A_bits"][j]) < 1e-12


def test_generated_model_selection_and_seed_spread(ticker_result):
    result, *_ = ticker_result
    gate = result["gate"]
    assert gate["best_model"] == max(gate["validation_S0_bits"], key=gate["validation_S0_bits"].get)
    curve = result["latency_adjusted"][gate["best_model"]]
    k = result["latency_ns"].index(2_000_000)
    assert gate["delta_bits"] == curve["A_bits"][0] - curve["A_bits"][k]
    assert gate["delta_ci95_bits"] == curve["A_fast_minus_L_ci95_bits"][k]
    low, high = gate["delta_ci95_bits"]
    assert gate["ci_excludes_zero"] == (low > 0 or high < 0)
    for j, spread in enumerate(result["hgb_seed_spread"]):
        assert len(spread["S_bits_by_seed"]) == 5
        assert abs(np.mean(spread["S_bits_by_seed"]) - result["retrained"]["hgb"]["S_bits"][j]) < 1e-12
        np.testing.assert_allclose(spread["S_min_max_bits"],
                                   [min(spread["S_bits_by_seed"]), max(spread["S_bits_by_seed"])])
        for fit in result["fit_metadata"]["hgb"][str(TAU_GRID[j])]:
            candidates = fit["validation_candidate_loss_bits"]
            assert fit["max_iter"] == int(min(candidates, key=candidates.get))
            assert fit["temperature"] > 0
    for name in result["retrained"]:
        assert result["retrained"][name]["S_bits"][0] == result["naive"][name]["S_bits"][0]


def test_completed_gate_verdict_and_source_manifest():
    summary = json.loads((RESULTS / "summary.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert set(summary["completed_tickers"]) == set(TICKERS)
    significant = sum(g["ci_excludes_zero"] for g in summary["primary"].values())
    assert summary["significant_tickers"] == significant
    assert summary["verdict"] == ("PASS" if significant >= 2 else "FAIL")
    for relative, expected in manifest["source_sha256"].items():
        source = (ROOT / relative).read_bytes()
        if relative == "src/lob_latency/io_lobster.py":
            # Gate 0 records the historical CSV loader. Parquet support changes its
            # source bytes; verify the recorded version rather than rewriting results.
            # Current CSV/Parquet behavior is exercised independently in test_itch.py.
            source = subprocess.run(
                ["git", "show", "6a2c3aa:src/lob_latency/io_lobster.py"],
                cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(source).hexdigest() == expected
