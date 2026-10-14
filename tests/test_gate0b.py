"""Day isolation, deterministic asset picks, and offline multi-day pipeline checks."""

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import polars as pl
import pytest
from threadpoolctl import threadpool_limits

from lob_latency.events import Events
from lob_latency.experiment import fit_grid
from lob_latency.io_lobster import LobsterData, load_lobster
from lob_latency.labels import TAU_GRID
from lob_latency.latency import LATENCY_NS
from lob_latency.metrics import entropy_bits, logit_losses_bits
from lob_latency.multiday import (ALL_DAYS, SESSION_SECONDS, TEST_DAYS, TRAIN_DAYS,
                                  VALIDATION_DAY, half_life, half_life_summary,
                                  prepare_day, split_arrays)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_assets import select_assets
from symbol_stats import symbol_day_statistics


def synthetic_day(n=400):
    """Alternating price levels and finite windows spanning seconds in a session."""
    book = np.tile([10_100, 10, 10_000, 20], (n, 10)).astype(np.int64)
    shift = ((np.arange(n) // 3) % 2) * 100
    book[:, 0::2] += shift[:, None]
    book[:, 3] += np.arange(n) % 5
    times = np.linspace(35_100 * 10**9, 57_000 * 10**9 - 1, n, dtype=np.int64)
    return LobsterData(times, np.ones(n, dtype=int), np.ones(n, dtype=int), book, n, 0)


def stats_table():
    """Hand-built rates in events/second with single-day quality failures."""
    rows = []
    for symbol, rate, spread in (("LF", 100, 1), ("LM", 60, 1.5), ("LS", 20, 1),
                                  ("SF", 80, 4), ("SS", 40, 3), ("MF", 70, 2),
                                  ("MS", 30, 2), ("MISS", 999, 1), ("CROSS", 999, 1),
                                  ("UNKNOWN", 999, 4), ("LOCK", 999, 1), ("NOQUAL", 999, 1)):
        for day in TRAIN_DAYS:
            if symbol == "MISS" and day == TRAIN_DAYS[-1]:
                continue
            rows.append(dict(date=day, symbol=symbol, event_rate_hz=rate,
                             mean_spread_ticks=float(spread), crossed_locked_session_share=(
                                 0.02 if symbol in ("CROSS", "LOCK") and day == TRAIN_DAYS[-1] else 0.01),
                             unknown_order_refs=(None if symbol == "NOQUAL" else
                                                 1 if symbol == "UNKNOWN" and day == TRAIN_DAYS[-1] else 0)))
    # Held-out rates and integrity must never influence selection.
    rows.append(dict(date=TEST_DAYS[0], symbol="LF", event_rate_hz=0,
                     mean_spread_ticks=20.0, crossed_locked_session_share=1.0, unknown_order_refs=1000))
    return pl.DataFrame(rows)


def test_asset_selection_exclusions_and_training_only():
    selected = select_assets(stats_table())
    assert selected["symbols"] == ["LF", "LM", "SF", "MF"]
    assert selected["rule"]["large_tick_median_event_rate_hz"] == 60
    assert set(selected["excluded"]) == {"MISS", "CROSS", "UNKNOWN", "LOCK", "NOQUAL"}
    assert selected["picks"][0]["averages"]["event_rate_hz"] == 100
    assert all(len(p["training_rows"]) == 3 for p in selected["picks"])
    assert select_assets(stats_table().reverse()) == selected


def test_selection_optional_medium_ties_and_day_means():
    table = stats_table().filter(~pl.col("symbol").is_in(["MF", "MS"]))
    assert select_assets(table)["symbols"] == ["LF", "LM", "SF"]
    # Equal large rates still give distinct fast/slow picks with lexical ties.
    table = table.with_columns(pl.when(pl.col("symbol").is_in(["LF", "LM", "LS"]))
                               .then(pl.lit(60)).otherwise(pl.col("event_rate_hz")).alias("event_rate_hz"))
    assert select_assets(table)["symbols"][:2] == ["LF", "LM"]
    averaging = stats_table().with_columns(pl.when((pl.col("symbol") == "LF") & (pl.col("date") == TRAIN_DAYS[0]))
                                          .then(pl.lit(130)).otherwise(pl.col("event_rate_hz")).alias("event_rate_hz"))
    assert select_assets(averaging)["picks"][0]["averages"]["event_rate_hz"] == 110


def test_selection_rejects_missing_integrity_schema():
    with pytest.raises(ValueError, match="rerun scripts/symbol_stats.py"):
        select_assets(stats_table().drop("unknown_order_refs"))
    duplicate = pl.concat([stats_table(), stats_table().filter((pl.col("symbol") == "LF") & (pl.col("date") == TRAIN_DAYS[0]))])
    assert "LF" in select_assets(duplicate)["excluded"]


def test_day_isolation_windows_targets_counts_and_trim():
    raw = synthetic_day()
    # Include pre/post-trim books so neither can supply a feature or a label.
    extended = LobsterData(np.r_[35_099 * 10**9, raw.time_ns, 57_001 * 10**9],
                           np.ones(402, int), np.ones(402, int),
                           np.vstack([raw.book[0] * 10, raw.book, raw.book[-1] * 10]), 402, 0)
    day = prepare_day(extended)
    np.testing.assert_array_equal(day.time_ns, raw.time_ns)
    assert day.anchors[0] >= 50
    for tau in TAU_GRID:
        assert np.all(day.next_end[day.anchors + tau] > day.anchors + tau)
        assert np.all(day.next_end[day.anchors + tau] < len(day.time_ns))
    assert np.all(day.anchors + 64 < len(day.time_ns))
    # The final event has no next label and cannot borrow the next day's price.
    assert day.next_dir[-1] == day.next_end[-1] == -1
    assert np.isnan(__import__('lob_latency.features', fromlist=['make_features']).make_features(
        Events(raw.time_ns, raw.book, np.ones(400), np.zeros(400, bool), np.zeros(400)))[:49]).any()
    other = prepare_day(synthetic_day())
    other.time_ns[:] += day.time_ns[-1] + 1  # Even a later adjacent timeline is separate.
    other.features[:] *= 100
    np.testing.assert_array_equal(prepare_day(extended).features, day.features)
    from lob_latency.latency import event_delays
    assert event_delays(day.time_ns, np.array([len(day.time_ns) - 1]), 10**15)[0] == 0
    labels, mask = day.direct_target(10**15)
    assert not mask.any() and np.all(labels == -1)


def test_pooled_scaling_uses_training_anchors_only():
    prepared = prepare_day(synthetic_day(), 100)
    days = {date: replace(prepared, features=prepared.features.copy()) for date in ALL_DAYS}
    days[TRAIN_DAYS[1]].features += 1
    days[TRAIN_DAYS[2]].features -= 3
    _, ys, mean, scale = split_arrays(days)
    train = np.concatenate([days[date].features for date in TRAIN_DAYS])
    np.testing.assert_allclose(mean, train.mean(axis=0))
    expected = train.std(axis=0)
    expected[expected == 0] = 1
    np.testing.assert_allclose(scale, expected)
    for date in (VALIDATION_DAY, *TEST_DAYS):
        days[date].features *= 1000
    _, _, new_mean, new_scale = split_arrays(days)
    np.testing.assert_array_equal(new_mean, mean)
    np.testing.assert_array_equal(new_scale, scale)
    assert ys["train"].shape == (300, len(TAU_GRID))


def test_half_life_isotonic_crossing_and_unidentified():
    curve = half_life(np.array([.4, .3, .32, .1, -.02]), np.array([0, 1, 2, 4, 8]))
    np.testing.assert_allclose(curve["fit_bits"], [.4, .31, .31, .1, 0])
    assert curve["half_life_events"] == pytest.approx(3.047619047619)
    assert half_life(np.zeros(8))["status"] == "no_positive_initial_skill"
    censored = half_life(np.ones(8))
    assert censored["status"] == "right_censored" and censored["lower_bound_events"] == 64
    with pytest.raises(ValueError):
        half_life(np.full(8, np.nan))


def test_half_life_bootstrap_preserves_censoring_and_fractional_clock():
    day = prepare_day(synthetic_day())
    # A linear skill curve crosses at half an event; convert timestamps fractionally.
    point = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    boot = np.vstack([point, np.ones(8), np.zeros(8)])
    result = half_life_summary(point, boot, day)
    assert result["half_life_events"] == .5
    assert result["ci95_events"] == [.5, None]
    assert result["ci95_unbounded"] == [False, True]
    assert result["bootstrap_right_censored_fraction"] == pytest.approx(1 / 3)
    assert result["bootstrap_no_positive_initial_fraction"] == pytest.approx(1 / 3)
    elapsed = (day.time_ns[day.anchors + 1] - day.time_ns[day.anchors]) / 2e9
    assert result["wall_clock_seconds"]["median"] == pytest.approx(np.median(elapsed))
    assert half_life_summary(point, np.zeros((2, 8)), day)["ci95_events"] is None


def test_symbol_statistics_quality_seconds_and_unknown_refs():
    raw = synthetic_day()
    raw.book[100:102, 0] = raw.book[100:102, 2]
    stats = symbol_day_statistics(raw, "TEST", TRAIN_DAYS[0], integrity={"unknown_order_refs": 2})
    duration = (raw.time_ns[102] - raw.time_ns[100]) / 10**9
    assert stats["crossed_locked_session_share"] == pytest.approx(duration / (57_000 - 35_100))
    assert stats["event_rate_hz"] == 400 / (57_000 - 35_100)
    assert stats["unknown_order_refs"] == 2
    assert symbol_day_statistics(raw, "TEST", TRAIN_DAYS[0])["unknown_order_refs"] is None


def test_grid_reuses_fits_for_both_test_days(monkeypatch):
    import lob_latency.experiment as experiment
    rng = np.random.default_rng(0)
    xs = {name: rng.normal(size=(90, 10)) for name in ("train", "validation", *TEST_DAYS)}
    ys = {name: rng.integers(0, 2, size=(90, 8)) for name in xs}
    original = experiment.fit_model
    calls = []
    def tracked(name, xt, yt, xv, yv, seed):
        assert xt is xs["train"] and xv is xs["validation"]
        calls.append((name, seed))
        return original(name, xt, yt, xv, yv, seed)
    monkeypatch.setattr(experiment, "fit_model", tracked)
    with threadpool_limits(limits=1):
        result = fit_grid(xs, ys, TEST_DAYS, hgb_seeds=(0,))
    assert len(calls) == len(TAU_GRID) * 3
    for day in TEST_DAYS:
        for model in ("qi_logit", "ofi_logit", "hgb"):
            scored = result["evaluation"][day]
            np.testing.assert_array_equal(scored["losses"][model][:, 0], scored["naive_losses"][model][:, 0])


def build_fake_root(root):
    """Convert all five local CSV pairs; repeat Parquet streams on synthetic dates.

    Hard links save disk space without changing contents. Fitting uses an evenly
    spaced reduced anchor population, while the event clock retains all rows.
    """
    symbols = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
    rows, provenance = [], {}
    for symbol in symbols:
        prefix = ROOT / "data/raw/lobster" / f"{symbol}_2012-06-21_34200000_57600000"
        message, book = Path(f"{prefix}_message_10.csv"), Path(f"{prefix}_orderbook_10.csv")
        first = root / ALL_DAYS[0]
        first.mkdir(parents=True, exist_ok=True)
        pl.read_csv(message, has_header=False, schema_overrides={"column_0": pl.String, "column_2": pl.UInt64}).write_parquet(first / f"{symbol}_message_10.parquet")
        pl.read_csv(book, has_header=False).write_parquet(first / f"{symbol}_orderbook_10.parquet")
        data = load_lobster(message, book)
        statistics = symbol_day_statistics(data, symbol, ALL_DAYS[0], integrity={"unknown_order_refs": 0})
        for date in ALL_DAYS:
            folder = root / date
            folder.mkdir(parents=True, exist_ok=True)
            if date != ALL_DAYS[0]:
                for kind in ("message", "orderbook"):
                    os.link(first / f"{symbol}_{kind}_10.parquet", folder / f"{symbol}_{kind}_10.parquet")
            rows.append({**statistics, "date": date})
        provenance[symbol] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (message, book)}
    for date in ALL_DAYS:
        (root / date / "manifest.json").write_text(json.dumps({
            "date": date, "symbols": symbols, "complete_input": True, "synthetic_fixture": True,
            "source_sha256": provenance, "source_day": "2012-06-21",
            "assumption": "Historical streams repeated under synthetic dates; zero unknown refs is a fixture convention"}))
    table = root / "symbol_stats.csv"
    pl.DataFrame(rows).write_csv(table)
    return table


def run_logged(command, output):
    """Record the exact executed command, exit code and output as local evidence."""
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "MPLCONFIGDIR": ".cache/mpl", "OMP_NUM_THREADS": "1"})
    output.write_text("Command: " + __import__('shlex').join(map(str, command)) + "\n" + result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_fake_multiday_end_to_end(tmp_path):
    root = tmp_path / "fake_lobster"
    stats = build_fake_root(root)
    output = Path(os.environ.get("GATE0B_TEST_OUTPUT", str(tmp_path / "results"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "selected_assets.json"
    run_logged([str(ROOT / ".venv/bin/python"), "scripts/select_assets.py", "--stats", str(stats),
                "--output", str(assets)], output / "selection.log")
    report = output / "report.md"
    run_logged([str(ROOT / ".venv/bin/python"), "scripts/run_gate0b.py", "--data-root", str(root),
                "--assets", str(assets), "--output", str(output), "--report", str(report),
                "--fake-data", "--resamples", "20", "--hgb-seeds", "0", "--max-anchors-per-day", "1200"], output / "run.log")
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert summary["run_kind"] == "synthetic_fixture"
    assert manifest["configuration"]["out_of_time_refit"] is False
    assert manifest["configuration"]["hgb_iterations"] == [50, 100, 200, 400]
    assert "## Assumptions" in report.read_text()
    assert "within-day" in report.read_text().lower()
    assert "scripts/symbol_stats.py" in summary["full_run_command"]
    selected = json.loads(assets.read_text())
    assert set(selected["symbols"]) <= {"AAPL", "AMZN", "GOOG", "INTC", "MSFT"}
    for symbol in selected["symbols"]:
        metric = json.loads((output / symbol / "metrics.json").read_text())
        for date in TEST_DAYS:
            day = metric["tests"][date]
            assert day["bootstrap"]["block_seconds"] == 1800
            assert day["sensitivity_5min"]["bootstrap"]["block_seconds"] == 300
            assert day["gate"]["best_model"] == max(day["gate"]["validation_S0_bits"], key=day["gate"]["validation_S0_bits"].get)
            assert day["gate"]["delta_ci95_bits"] == day["latency_adjusted"][day["gate"]["best_model"]]["A_fast_minus_L_ci95_bits"][8]
            assert metric["data"][date]["count"] == 1200
            assert day["gate"]["validation_S0_bits"] == metric["tests"][TEST_DAYS[0]]["gate"]["validation_S0_bits"]
            curve = day["retrained"][day["gate"]["best_model"]]["S_bits"]
            for k, entry in enumerate(day["latency_distribution"]):
                values = np.array([int(t) for t in entry["pmf"]])
                probs = np.array(list(entry["pmf"].values()))
                expected = np.sum(probs * np.interp(values, TAU_GRID, curve))
                assert day["latency_adjusted"][day["gate"]["best_model"]]["A_bits"][k] == pytest.approx(expected, abs=1e-12)
            for seconds in (1800, 300):
                draw = np.load(output / symbol / f"bootstrap_{date}_{seconds}s.npz")
                assert draw["left"].shape[0] == 20
                times = draw["test_time_ns"]
                assert np.all(times[draw["left"]] + seconds * 10**9 <= times[-1])
                np.testing.assert_array_equal(draw["right"], np.searchsorted(times, times[draw["left"]] + seconds * 10**9))
            assert (output / "figures" / f"{symbol}_{date}_skill.png").stat().st_size > 0
        assert len(metric["tests"][TEST_DAYS[0]]["direct_checks"]) == 2
        assert "direct_checks" not in metric["tests"][TEST_DAYS[1]]
        assert "burst_split" in metric["tests"][TEST_DAYS[0]]
    for date in TEST_DAYS:
        gate = summary["test_days"][date]
        count = sum(g["ci_excludes_zero"] for g in gate["primary"].values())
        assert gate["significant_symbols"] == count
        assert gate["verdict"] == ("PASS" if count >= 2 else "FAIL")
    write = dict(fixture="all five local CSV samples repeated on synthetic dates", seed=0,
                 conversion="Polars CSV to exact-string-second Parquet, shared by hard links",
                 sources=manifest["inputs"][ALL_DAYS[0]]["manifest"]["source_sha256"],
                 selected_symbols=selected["symbols"], test_command="GATE0B_TEST_OUTPUT=results/gate0b/fake MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_gate0b.py::test_fake_multiday_end_to_end",
                 resamples=20, hgb_seeds=[0], max_anchors_per_day=1200)
    (output / "fixture_evidence.json").write_text(json.dumps(write, indent=2) + "\n")
