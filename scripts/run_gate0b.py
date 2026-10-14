#!/usr/bin/env python3
"""Gate 0b whole-day training and frozen held-out scoring (bits/events/ns)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import polars
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from run_gate0 import REFERENCE_HARDWARE, format_ci, plot_ticker, write_json
from lob_latency.experiment import MODEL_NAMES, curve_statistics, fit_grid, latency_statistics
from lob_latency.features import FEATURE_NAMES
from lob_latency.io_lobster import load_lobster
from lob_latency.labels import TAU_GRID
from lob_latency.latency import LATENCY_NAMES, LATENCY_NS, delay_statistics, interpolation_weights
from lob_latency.metrics import (block_means, classification_metrics, entropy_bits, interval,
                                 logit_losses_bits, masked_skill, moving_time_blocks)
from lob_latency.models import HGB_ITERATIONS, SEEDS, fit_model
from lob_latency.multiday import (ALL_DAYS, SESSION_SECONDS, TEST_DAYS, TRAIN_DAYS,
                                  VALIDATION_DAY, PreparedDay, half_life_summary,
                                  prepare_day, split_arrays)

FULL_COMMAND = (
    "MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/symbol_stats.py "
    "--input data/raw/itch_lobster --out results/itch && "
    ".venv/bin/python scripts/select_assets.py && "
    "MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/run_gate0b.py "
    "--data-root data/raw/itch_lobster --assets results/itch/selected_assets.json "
    "--output results/gate0b --report docs/gate0b_report.md"
)

ASSUMPTIONS = [
    "Input timestamps are exchange-local seconds after midnight, with exact decimal parsing and equal-timestamp merging as in Gate 0.",
    "Each day is trimmed before features, labels and counts; the preceding-book OFI guard and common valid labels apply. Complete maximum-latency windows are required through the last observed event.",
    "Training days are pooled by anchor count for feature statistics, class priors and fitting; validation alone selects iterations, temperatures and the best model.",
    "Out-of-time scores reuse the same fitted models, temperatures, training priors and feature scales without refitting or reselection.",
    "Asset selection uses equal-day averaged event rates and event-mean spreads; per-day quality exclusions use trimmed-session time fractions. Missing integrity evidence is excluded. Ties use ascending symbol order, and picks must be distinct.",
    "The symbol-statistics producer was extended to fix missing event-rate and integrity fields required by selection; unknown references come from reconstruction manifests or integrity JSON, never assumed zero.",
    "HGB metrics average individual calibrated seed losses and classification metrics, rather than probability-ensemble scores. Naive scoring freezes the zero-delay model and temperature.",
    "Bootstrap blocks start uniformly at eligible anchors, retain variable event counts and use the same draws for all losses, distributions, differences and diagnostics within each day. Fits and training priors stay fixed.",
    "Within-day blocks understate day-to-day variation; these intervals do not include training, calibration or asset-selection uncertainty.",
    "Latency mixtures interpolate event-delay skills and hold the last grid value beyond the grid; burst dependence can violate this approximation. Direct checks use matching surviving anchors.",
    "Half-life uses equal-weight non-increasing isotonic skill clipped at zero as a surrogate, with first-crossing interpolation. Censored bootstrap crossings retain unbounded endpoints; crossing intervals condition on positive initial skill and report unidentified mass.",
    "Fractional half-life event indices interpolate adjacent timestamps; median and IQR describe elapsed times across the common test anchors, rather than a confidence interval for clock conversion.",
    "Latencies are assumed scenarios; inference timing, quantization, FPGA behavior and profitability are unverified. Tables name the planned reference hardware, whose timing is unmeasured.",
    "The prescribed gate accepts an interval excluding zero in either direction, without an added multiple-comparison correction; a negative Delta is not evidence of a latency penalty.",
    "Synthetic dates repeat historical sample streams and reduced fixture settings; their verdicts exercise code and provide no evidence of modern-market or across-day generalization.",
    "The later full-run command assumes complete converter output under data/raw/itch_lobster and regeneration of statistics before selection. Real-day results remain unverified until that command runs.",
]


def sha256(path: Path) -> str:
    """Streaming input provenance without copying large Parquet files into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs(root: Path, symbols: list[str], fake_data: bool) -> dict:
    """Preflight every day and pair before fitting; require completed conversion."""
    manifests = {}
    for day in ALL_DAYS:
        path = root / day / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("date") != day or manifest.get("complete_input") is not True:
            raise ValueError(f"{path}: date mismatch or incomplete input")
        if manifest.get("synthetic_fixture") and not fake_data:
            raise ValueError("Synthetic input requires --fake-data")
        for symbol in symbols:
            if symbol not in manifest.get("symbols", []):
                raise ValueError(f"{path}: missing {symbol}")
            for kind in ("message", "orderbook"):
                pair = root / day / f"{symbol}_{kind}_10.parquet"
                if not pair.is_file():
                    raise ValueError(f"Missing input {pair}")
        manifests[day] = {"path": str(path.resolve()), "sha256": sha256(path), "manifest": manifest}
    return manifests


def summarize_bootstrap(evaluated: dict, entropy: np.ndarray, validation: dict,
                        day: PreparedDay, left: np.ndarray, right: np.ndarray) -> dict:
    """Jointly resample event losses and latency PMFs on common day-local anchors."""
    delays = np.column_stack([day.delays(int(latency)) for latency in LATENCY_NS])
    weights = [interpolation_weights(delays[:, k], TAU_GRID) for k in range(len(LATENCY_NS))]
    weight_point = np.array([w.mean(axis=0) for w in weights])
    weight_boot = np.stack([block_means(w, left, right) for w in weights], axis=1)
    result = {key: {} for key in ("retrained", "naive", "latency_adjusted", "naive_latency_adjusted", "half_life")}
    for name in MODEL_NAMES:
        points, boots = [], []
        for loss_key, score_key, curve_key, latency_key in (
            ("losses", "scores", "retrained", "latency_adjusted"),
            ("naive_losses", "naive_scores", "naive", "naive_latency_adjusted"),
        ):
            point = entropy - evaluated[loss_key][name].mean(axis=0)
            boot = entropy - block_means(evaluated[loss_key][name], left, right)
            points.append(point)
            boots.append(boot)
            result[curve_key][name] = {**curve_statistics(point, boot), **evaluated[score_key][name]}
            a_point = weight_point @ point
            a_boot = np.einsum("blt,bt->bl", weight_boot, boot)
            result[latency_key][name] = latency_statistics(a_point, a_boot)
        result["retrained"][name].update(
            validation_S_bits=validation[name],
            retrained_minus_naive_bits=(points[0] - points[1]).tolist(),
            retrained_minus_naive_ci95_bits=[interval(boots[0][:, j] - boots[1][:, j])
                                            for j in range(len(TAU_GRID))])
        result["half_life"][name] = half_life_summary(points[0], boots[0], day)
    p_boot = block_means((delays >= 1).astype(float), left, right)
    mean_boot = block_means(delays, left, right)
    tail_boot = block_means((delays > TAU_GRID[-1]).astype(float), left, right)
    result["latency_distribution"] = []
    for k, (latency, label) in enumerate(zip(LATENCY_NS, LATENCY_NAMES)):
        result["latency_distribution"].append({
            **delay_statistics(delays[:, k]), "latency_ns": int(latency), "label": label,
            "p_at_least_one_ci95": interval(p_boot[:, k]), "mean_events_ci95": interval(mean_boot[:, k]),
            "p_beyond_64_ci95": interval(tail_boot[:, k])})
    best = max(MODEL_NAMES, key=lambda name: validation[name][0])
    k = LATENCY_NS.tolist().index(2_000_000)
    curve = result["latency_adjusted"][best]
    bounds = curve["A_fast_minus_L_ci95_bits"][k]
    result["gate"] = {"best_model": best, "validation_S0_bits": {name: validation[name][0] for name in MODEL_NAMES},
                      "delta_bits": curve["A_fast_minus_L_bits"][k], "delta_ci95_bits": bounds,
                      "ci_excludes_zero": bool(bounds[0] > 0 or bounds[1] < 0)}
    return result


def direct_checks(days: dict, xs: dict, evaluated: dict, entropy: np.ndarray,
                  scored: dict, left: np.ndarray, right: np.ndarray) -> list:
    """Retrain OFI only on train/validation realized labels; score December test."""
    result = []
    test_day = TEST_DAYS[0]
    for latency in (10_000, 1_000_000):
        targets = {day: days[day].direct_target(latency) for day in (*TRAIN_DAYS, VALIDATION_DAY, test_day)}
        train_mask = np.concatenate([targets[day][1] for day in TRAIN_DAYS])
        train_y = np.concatenate([targets[day][0][targets[day][1]] for day in TRAIN_DAYS])
        val_y, val_mask = targets[VALIDATION_DAY]
        test_y, mask = targets[test_day]
        model = fit_model("ofi_logit", xs["train"][train_mask], train_y,
                          xs["validation"][val_mask], val_y[val_mask])
        loss = np.zeros(len(mask))
        logits = model.logits(xs[test_day][mask])
        loss[mask] = logit_losses_bits(test_y[mask], logits)
        direct_point, direct_boot = masked_skill(loss, mask, entropy_bits(model.prior), left, right)
        grid_point, grid_boot = masked_skill(evaluated["losses"]["ofi_logit"], mask, entropy, left, right)
        weights = interpolation_weights(days[test_day].delays(latency), TAU_GRID)
        paired_weights = weights[mask].mean(axis=0)
        numerator = block_means(np.where(mask[:, None], weights, 0), left, right)
        denominator = block_means(mask.astype(float), left, right)
        paired_boot = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        paired_a = float(paired_weights @ grid_point)
        paired_a_boot = np.einsum("bt,bt->b", paired_boot, grid_boot)
        k = LATENCY_NS.tolist().index(latency)
        curve = scored["latency_adjusted"]["ofi_logit"]
        result.append({
            "latency_ns": latency, "metadata": model.metadata(),
            "kept_anchors": {day: int(mask.sum()) for day, (_, mask) in targets.items()},
            "fraction_dropped": {day: float(np.mean(~mask)) for day, (_, mask) in targets.items()},
            "direct_S_bits": float(direct_point[0]), "direct_S_ci95_bits": interval(direct_boot[:, 0]),
            "direct_accuracy_mcc": classification_metrics(test_y[mask], logits),
            "A_full_bits": curve["A_bits"][k], "A_full_ci95_bits": curve["A_ci95_bits"][k],
            "A_same_anchors_bits": paired_a, "A_same_anchors_ci95_bits": interval(paired_a_boot),
            "direct_minus_A_same_anchors_bits": float(direct_point[0] - paired_a),
            "direct_minus_A_same_anchors_ci95_bits": interval(direct_boot[:, 0] - paired_a_boot)})
    return result


def burst_split(losses: np.ndarray, entropy: np.ndarray, day: PreparedDay,
                left: np.ndarray, right: np.ndarray) -> dict:
    """Retrospective OFI skill split on realized 1 ms event counts, in bits."""
    tau = day.delays(1_000_000)
    result, points, boots = {}, [], []
    for group, mask in (("tau_1ms_zero", tau == 0), ("tau_1ms_at_least_one", tau >= 1)):
        point, boot = masked_skill(losses, mask, entropy, left, right)
        result[group] = {"n_anchors": int(mask.sum()),
                         "S_bits": [float(v) if np.isfinite(v) else None for v in point],
                         "S_ci95_bits": [interval(boot[:, j]) for j in range(len(TAU_GRID))],
                         "finite_bootstrap_resamples": np.isfinite(boot).sum(axis=0).tolist()}
        points.append(point)
        boots.append(boot)
    result["active_minus_quiet_bits"] = [float(v) if np.isfinite(v) else None for v in points[1] - points[0]]
    result["active_minus_quiet_ci95_bits"] = [interval(boots[1][:, j] - boots[0][:, j]) for j in range(len(TAU_GRID))]
    return result


def run_symbol(symbol: str, args: argparse.Namespace, manifests: dict) -> dict:
    """Fit one symbol, hold both test days out, and persist paired day artifacts."""
    started = time.perf_counter()
    days, provenance = {}, {}
    for day in ALL_DAYS:
        message = args.data_root / day / f"{symbol}_message_10.parquet"
        book = message.with_name(f"{symbol}_orderbook_10.parquet")
        days[day] = prepare_day(load_lobster(message, book), args.max_anchors_per_day)
        provenance[day] = {path.name: sha256(path) for path in (message, book)}
    xs, ys, mean, scale = split_arrays(days)
    print(f"{symbol}: " + ", ".join(f"{day}={len(days[day].anchors)} anchors" for day in ALL_DAYS), flush=True)
    fitted = fit_grid(xs, ys, TEST_DAYS, hgb_seeds=tuple(args.hgb_seeds), progress=lambda tau: print(
        f"{symbol}: fitting tau={tau} events", flush=True))
    symbol_dir = args.output / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    tests = {}
    for test_day in TEST_DAYS:
        day = days[test_day]
        evaluated = fitted["evaluation"][test_day]
        variants = {}
        for seconds in (1800, 300):
            left, right = moving_time_blocks(day.time_ns[day.anchors], args.resamples, seconds, seed=0)
            np.savez_compressed(symbol_dir / f"bootstrap_{test_day}_{seconds}s.npz",
                                left=left, right=right, test_anchor_indices=day.anchors,
                                test_time_ns=day.time_ns[day.anchors])
            scored = summarize_bootstrap(evaluated, fitted["entropy"], fitted["validation_skill"], day, left, right)
            scored["bootstrap"] = {"seed": 0, "resamples": args.resamples, "block_seconds": seconds,
                                   "blocks_per_resample": left.shape[1], "paired": True}
            if test_day == TEST_DAYS[0]:
                if seconds == 1800:
                    scored["direct_checks"] = direct_checks(days, xs, evaluated, fitted["entropy"], scored, left, right)
                scored["burst_split"] = burst_split(evaluated["losses"]["ofi_logit"], fitted["entropy"], day, left, right)
            variants[str(seconds)] = scored
        primary = variants.pop("1800")
        primary["sensitivity_5min"] = variants["300"]
        primary["hgb_seed_spread"] = evaluated["seed_spread"]
        tests[test_day] = primary
        plot_ticker({"ticker": f"{symbol}_{test_day}", **primary}, args.output / "figures")
        gate = primary["gate"]
        print(f"{symbol} {test_day}: best={gate['best_model']}, Delta={gate['delta_bits']:.8f} bits, CI={gate['delta_ci95_bits']}", flush=True)
    result = {"symbol": symbol, "tau_events": TAU_GRID.tolist(), "latency_ns": LATENCY_NS.tolist(),
              "data": {day: days[day].metadata for day in ALL_DAYS}, "source_sha256": provenance,
              "day_manifest_sha256": {day: value["sha256"] for day, value in manifests.items()},
              "feature_names": FEATURE_NAMES, "training_feature_mean": mean.tolist(),
              "training_feature_scale": scale.tolist(), "train_prior_up": ys["train"].mean(axis=0).tolist(),
              "train_prior_entropy_bits": fitted["entropy"].tolist(), "fit_metadata": fitted["fit_metadata"],
              "model_seeds": {"logit": [0], "hgb": args.hgb_seeds}, "tests": tests,
              "runtime_seconds": time.perf_counter() - started}
    write_json(symbol_dir / "metrics.json", result)
    return result


def write_reports(results: list[dict], manifest: dict, args: argparse.Namespace) -> None:
    """Write numeric tables from persisted JSON, including the historical comparison."""
    results = [json.loads((args.output / r["symbol"] / "metrics.json").read_text()) for r in results]
    gates = {}
    for day in TEST_DAYS:
        primary = {r["symbol"]: r["tests"][day]["gate"] for r in results}
        sensitivity = {r["symbol"]: r["tests"][day]["sensitivity_5min"]["gate"] for r in results}
        count = sum(g["ci_excludes_zero"] for g in primary.values())
        gates[day] = {"verdict": "PASS" if count >= 2 else "FAIL", "significant_symbols": count,
                      "required_significant_symbols": 2, "primary": primary, "sensitivity_5min": sensitivity}
    historical = json.loads((ROOT / "results/gate0/summary.json").read_text())
    summary = {"run_kind": manifest["run_kind"], "symbols": manifest["configuration"]["symbols"],
               "test_days": gates, "historical_gate0": historical, "full_run_command": FULL_COMMAND,
               "assumptions": ASSUMPTIONS}
    write_json(args.output / "summary.json", summary)
    title = "Gate 0b synthetic integration report" if args.fake_data else "Gate 0b report"
    lines = [f"# {title}", "", f"Run kind: **{manifest['run_kind']}**.", "",
             "The real-day Gate 0b result is unverified until the later full run." if args.fake_data else
             "Both held-out days use the same training and validation fits.", "",
             "## Commands and evidence", "", "Executed pipeline command:", "", "```sh", manifest["command"], "```", "",
             "Evidence: manifest.json, summary.json, per-symbol metrics.json and bootstrap artifacts under " + str(args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output) + ".",
             "All reported values below come from those result JSON files; bootstrap intervals condition on fixed fits.", "",
             "Later full run, from the repository root after placing complete converter output in data/raw/itch_lobster:",
             "", "```sh", FULL_COMMAND, "```", "",
             "## Configuration", "", "```json", json.dumps(manifest["configuration"], indent=2), "```", "",
             "Execution hardware: " + manifest["hardware"] + ". Reference: " + REFERENCE_HARDWARE + ".", "",
             "## Verdicts and primary results", "", "Skill is in bits/prediction: zero means test cross-entropy equals training-prior entropy.",
             "Delta is fast minus slow adjusted skill. Brackets are paired percentile intervals.", ""]
    slow = LATENCY_NS.tolist().index(2_000_000)
    for day, gate in gates.items():
        lines += [f"**{day}: {gate['verdict']}**; {gate['significant_symbols']} selected symbols exclude zero.", ""]
    lines += ["| Day | Symbol | Validation best | Fast A | Slow A | Delta, primary | Delta, sensitivity |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        for day in TEST_DAYS:
            test = r["tests"][day]
            gate, sense = test["gate"], test["sensitivity_5min"]["gate"]
            curve = test["latency_adjusted"][gate["best_model"]]
            lines.append(f"| {day} | {r['symbol']} | {gate['best_model']} | {curve['A_bits'][0]:.8f} | {curve['A_bits'][slow]:.8f} | {format_ci(gate['delta_bits'], gate['delta_ci95_bits'])} | {format_ci(sense['delta_bits'], sense['delta_ci95_bits'])} |")
    lines += ["", "## Half-life", "", "The raw curves and isotonic fits for every model are in metrics.json. These are fitted skill surrogates.",
              "A null upper interval endpoint with an unbounded flag denotes a right-censored crossing.", "",
              "| Day | Symbol | Selected model | Status | Events | Event interval | Clock median seconds | Clock quartiles seconds | Censored mass | Unidentified mass |",
              "|---|---|---|---|---:|---|---:|---|---:|---:|"]
    for r in results:
        for day in TEST_DAYS:
            test = r["tests"][day]
            name = test["gate"]["best_model"]
            half = test["half_life"][name]
            clock = half["wall_clock_seconds"]
            lines.append(f"| {day} | {r['symbol']} | {name} | {half['status']} | {half['half_life_events']} | {half['ci95_events']} | {clock['median'] if clock else None} | {[clock['q25'], clock['q75']] if clock else None} | {half['bootstrap_right_censored_fraction']} | {half['bootstrap_no_positive_initial_fraction']} |")
    lines += ["", "## Direct wall-clock check and burst split", "",
              "The December test's direct checks and full OFI subgroup curves, paired differences, boundary drops and finite bootstrap counts are stored in each metrics.json.", "",
              "| Symbol | Latency ns | Direct S | Mixture on same anchors | Direct minus mixture |",
              "|---|---:|---:|---:|---:|"]
    for r in results:
        for check in r["tests"][TEST_DAYS[0]]["direct_checks"]:
            lines.append(f"| {r['symbol']} | {check['latency_ns']} | {format_ci(check['direct_S_bits'], check['direct_S_ci95_bits'])} | {format_ci(check['A_same_anchors_bits'], check['A_same_anchors_ci95_bits'])} | {format_ci(check['direct_minus_A_same_anchors_bits'], check['direct_minus_A_same_anchors_ci95_bits'])} |")
    lines += ["", "## Historical Gate 0 comparison", "",
              "Source: the unchanged results/gate0/summary.json, also copied into this run's summary.json. Historical verdict: **" + historical["verdict"] + "**.",
              "Historical within-day splits and block duration differ from this whole-day experiment. Synthetic-date repetitions cannot establish a modern-market change in skill.", "",
              "| Historical symbol | Selected model | Historical Delta bits | Historical interval |",
              "|---|---|---:|---|"]
    for symbol, gate in historical["primary"].items():
        lines.append(f"| {symbol} | {gate['best_model']} | {gate['delta_bits']:.8f} | {gate['delta_ci95_bits']} |")
    lines += ["", "## Assumptions", "", *[f"- {value}" for value in ASSUMPTIONS]]
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    for day, gate in gates.items():
        print(f"Gate 0b {day} {gate['verdict']}: {gate['significant_symbols']}/{len(results)} symbols exclude zero ({manifest['run_kind']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "results/itch/selected_assets.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gate0b")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/gate0b_report.md")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--hgb-seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-anchors-per-day", type=int, help="deterministic reduced fixture sample")
    parser.add_argument("--fake-data", action="store_true", help="explicitly label synthetic fixture outputs")
    args = parser.parse_args()
    if args.resamples < 2 or len(set(args.hgb_seeds)) != len(args.hgb_seeds) or min(args.hgb_seeds) < 0:
        parser.error("Need at least two resamples and distinct nonnegative HGB seeds")
    if args.max_anchors_per_day is not None and (not args.fake_data or args.max_anchors_per_day < 2):
        parser.error("Anchor reduction requires --fake-data and at least two anchors")
    args.data_root, args.output, args.report = (path.resolve() for path in (args.data_root, args.output, args.report))
    if args.output == ROOT / "results/gate0" or (ROOT / "results/gate0") in args.output.parents:
        parser.error("Gate 0 results are protected")
    if args.report == ROOT / "docs/theory.md":
        parser.error("docs/theory.md is protected")
    started = time.perf_counter()
    try:
        assets = json.loads(args.assets.read_text())
        symbols = assets["symbols"]
        if (len(symbols) < 3 or len(set(symbols)) != len(symbols)
                or not all(isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,8}", s) for s in symbols)
                or assets.get("training_days") != list(TRAIN_DAYS)):
            raise ValueError("Assets must specify distinct safe symbols selected on the prescribed training days")
        manifests = read_inputs(args.data_root, symbols, args.fake_data)
        args.output.mkdir(parents=True, exist_ok=True)
        # Remove success summaries before a rerun, so a failed fit cannot claim a
        # completed new result using a previous run's verdict.
        for name in ("summary.json", "summary.md"):
            (args.output / name).unlink(missing_ok=True)
        manifest = {
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
            "run_kind": "synthetic_fixture" if args.fake_data else "real_days",
            "command": "MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python " + shlex.join(["scripts/run_gate0b.py", *sys.argv[1:]]),
            "full_run_command": FULL_COMMAND, "hardware": platform.platform() + " / " + platform.machine(),
            "reference_hardware": REFERENCE_HARDWARE, "python": sys.version,
            "versions": {m.__name__: m.__version__ for m in (np, polars, scipy, sklearn)},
            "configuration": {"symbols": symbols, "train_days": list(TRAIN_DAYS), "validation_day": VALIDATION_DAY,
                              "test_days": list(TEST_DAYS), "session_seconds": list(SESSION_SECONDS),
                              "tau_events": TAU_GRID.tolist(), "latency_ns": LATENCY_NS.tolist(),
                              "lookback_events": 50, "ofi_preceding_book_guard_events": 1,
                              "direct_check_latency_ns": [10_000, 1_000_000], "hgb_seeds": args.hgb_seeds,
                              "hgb_iterations": list(HGB_ITERATIONS), "logit_seed": 0,
                              "bootstrap_seed": 0, "resamples": args.resamples,
                              "primary_block_seconds": 1800, "sensitivity_block_seconds": 300,
                              "ci_percentiles": [2.5, 97.5], "max_anchors_per_day": args.max_anchors_per_day,
                              "out_of_time_refit": False, "required_significant_symbols": 2},
            "assets": assets, "assets_sha256": sha256(args.assets), "inputs": manifests,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
                Path(__file__), ROOT / "scripts/run_gate0.py", ROOT / "scripts/select_assets.py",
                ROOT / "scripts/symbol_stats.py", *sorted((ROOT / "src/lob_latency").glob("*.py"))]},
        }
        write_json(args.output / "manifest.json", manifest)
        with threadpool_limits(limits=1):
            results = [run_symbol(symbol, args, manifests) for symbol in symbols]
        manifest.update(status="complete", runtime_seconds=time.perf_counter() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_reports(results, manifest, args)
        write_json(args.output / "manifest.json", manifest)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Gate 0b failed: {error}\n")


if __name__ == "__main__":
    main()
