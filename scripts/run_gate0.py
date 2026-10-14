#!/usr/bin/env python3
"""Run Gate 0 on local samples, producing bit/event/ns results and figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

# Set this before any matplotlib import, including when called without env vars.
os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from lob_latency.events import merge_events
from lob_latency.features import FEATURE_NAMES, make_features, standardize
from lob_latency.io_lobster import load_lobster
from lob_latency.labels import (SPLITS, TAU_GRID, action_mask, delayed_labels,
                                 next_move, select_anchors)
from lob_latency.latency import (LATENCY_NAMES, LATENCY_NS, delay_statistics,
                                  event_delays, interpolation_weights)
from lob_latency.metrics import (block_means, classification_metrics, entropy_bits,
                                  interval, logit_losses_bits, masked_skill,
                                  moving_time_blocks)
from lob_latency.models import HGB_ITERATIONS, SEEDS, fit_model

TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
MODEL_NAMES = ["qi_logit", "ofi_logit", "hgb"]
REFERENCE_HARDWARE = "i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here)"


def write_json(path: Path, value: dict) -> None:
    """Strict JSON: undefined quantities use null, never NaN or Infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def curve_statistics(point: np.ndarray, boot: np.ndarray) -> dict:
    """Skill/difference intervals from the same bootstrap draws, in bits."""
    return {"S_bits": point.tolist(),
            "S_ci95_bits": [interval(boot[:, j]) for j in range(len(point))],
            "S_zero_minus_tau_bits": (point[0] - point).tolist(),
            "S_zero_minus_tau_ci95_bits": [interval(boot[:, 0] - boot[:, j])
                                           for j in range(len(point))]}


def latency_statistics(point: np.ndarray, boot: np.ndarray) -> dict:
    """Latency-adjusted skill and loss relative to the first latency, in bits."""
    return {"A_bits": point.tolist(),
            "A_ci95_bits": [interval(boot[:, j]) for j in range(len(point))],
            "A_fast_minus_L_bits": (point[0] - point).tolist(),
            "A_fast_minus_L_ci95_bits": [interval(boot[:, 0] - boot[:, j])
                                         for j in range(len(point))]}


def run_ticker(ticker: str, output: Path) -> dict:
    """Fit models and jointly resample test losses and event-delay distributions."""
    started = time.perf_counter()
    stem = ROOT / "data/raw/lobster" / f"{ticker}_2012-06-21_34200000_57600000"
    message_path, book_path = Path(f"{stem}_message_10.csv"), Path(f"{stem}_orderbook_10.csv")
    data = load_lobster(message_path, book_path)
    events = merge_events(data)
    direction, label_end = next_move(events.mid)
    features = make_features(events)
    anchors = select_anchors(events.time_ns, label_end, np.isfinite(features).all(axis=1))
    # A missing action mid is undefined at any tau, even if tau=64 is valid.
    candidate_counts = {split: len(ids) for split, ids in anchors.items()}
    for split, ids in anchors.items():
        valid = np.column_stack([delayed_labels(direction, ids, tau) for tau in TAU_GRID])
        anchors[split] = ids[np.all(valid >= 0, axis=1)]
        if not len(anchors[split]):
            raise ValueError(f"{ticker}: empty {split} anchor set")
    x, mean, scale = standardize(features, anchors["train"])
    xs = {split: x[ids] for split, ids in anchors.items()}
    ys = {split: np.column_stack([delayed_labels(direction, ids, tau) for tau in TAU_GRID])
          for split, ids in anchors.items()}
    n_test = len(anchors["test"])
    prior = ys["train"].mean(axis=0)
    entropy = np.array([entropy_bits(p) for p in prior])
    print(f"{ticker}: {len(events.time_ns):,} events; anchors "
          + ", ".join(f"{s}={len(i):,}" for s, i in anchors.items()), flush=True)

    losses = {name: np.empty((n_test, len(TAU_GRID))) for name in MODEL_NAMES}
    naive_losses = {name: np.empty((n_test, len(TAU_GRID))) for name in MODEL_NAMES}
    metadata = {name: {} for name in MODEL_NAMES}
    scores = {name: {"accuracy": [], "mcc": []} for name in MODEL_NAMES}
    naive_scores = {name: {"accuracy": [], "mcc": []} for name in MODEL_NAMES}
    zero_logits = {}
    validation_skill = {name: [] for name in MODEL_NAMES}
    seed_spread = []

    for j, tau in enumerate(TAU_GRID):
        print(f"{ticker}: fitting tau={tau} events", flush=True)
        for name in MODEL_NAMES:
            seeds = SEEDS if name == "hgb" else (0,)
            fitted_metadata, current_losses, current_logits, current_scores = [], [], [], []
            seed_skill = []
            for seed in seeds:
                model = fit_model(name, xs["train"], ys["train"][:, j],
                                  xs["validation"], ys["validation"][:, j], seed)
                logits = model.logits(xs["test"])
                loss = logit_losses_bits(ys["test"][:, j], logits)
                current_losses.append(loss)
                current_logits.append(logits)
                current_scores.append(classification_metrics(ys["test"][:, j], logits))
                fitted_metadata.append(model.metadata())
                seed_skill.append(float(entropy[j] - loss.mean()))
            losses[name][:, j] = np.mean(current_losses, axis=0)
            metadata[name][str(tau)] = fitted_metadata
            validation_skill[name].append(float(np.mean([
                m["validation_skill_bits"] for m in fitted_metadata])))
            for metric in ("accuracy", "mcc"):
                scores[name][metric].append(float(np.mean([s[metric] for s in current_scores])))
            if tau == 0:
                zero_logits[name] = current_logits
            old_losses = [logit_losses_bits(ys["test"][:, j], logits)
                          for logits in zero_logits[name]]
            naive_losses[name][:, j] = np.mean(old_losses, axis=0)
            old_scores = [classification_metrics(ys["test"][:, j], logits)
                          for logits in zero_logits[name]]
            for metric in ("accuracy", "mcc"):
                naive_scores[name][metric].append(float(np.mean([s[metric] for s in old_scores])))
            if name == "hgb":
                seed_spread.append({"tau_events": int(tau), "S_bits_by_seed": seed_skill,
                                    "S_min_max_bits": [min(seed_skill), max(seed_skill)],
                                    "S_std_bits": float(np.std(seed_skill)),
                                    "accuracy_by_seed": [s["accuracy"] for s in current_scores],
                                    "mcc_by_seed": [s["mcc"] for s in current_scores],
                                    "naive_S_bits_by_seed": [float(entropy[j] - l.mean())
                                                             for l in old_losses]})

    test_times = events.time_ns[anchors["test"]]
    left, right = moving_time_blocks(test_times)
    # Save the actual common block draws as an auditable/reusable result artifact.
    ticker_dir = output / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ticker_dir / "bootstrap_blocks.npz", left=left, right=right,
                        test_anchor_indices=anchors["test"], test_time_ns=test_times)
    delays = np.column_stack([event_delays(events.time_ns, anchors["test"], int(latency))
                              for latency in LATENCY_NS])
    weights = [interpolation_weights(delays[:, k], TAU_GRID) for k in range(len(LATENCY_NS))]
    weight_point = np.array([w.mean(axis=0) for w in weights])
    weight_boot = np.stack([block_means(w, left, right) for w in weights], axis=1)
    p_boot = block_means((delays >= 1).astype(float), left, right)
    mean_boot = block_means(delays, left, right)
    tail_boot = block_means((delays > 64).astype(float), left, right)
    latency_distribution = []
    for k, (latency, label) in enumerate(zip(LATENCY_NS, LATENCY_NAMES)):
        stat = delay_statistics(delays[:, k])
        stat.update({"latency_ns": int(latency), "label": label,
                     "p_at_least_one_ci95": interval(p_boot[:, k]),
                     "mean_events_ci95": interval(mean_boot[:, k]),
                     "p_beyond_64_ci95": interval(tail_boot[:, k])})
        latency_distribution.append(stat)

    curves, naive, latency_curves, naive_latency = {}, {}, {}, {}
    skill_points, skill_boots, a_points, a_boots = {}, {}, {}, {}
    for name in MODEL_NAMES:
        point = entropy - losses[name].mean(axis=0)
        boot = entropy - block_means(losses[name], left, right)
        naive_point = entropy - naive_losses[name].mean(axis=0)
        naive_boot = entropy - block_means(naive_losses[name], left, right)
        curves[name] = {**curve_statistics(point, boot), **scores[name],
                        "validation_S_bits": validation_skill[name]}
        naive[name] = {**curve_statistics(naive_point, naive_boot), **naive_scores[name]}
        curves[name]["retrained_minus_naive_bits"] = (point - naive_point).tolist()
        curves[name]["retrained_minus_naive_ci95_bits"] = [interval(boot[:, j] - naive_boot[:, j])
                                                          for j in range(len(TAU_GRID))]
        a_point = weight_point @ point
        a_boot = np.einsum("blt,bt->bl", weight_boot, boot)
        naive_a_point = weight_point @ naive_point
        naive_a_boot = np.einsum("blt,bt->bl", weight_boot, naive_boot)
        latency_curves[name] = latency_statistics(a_point, a_boot)
        naive_latency[name] = latency_statistics(naive_a_point, naive_a_boot)
        skill_points[name], skill_boots[name] = point, boot
        a_points[name], a_boots[name] = a_point, a_boot

    # Model choice uses calibrated validation S(0), with HGB's mean over seeds.
    best = max(MODEL_NAMES, key=lambda name: validation_skill[name][0])
    fast, slow = 0, LATENCY_NS.tolist().index(2_000_000)
    delta = float(a_points[best][fast] - a_points[best][slow])
    delta_ci = interval(a_boots[best][:, fast] - a_boots[best][:, slow])
    excludes_zero = bool(delta_ci[0] > 0 or delta_ci[1] < 0)

    direct_checks = []
    for latency in (10_000, 1_000_000):
        print(f"{ticker}: direct ofi_logit at {latency} ns", flush=True)
        realized = {split: event_delays(events.time_ns, ids, latency)
                    for split, ids in anchors.items()}
        masks = {split: action_mask(events.time_ns, label_end, ids, realized[split], split)
                 for split, ids in anchors.items()}
        direct_y = {split: delayed_labels(direction, ids[masks[split]], realized[split][masks[split]])
                    for split, ids in anchors.items()}
        model = fit_model("ofi_logit", xs["train"][masks["train"]], direct_y["train"],
                          xs["validation"][masks["validation"]], direct_y["validation"])
        mask = masks["test"]
        full_loss = np.zeros(n_test)
        direct_logits = model.logits(xs["test"][mask])
        full_loss[mask] = logit_losses_bits(direct_y["test"], direct_logits)
        direct_point, direct_boot = masked_skill(full_loss, mask, entropy_bits(model.prior), left, right)
        grid_point, grid_boot = masked_skill(losses["ofi_logit"], mask, entropy, left, right)
        k = LATENCY_NS.tolist().index(latency)
        paired_weight_point = weights[k][mask].mean(axis=0)
        numerator = block_means(np.where(mask[:, None], weights[k], 0), left, right)
        denominator = block_means(mask.astype(float), left, right)
        paired_weight_boot = numerator / denominator
        paired_a = float(paired_weight_point @ grid_point)
        paired_a_boot = np.einsum("bt,bt->b", paired_weight_boot, grid_boot)
        direct_checks.append({
            "latency_ns": latency, "metadata": model.metadata(),
            "kept_anchors": {s: int(m.sum()) for s, m in masks.items()},
            "fraction_dropped": {s: float(np.mean(~m)) for s, m in masks.items()},
            "direct_S_bits": float(direct_point[0]),
            "direct_S_ci95_bits": interval(direct_boot[:, 0]),
            "direct_accuracy_mcc": classification_metrics(direct_y["test"], direct_logits),
            "A_full_bits": float(a_points["ofi_logit"][k]),
            "A_full_ci95_bits": interval(a_boots["ofi_logit"][:, k]),
            "A_same_anchors_bits": paired_a,
            "A_same_anchors_ci95_bits": interval(paired_a_boot),
            "direct_minus_A_same_anchors_bits": float(direct_point[0] - paired_a),
            "direct_minus_A_same_anchors_ci95_bits": interval(direct_boot[:, 0] - paired_a_boot)})

    burst = {}
    burst_points, burst_boots = [], []
    burst_tau = delays[:, LATENCY_NS.tolist().index(1_000_000)]
    for group, mask in (("tau_1ms_zero", burst_tau == 0), ("tau_1ms_at_least_one", burst_tau >= 1)):
        point, boot = masked_skill(losses["ofi_logit"], mask, entropy, left, right)
        # This branch also handles samples with no anchors in a requested group.
        burst[group] = {"n_anchors": int(mask.sum()),
                        "S_bits": [float(p) if np.isfinite(p) else None for p in point],
                        "S_ci95_bits": [interval(boot[:, j]) for j in range(len(TAU_GRID))],
                        "finite_bootstrap_resamples": np.isfinite(boot).sum(axis=0).tolist()}
        burst_points.append(point)
        burst_boots.append(boot)
    difference = burst_points[1] - burst_points[0]
    burst["active_minus_quiet_bits"] = [float(v) if np.isfinite(v) else None for v in difference]
    burst["active_minus_quiet_ci95_bits"] = [interval(burst_boots[1][:, j] - burst_boots[0][:, j])
                                            for j in range(len(TAU_GRID))]

    unchanged = np.r_[False, np.all(events.book[1:] == events.book[:-1], axis=1)]
    result = {"ticker": ticker, "date": "2012-06-21", "tau_events": TAU_GRID.tolist(),
              "latency_ns": LATENCY_NS.tolist(), "model_seeds": {"logit": [0], "hgb": list(SEEDS)},
              "bootstrap": {"seed": 0, "resamples": len(left), "block_seconds": 300,
                            "blocks_per_resample": left.shape[1], "paired": True},
              "data": {"raw_messages": data.raw_rows, "dropped_halts": data.dropped_halts,
                       "sub_ns_timestamps_rounded": data.sub_ns_timestamps_rounded,
                       "retained_messages": len(data.time_ns), "merged_events": len(events.time_ns),
                       "invalid_mid_events": int(np.sum(~events.valid_mid)),
                       "unchanged_nonexecution_merged_events": int(np.sum(unchanged & ~events.execution)),
                       "execution_events": int(events.execution.sum()),
                       "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (message_path, book_path)}},
              "anchors": {s: {"count": len(ids), "before_common_label_filter": candidate_counts[s],
                              "first_time_ns": int(events.time_ns[ids[0]]),
                              "last_time_ns": int(events.time_ns[ids[-1]]),
                              "split_seconds": list(SPLITS[s])} for s, ids in anchors.items()},
              "feature_names": FEATURE_NAMES, "training_feature_mean": mean.tolist(),
              "training_feature_scale": scale.tolist(), "train_prior_up": prior.tolist(),
              "train_prior_entropy_bits": entropy.tolist(), "fit_metadata": metadata,
              "retrained": curves, "naive": naive, "hgb_seed_spread": seed_spread,
              "latency_distribution": latency_distribution, "latency_adjusted": latency_curves,
              "naive_latency_adjusted": naive_latency, "direct_checks": direct_checks,
              "burst_split": burst,
              "gate": {"best_model": best,
                       "validation_S0_bits": {name: validation_skill[name][0] for name in MODEL_NAMES},
                       "delta_bits": delta, "delta_ci95_bits": delta_ci,
                       "ci_excludes_zero": excludes_zero},
              "runtime_seconds": time.perf_counter() - started}
    write_json(ticker_dir / "metrics.json", result)
    plot_ticker(result, output / "figures")
    print(f"{ticker}: best={best}, Delta={delta:.8f} bits, CI={delta_ci}; "
          f"elapsed={result['runtime_seconds']:.1f}s", flush=True)
    return result


def plot_ticker(result: dict, output: Path) -> None:
    """Static, exportable plots with paired percentile intervals in bits."""
    output.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in MODEL_NAMES:
        curve = result["retrained"][name]
        line, = axes[0].plot(TAU_GRID, curve["S_bits"], marker="o", label=f"{name}: retrained")
        ci = np.array(curve["S_ci95_bits"])
        axes[0].fill_between(TAU_GRID, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=0.14)
        axes[0].plot(TAU_GRID, result["naive"][name]["S_bits"], linestyle="--",
                     color=line.get_color(), alpha=0.7)
        curve = result["latency_adjusted"][name]
        axes[1].plot(LATENCY_NS / 1e6, curve["A_bits"], marker="o", label=name)
        ci = np.array(curve["A_ci95_bits"])
        axes[1].fill_between(LATENCY_NS / 1e6, ci[:, 0], ci[:, 1], alpha=0.14)
    axes[0].set(xlabel="Delay (events)", ylabel="Skill S (bits/prediction)",
                title=f"{ticker}: solid retrained; dashed naive")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[1].set(xlabel="Assumed latency (ms)", ylabel="Adjusted skill A (bits/prediction)",
                title=f"{ticker}: empirical event-delay mixture", xscale="log")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / f"{ticker}_skill.png", dpi=160)
    plt.close(fig)


def format_ci(point: float, bounds: list | None) -> str:
    if bounds is None:
        return f"{point:.6f} [undefined]"
    return f"{point:.6f} [{bounds[0]:.6f}, {bounds[1]:.6f}]"


def write_summary(results: list[dict], manifest: dict, output: Path) -> None:
    """Generate tables exclusively from the persisted per-ticker result values."""
    n_significant = sum(r["gate"]["ci_excludes_zero"] for r in results)
    complete = set(r["ticker"] for r in results) == set(TICKERS)
    verdict = ("PASS" if n_significant >= 2 else "FAIL") if complete else "INCOMPLETE"
    summary = {"verdict": verdict, "significant_tickers": n_significant,
               "required_significant_tickers": 2, "required_tickers": TICKERS,
               "completed_tickers": [r["ticker"] for r in results],
               "primary": {r["ticker"]: r["gate"] for r in results}}
    write_json(output / "summary.json", summary)
    lines = ["# Gate 0 results", "", f"**{verdict}**: {n_significant} tickers have a paired interval excluding zero.",
             "", f"Execution hardware: {manifest['cpu']}. Reference: {REFERENCE_HARDWARE}.", "",
             "All skill values are bits/prediction; brackets are paired 95% percentile intervals.", "",
             "| Ticker | Best on validation | A(100 ns) | A(2 ms) | Delta | Excludes zero |",
             "|---|---|---:|---:|---:|---|"]
    k = LATENCY_NS.tolist().index(2_000_000)
    for r in results:
        gate = r["gate"]
        curve = r["latency_adjusted"][gate["best_model"]]
        lines.append(f"| {r['ticker']} | {gate['best_model']} | "
                     f"{format_ci(curve['A_bits'][0], curve['A_ci95_bits'][0])} | "
                     f"{format_ci(curve['A_bits'][k], curve['A_ci95_bits'][k])} | "
                     f"{format_ci(gate['delta_bits'], gate['delta_ci95_bits'])} | "
                     f"{gate['ci_excludes_zero']} |")
    for r in results:
        lines += ["", f"## {r['ticker']}", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                  "", "| Split | Eligible anchors |", "|---|---:|"]
        lines.extend(f"| {s} | {a['count']} |" for s, a in r["anchors"].items())
        for variant in ("retrained", "naive"):
            lines += ["", f"### {variant}", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                      "", "| Model | Delay (events) | S [95% CI] | Accuracy | MCC | S(0)-S(tau) [95% CI] |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name, c in r[variant].items():
                for j, tau in enumerate(TAU_GRID):
                    lines.append(f"| {name} | {tau} | {format_ci(c['S_bits'][j], c['S_ci95_bits'][j])} | "
                                 f"{c['accuracy'][j]:.6f} | {c['mcc'][j]:.6f} | "
                                 f"{format_ci(c['S_zero_minus_tau_bits'][j], c['S_zero_minus_tau_ci95_bits'][j])} |")
        lines += ["", "### Wall-clock conversion", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                  "", "| L | P(tau>=1) | Mean tau | p50 | p90 | p99 | P(tau>64) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for d in r["latency_distribution"]:
            q = d["percentiles_events"]
            lines.append(f"| {d['label']} | {d['p_at_least_one']:.6f} | {d['mean_events']:.6f} | "
                         f"{q['50']:.2f} | {q['90']:.2f} | {q['99']:.2f} | {d['p_beyond_64']:.6f} |")
        for variant in ("latency_adjusted", "naive_latency_adjusted"):
            lines += ["", f"### {variant}", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                      "", "| Model | L | A [95% CI] | A(100 ns)-A(L) [95% CI] |",
                      "|---|---|---:|---:|"]
            for name, c in r[variant].items():
                for j, label in enumerate(LATENCY_NAMES):
                    lines.append(f"| {name} | {label} | {format_ci(c['A_bits'][j], c['A_ci95_bits'][j])} | "
                                 f"{format_ci(c['A_fast_minus_L_bits'][j], c['A_fast_minus_L_ci95_bits'][j])} |")
        lines += ["", "### Direct checks (same retained test anchors for the paired comparison)", "",
                  f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.", "",
                  "| L (ns) | Direct S [95% CI] | Paired A [95% CI] | Direct - A [95% CI] | Dropped train/val/test |",
                  "|---:|---:|---:|---:|---|"]
        for d in r["direct_checks"]:
            dropped = "/".join(f"{d['fraction_dropped'][s]:.6f}" for s in SPLITS)
            lines.append(f"| {d['latency_ns']} | {format_ci(d['direct_S_bits'], d['direct_S_ci95_bits'])} | "
                         f"{format_ci(d['A_same_anchors_bits'], d['A_same_anchors_ci95_bits'])} | "
                         f"{format_ci(d['direct_minus_A_same_anchors_bits'], d['direct_minus_A_same_anchors_ci95_bits'])} | {dropped} |")
        lines += ["", "### OFI burst split", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                  "", "| Group | Anchors | Delay (events) | S [95% CI] |", "|---|---:|---:|---:|"]
        for group in ("tau_1ms_zero", "tau_1ms_at_least_one"):
            c = r["burst_split"][group]
            for j, tau in enumerate(TAU_GRID):
                text = "undefined" if c["S_bits"][j] is None else format_ci(c["S_bits"][j], c["S_ci95_bits"][j])
                lines.append(f"| {group} | {c['n_anchors']} | {tau} | {text} |")
        lines += ["", "### HGB seed spread", "", f"Hardware: {manifest['cpu']}; reference: {REFERENCE_HARDWARE}.",
                  "", "| Delay (events) | S min | S max | S std |", "|---:|---:|---:|---:|"]
        for c in r["hgb_seed_spread"]:
            lines.append(f"| {c['tau_events']} | {c['S_min_max_bits'][0]:.9f} | "
                         f"{c['S_min_max_bits'][1]:.9f} | {c['S_std_bits']:.9f} |")
        lines += ["", f"![{r['ticker']} skill curves](figures/{r['ticker']}_skill.png)"]
    lines += ["", "## Assumptions", "",
              "- Each retained timestamp group is an event, using its final visible book, as specified.",
              "- Sub-nanosecond serialization tails in raw times are rounded to the nearest ns from decimal digits.",
              "- Missing mids retain their event index but cannot be label references or endpoints.",
              "- OFI requires one extra previous book inside the split; this is a conservative warmup guard.",
              "- The most recent execution within a merged group supplies its trade sign.",
              "- HGB summaries average losses and metrics across seeds; they are not a probability ensemble.",
              "- Naive weights and temperature are frozen at zero delay; each scored delay uses its own training prior.",
              "- Temperature fitting searches log(T) between the bounds recorded in manifest.json.",
              "- Bootstrap intervals condition on fitted models, training priors and validation model selection.",
              "- Complete overlapping time blocks start uniformly at eligible test anchors and retain variable event counts.",
              "- Direct checks use the common anchors, drop out-of-split labels, and pair A on the same retained test anchors.",
              "- Grid interpolation and its constant tail are approximations; the direct checks quantify disagreement.",
              "- Assumed latency is swept; inference latency and FPGA behavior are unmeasured.",
              "- This single historical day cannot establish across-day generalization or trading profitability."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Gate 0 {verdict}: {n_significant}/{len(results)} tickers exclude zero", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=TICKERS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/gate0")
    args = parser.parse_args()
    if len(set(args.tickers)) != len(args.tickers):
        parser.error("Tickers must be unique")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                         text=True, check=False).stdout.strip() or platform.processor()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "cpu": cpu,
                "platform": platform.platform(), "python": sys.version, "reference_hardware": REFERENCE_HARDWARE,
                "versions": {m.__name__: m.__version__ for m in (np, polars, scipy, sklearn, matplotlib)},
                "argv": sys.argv, "cwd": str(Path.cwd()),
                "command": "MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python "
                           + shlex.join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
                "configuration": {"tickers": args.tickers, "date": "2012-06-21", "lookback_events": 50,
                                  "ofi_preceding_book_guard_events": 1, "book_levels": 10,
                                  "tau_grid_events": TAU_GRID.tolist(), "latency_grid_ns": LATENCY_NS.tolist(),
                                  "direct_check_latency_ns": [10_000, 1_000_000],
                                  "splits_seconds_after_midnight": SPLITS,
                                  "tick_dollars": 0.01, "price_units_per_dollar": 10_000,
                                  "hgb_seeds": list(SEEDS), "logit_seed": 0, "hgb_max_iter_candidates": HGB_ITERATIONS,
                                  "hgb_learning_rate": 0.1, "hgb_early_stopping": False,
                                  "hgb_max_leaf_nodes": 31, "hgb_min_samples_leaf": 20,
                                  "hgb_l2_regularization": 0.0, "logit_C": 1.0, "logit_max_iter": 1000,
                                  "temperature_log_bounds": [-8., 8.], "temperature_xatol": 1e-8,
                                  "bootstrap_seed": 0, "bootstrap_resamples": 1000, "bootstrap_block_seconds": 300,
                                  "ci_percentiles": [2.5, 97.5], "threads": 1,
                                  "classification_threshold": 0.5, "gate_required_significant_tickers": 2},
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), *sorted((ROOT / "src/lob_latency").glob("*.py"))]}}
    write_json(output / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        results = [run_ticker(ticker, output) for ticker in args.tickers]
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_summary(results, manifest, output)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
