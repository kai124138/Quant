"""Offline theory checks: event counts, latency in seconds, skill in bits/prediction.

Run from any directory with the repository's .venv/bin/python. No fitted market
predictor is simulated: PGF checks assume exponential *event-time* skill decay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import polars as pl
from scipy import __version__ as scipy_version
from scipy.optimize import isotonic_regression


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
N_TRIALS = 200_000
TICKERS = ("AAPL", "MSFT", "INTC", "AMZN", "GOOG")
REFERENCE_RATES = (17.1, 28.6, 26.7, 11.5, 6.3)
REFERENCE_GAPS_US = (639, 51, 66, 620, 830)
TAU_C_GRID = (5, 20, 100)
DELTA_1 = 0.10
DELTA_L = 0.002
SESSION_START_S = 34200
SESSION_END_S = 57600
TRIM_START_S = 35100
TRIM_END_S = 57000
COUNT_RATE = 20_000.0
RHO = 0.8
FAST_LATENCY_S = 0.1e-6
SLOW_LATENCY_S = 50e-6
OVERHEADS_US = (0, 1, 5, 40)
BRANCHING_RATIO = 0.7
HAWKES_BETA_PER_S = 200_000.0
HAWKES_BURN_S = 400e-6


def moments(values: np.ndarray) -> dict:
    """Sample mean, variance and standard error in the input's units."""
    variance = float(np.var(values, ddof=1))
    return {
        "mean": float(np.mean(values)),
        "variance": variance,
        "se": float(np.sqrt(variance / len(values))),
    }


def ratio_estimate(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, np.ndarray]:
    """Ratio and paired delta-method influence, both dimensionless."""
    ratio = float(np.mean(numerator) / np.mean(denominator))
    influence = (numerator - ratio * denominator) / np.mean(denominator)
    return ratio, influence


def poisson_counts(rng: np.random.Generator, lengths_s: np.ndarray, rate: float) -> np.ndarray:
    """Coupled Poisson counts at sorted window lengths (seconds, events/second)."""
    increments = rng.poisson(rate * np.diff(np.r_[0, lengths_s]), (N_TRIALS, len(lengths_s)))
    return np.cumsum(increments, axis=1)


def hawkes_counts(rng: np.random.Generator, lengths_s: np.ndarray) -> tuple[np.ndarray, dict]:
    """Independent stationary-approximate Hawkes windows via Poisson clusters.

    Kernel is n*beta*exp(-beta*t), beta in inverse seconds. Immigrants start
    burn_s seconds before each window. The omitted prehistory has an analytic
    expected-count bound, recorded below; no empty-history claim is made.
    """
    rate, n, beta, burn = COUNT_RATE, BRANCHING_RATIO, HAWKES_BETA_PER_S, HAWKES_BURN_S
    maximum = float(lengths_s[-1])
    immigrant_counts = rng.poisson(rate * (1 - n) * (burn + maximum), N_TRIALS)
    ids = np.repeat(np.arange(N_TRIALS), immigrant_counts)
    births = rng.uniform(-burn, maximum, len(ids))
    counts = np.zeros((N_TRIALS, len(lengths_s)), dtype=np.int64)
    generations = 0
    simulated_events = 0
    while len(ids):
        simulated_events += len(ids)
        visible = births > 0
        for j, length in enumerate(lengths_s):
            selected = visible & (births <= length)
            counts[:, j] += np.bincount(ids[selected], minlength=N_TRIALS)
        children = rng.poisson(n, len(ids))
        child_ids = np.repeat(ids, children)
        child_times = np.repeat(births, children) + rng.exponential(1 / beta, len(child_ids))
        keep = child_times <= maximum
        ids, births = child_ids[keep], child_times[keep]
        generations += 1
        if generations > 1000:
            raise RuntimeError("Hawkes cluster simulation failed to terminate")
    relaxation = beta * (1 - n)
    missing_bound = rate * n / relaxation * np.exp(-relaxation * burn) * (-np.expm1(-relaxation * maximum))
    return counts, {
        "branching_ratio": n,
        "beta_per_second": beta,
        "immigrant_rate_per_second": rate * (1 - n),
        "target_mean_rate_per_second": rate,
        "burn_seconds": burn,
        "anchor_sampling": "independent uniform calendar-time windows, not event-Palm anchors",
        "omitted_expected_count_bound_per_window": float(missing_bound),
        "omitted_expected_count_bound_all_windows": float(N_TRIALS * missing_bound),
        "generations": generations,
        "simulated_events_including_burn": simulated_events,
    }


def check_arrival_models(rng: np.random.Generator) -> dict:
    """Check PGFs and overhead ratios; lengths are seconds, counts are events."""
    overhead_s = np.array(OVERHEADS_US) * 1e-6
    fast_lengths = FAST_LATENCY_S + overhead_s
    slow_lengths = SLOW_LATENCY_S + overhead_s
    lengths = np.unique(np.r_[fast_lengths, slow_lengths])
    fast_columns = np.searchsorted(lengths, fast_lengths)
    slow_columns = np.searchsorted(lengths, slow_lengths)
    poisson = poisson_counts(rng, lengths, COUNT_RATE)
    hawkes, hawkes_parameters = hawkes_counts(rng, lengths)
    g_poisson, g_hawkes = RHO**poisson, RHO**hawkes
    closed = np.exp(-COUNT_RATE * lengths * (1 - RHO))
    pgf_rows = []
    for j, length in enumerate(lengths):
        mc = moments(g_poisson[:, j])
        relative_error = abs(mc["mean"] / closed[j] - 1)
        pgf_rows.append({
            "length_seconds": float(length), "closed_form": float(closed[j]),
            "monte_carlo": mc, "relative_error": float(relative_error),
            "passed": bool(relative_error < 0.01),
        })
    rows, influences = [], {"poisson": [], "hawkes": []}
    for j, overhead in enumerate(OVERHEADS_US):
        row = {"c_microseconds": overhead}
        for name, g in (("poisson", g_poisson), ("hawkes", g_hawkes)):
            ratio, influence = ratio_estimate((1 - DELTA_1) * g[:, fast_columns[j]], g[:, slow_columns[j]])
            influences[name].append(influence)
            delta_se = moments(influence - influences[name][0])["se"]
            row[name] = {
                "ratio": ratio, "se": moments(influence)["se"],
                "change_from_c0_se": delta_se,
            }
        row["poisson_closed_ratio"] = float((1 - DELTA_1) * closed[fast_columns[j]] / closed[slow_columns[j]])
        rows.append(row)
    for row in rows:
        for name in influences:
            row[name]["change_from_c0"] = row[name]["ratio"] - rows[0][name]["ratio"]
    column = int(slow_columns[0])
    poisson_stats = moments(poisson[:, column])
    hawkes_stats = moments(hawkes[:, column])
    poisson_g, hawkes_g = moments(g_poisson[:, column]), moments(g_hawkes[:, column])
    gap = hawkes_g["mean"] - poisson_g["mean"]
    gap_se = np.hypot(poisson_g["se"], hawkes_g["se"])
    poisson_invariant = all(
        abs(row["poisson"]["change_from_c0"]) <= 5 * row["poisson"]["change_from_c0_se"] + 1e-12
        for row in rows
    )
    hawkes_changes = any(
        abs(row["hawkes"]["change_from_c0"]) > 5 * row["hawkes"]["change_from_c0_se"]
        for row in rows[1:]
    )
    return {
        "parameters": {
            "trials": N_TRIALS, "lambda_per_second": COUNT_RATE, "rho": RHO,
            "fast_latency_seconds": FAST_LATENCY_S, "slow_latency_seconds": SLOW_LATENCY_S,
            "fast_skill_bits": 1 - DELTA_1, "slow_skill_bits": 1.0,
            "comparison_sigma_multiplier": 5,
        },
        "poisson_pgf": {
            "relative_error_limit": 0.01,
            "rows": pgf_rows,
            "passed": all(row["passed"] for row in pgf_rows),
        },
        "hawkes_parameters": hawkes_parameters,
        "overdispersion": {
            "window_seconds": float(lengths[column]),
            "target_mean_count": float(COUNT_RATE * lengths[column]),
            "poisson_counts": poisson_stats, "hawkes_counts": hawkes_stats,
            "poisson_pgf": poisson_g, "hawkes_pgf": hawkes_g,
            "pgf_gap": float(gap), "pgf_gap_se": float(gap_se),
            "passed": bool(
                gap > 5 * gap_se
                and hawkes_stats["variance"] > poisson_stats["variance"]
                and abs(hawkes_stats["mean"] - COUNT_RATE * lengths[column]) < 5 * hawkes_stats["se"]
            ),
        },
        "overhead": {
            "rows": rows, "poisson_invariance_passed": poisson_invariant,
            "hawkes_noninvariance_passed": hawkes_changes,
            "passed": poisson_invariant and hawkes_changes,
        },
    }


def check_crossover(rng: np.random.Generator) -> dict:
    """Brute-force coupled arrival simulation over latency gaps in seconds."""
    gaps = np.arange(241) * 0.25e-6
    fast_counts = rng.poisson(COUNT_RATE * FAST_LATENCY_S, N_TRIALS)
    slow_counts = fast_counts.copy()
    fast_g = RHO**fast_counts
    fast_score = (1 - DELTA_1) * np.mean(fast_g)
    rows = []
    for j, gap in enumerate(gaps):
        if j:
            slow_counts += rng.poisson(COUNT_RATE * (gap - gaps[j - 1]), N_TRIALS)
        slow_g = RHO**slow_counts
        difference = moments((1 - DELTA_1) * fast_g - slow_g)
        rows.append({"gap_seconds": float(gap), "fast_minus_slow_bits": difference})
    first = next(j for j, row in enumerate(rows) if row["fast_minus_slow_bits"]["mean"] > 0)
    lower, upper = rows[first - 1], rows[first]
    lo_score, hi_score = lower["fast_minus_slow_bits"]["mean"], upper["fast_minus_slow_bits"]["mean"]
    estimate = lower["gap_seconds"] - lo_score * (upper["gap_seconds"] - lower["gap_seconds"]) / (hi_score - lo_score)
    threshold = float(np.log(1 / (1 - DELTA_1)) / (COUNT_RATE * (1 - RHO)))
    derivative = COUNT_RATE * (1 - RHO) * fast_score
    se_seconds = upper["fast_minus_slow_bits"]["se"] / derivative
    tolerance = 5 * se_seconds + float(gaps[1])
    return {
        "trials": N_TRIALS, "grid_step_seconds": float(gaps[1]),
        "analytic_gap_seconds": threshold, "simulated_gap_seconds": float(estimate),
        "simulated_bracket_seconds": [lower["gap_seconds"], upper["gap_seconds"]],
        "delta_method_se_seconds": float(se_seconds), "tolerance_seconds": tolerance,
        "absolute_error_seconds": float(abs(estimate - threshold)),
        "rows": rows, "passed": bool(abs(estimate - threshold) < tolerance),
    }


def merge_events(times_ns: np.ndarray, types: np.ndarray, books: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Top-book-changing messages plus trades, merged at identical nanoseconds.

    Return event timestamps (ns) and twice-mid prices (input price units). Each
    timestamp takes the last eligible message's post-message book. Compare to
    the actual preceding row before filtering; the first state is initialization.
    """
    if len(times_ns) != len(books) or len(types) != len(books):
        raise ValueError("Message/book row count mismatch")
    if np.any(np.diff(times_ns) < 0):
        raise ValueError("Messages are not chronological")
    if np.any(~np.isin(types, [1, 2, 3, 4, 5, 6, 7])):
        raise ValueError("Unknown message type")
    changed = np.r_[True, np.any(books[1:] != books[:-1], axis=1)]
    eligible = np.isin(types, [1, 2, 3, 4, 5, 6]) & (changed | np.isin(types, [4, 5, 6]))
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError("No eligible book messages or trades")
    retained_times = times_ns[indices]
    ends = np.r_[np.flatnonzero(np.diff(retained_times) != 0), len(indices) - 1]
    last = indices[ends]
    ask, bid = books[last, 0], books[last, 2]
    valid = (np.abs(ask) != 9999999999) & (np.abs(bid) != 9999999999)
    twice_mid = np.where(valid, ask + bid, np.iinfo(np.int64).min)
    return times_ns[last], twice_mid, {
        "raw_message_rows": len(times_ns), "halt_rows_dropped": int(np.sum(types == 7)),
        "unchanged_nontrade_rows_dropped": int(np.sum(~eligible & (types != 7))),
        "retained_message_rows": len(indices), "merged_event_count": len(last),
        "invalid_mid_events": int(np.sum(~valid)),
    }


def window_statistics(times_ns: np.ndarray, twice_mid: np.ndarray, start_s: int, end_s: int) -> dict:
    """Rates in events/second, transitions in events, gaps in microseconds."""
    use = (times_ns >= start_s * 10**9) & (times_ns < end_s * 10**9)
    times, mid = times_ns[use], twice_mid[use]
    valid = mid != np.iinfo(np.int64).min
    if len(times) < 2 or not np.all(valid):
        raise ValueError("Window too short or contains dummy best prices; segment before measuring p")
    changes = np.flatnonzero(mid[1:] != mid[:-1]) + 1
    p = len(changes) / (len(times) - 1)
    if not 0 < p < 1 or len(times) <= max(TAU_C_GRID):
        raise ValueError("Need a nondegenerate mid-change rate and enough events for the survival grid")
    tau_c = float(-1 / np.log1p(-p))
    counts = np.searchsorted(times, times + round(DELTA_L * 1e9), side="right") - np.arange(len(times)) - 1
    complete = times + round(DELTA_L * 1e9) < end_s * 10**9
    counts = counts[complete]
    survival = []
    for tau in TAU_C_GRID:
        anchors = np.arange(len(times) - tau)
        # No mid change in (i, i+tau] guarantees that the L-next label survives.
        next_pos = np.searchsorted(changes, anchors, side="right")
        next_change = np.r_[changes, len(times)][next_pos]
        survival.append({
            "tau_events": tau, "anchors": len(anchors),
            "empirical_no_mid_change_probability": float(np.mean(next_change > anchors + tau)),
            "geometric_probability": float((1 - p)**tau),
        })
    positive = counts[counts > 0]
    return {
        "start_seconds_after_midnight": start_s, "end_seconds_after_midnight": end_s,
        "duration_seconds": end_s - start_s, "event_count": len(times),
        "mean_rate_per_second": len(times) / (end_s - start_s),
        "median_gap_microseconds": float(np.median(np.diff(times)) / 1000),
        "mid_change_count": len(changes), "eligible_adjacent_transitions": len(times) - 1,
        "mid_changes_per_event_p": p, "events_per_mid_change": 1 / p,
        "geometric_tau_c_events": tau_c, "geometric_rho": 1 - p,
        "label_survival": survival,
        "wall_clock_diagnostic": {
            "latency_seconds": DELTA_L, "complete_anchors": len(counts),
            "probability_positive_count": float(np.mean(counts > 0)),
            "mean_count": float(np.mean(counts)),
            "mean_rate_from_counts_per_second": float(np.mean(counts) / DELTA_L),
            "positive_count_mean": float(np.mean(positive)),
            "positive_count_mean_rate_per_second": float(np.mean(positive) / DELTA_L),
            "warning": "Conditioning on future count is an offline diagnostic, not a causal feature or a Poisson fit.",
            "anchor_population": "All events in this window with a complete latency interval, before model lookback/label/split filtering.",
            "pgfs": [
                {
                    "tau_c_events": tau, "empirical_pgf": float(np.mean(np.exp(-counts / tau))),
                    "positive_count_pgf": float(np.mean(np.exp(-positive / tau))),
                    "minimum_deterministic_count_for_h1_advantage": int(np.floor(np.log(1 / (1 - DELTA_1)) * tau)) + 1,
                    "fraction_above_h1_count_threshold": float(np.mean(counts > np.log(1 / (1 - DELTA_1)) * tau)),
                }
                for tau in (*TAU_C_GRID, tau_c)
            ],
        },
    }


def measure_lobster() -> list[dict]:
    """Read local message/book pairs only; no network or model fitting."""
    output = []
    for ticker, reference_rate, reference_gap in zip(TICKERS, REFERENCE_RATES, REFERENCE_GAPS_US):
        stem = f"{ticker}_2012-06-21_34200000_57600000"
        paths = [ROOT / f"data/raw/lobster/{stem}_{kind}_10.csv" for kind in ("message", "orderbook")]
        messages = pl.read_csv(paths[0], has_header=False, columns=[0, 1], schema_overrides={"column_1": pl.Float64, "column_2": pl.Int64})
        books = pl.read_csv(paths[1], has_header=False, infer_schema_length=100).to_numpy()
        times_ns = np.rint(messages[:, 0].to_numpy() * 1e9).astype(np.int64)
        times, mid, event_metadata = merge_events(times_ns, messages[:, 1].to_numpy(), books)
        raw_session = times_ns[(times_ns >= SESSION_START_S * 10**9) & (times_ns < SESSION_END_S * 10**9)]
        row = {
            "ticker": ticker,
            "files": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
            "event_metadata": event_metadata,
            "raw_message_session": {
                "message_count": len(raw_session),
                "mean_message_rate_per_second": len(raw_session) / (SESSION_END_S - SESSION_START_S),
                "median_message_gap_microseconds": float(np.median(np.diff(raw_session)) / 1000),
                "zero_gap_fraction": float(np.mean(np.diff(raw_session) == 0)),
            },
            "supplied_reference": {
                "mean_rate_per_second": reference_rate, "median_gap_microseconds": reference_gap,
                "source": "docs/theory_spec.md section 6(e), supplied rounded values; not silently substituted for recomputed values",
            },
            "full_session": window_statistics(times, mid, SESSION_START_S, SESSION_END_S),
            "trimmed_session": window_statistics(times, mid, TRIM_START_S, TRIM_END_S),
        }
        output.append(row)
        print(f"{ticker}: trimmed rate={row['trimmed_session']['mean_rate_per_second']:.6f}/s, p={row['trimmed_session']['mid_changes_per_event_p']:.6f}, tau_c={row['trimmed_session']['geometric_tau_c_events']:.6f} events", flush=True)
    return output


def h1_cell(ticker: str, rate: float, tau_c: float, source: str) -> dict:
    """H1/H3 plug-in prediction: seconds, events/second, dimensionless ratio."""
    rho = float(np.exp(-1 / tau_c))
    threshold = float(-np.log1p(-DELTA_1))
    lhs = rate * (1 - rho) * DELTA_L
    return {
        "ticker": ticker, "rate_source": source, "lambda_per_second": rate,
        "tau_c_events": tau_c, "rho": rho, "lambda_one_minus_rho_delta_l": lhs,
        "required_dimensionless_threshold": threshold,
        "critical_rate_per_second": threshold / ((1 - rho) * DELTA_L),
        "critical_latency_gap_seconds": threshold / (rate * (1 - rho)),
        "fast_to_slow_score_ratio": float((1 - DELTA_1) * np.exp(lhs)),
        "fast_wins": bool(lhs > threshold),
    }


def make_h1_tables(lobster: list[dict]) -> dict:
    """Compute reference, remeasured, and measured-survival event-time tables."""
    reference, measured, measured_survival = [], [], []
    for row in lobster:
        ticker = row["ticker"]
        rate = row["trimmed_session"]["mean_rate_per_second"]
        for tau in TAU_C_GRID:
            reference.append(h1_cell(ticker, row["supplied_reference"]["mean_rate_per_second"], tau, "supplied_reference"))
            measured.append(h1_cell(ticker, rate, tau, "trimmed_remeasurement"))
        measured_survival.append(h1_cell(ticker, rate, row["trimmed_session"]["geometric_tau_c_events"], "trimmed_remeasurement"))
    return {
        "delta_1": DELTA_1, "delta_l_seconds": DELTA_L,
        "log_h1_threshold": float(-np.log1p(-DELTA_1)),
        "reference_rate_grid": reference, "remeasured_rate_grid": measured,
        "measured_survival_grid": measured_survival,
        "reference_cells": len(reference), "reference_fast_wins": sum(row["fast_wins"] for row in reference),
        "remeasured_cells": len(measured), "remeasured_fast_wins": sum(row["fast_wins"] for row in measured),
        "measured_survival_fast_wins": sum(row["fast_wins"] for row in measured_survival),
    }


def half_life(delays: np.ndarray, skill: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Nonincreasing least-squares fit and linearly interpolated half-life (events).

    This returns a clipped skill surrogate, not an estimator of unrestricted
    mutual information. An unresolved tail is right-censored, never extrapolated.
    """
    if len(delays) != len(skill) or len(delays) < 2 or delays[0] != 0 or np.any(np.diff(delays) <= 0):
        raise ValueError("Delays must be increasing and start at zero")
    if not np.all(np.isfinite(skill)) or not np.all(np.isfinite(delays)):
        raise ValueError("Half-life inputs must be finite")
    fit = np.maximum(isotonic_regression(skill, weights=weights, increasing=False).x, 0)
    if fit[0] == 0:
        return {"fit_bits": fit.tolist(), "half_life_events": None, "status": "no_positive_initial_skill"}
    target = fit[0] / 2
    crossing = np.flatnonzero(fit <= target)
    if not len(crossing):
        return {"fit_bits": fit.tolist(), "half_life_events": None, "status": "right_censored", "lower_bound_events": float(delays[-1])}
    j = int(crossing[0])
    fraction = (fit[j - 1] - target) / (fit[j - 1] - fit[j])
    estimate = delays[j - 1] + fraction * (delays[j] - delays[j - 1])
    return {"fit_bits": fit.tolist(), "half_life_events": float(estimate), "status": "observed"}


def analytic_examples() -> dict:
    """Explicit finite-alphabet examples (bits), plus two-timescale overhead."""
    def binary_entropy(p: float) -> float:
        return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))

    survival = 0.5
    initial_accuracy = 0.9
    original_information = 1 - binary_entropy(initial_accuracy)
    mixed_accuracy = survival * initial_accuracy + (1 - survival) * 0.5
    weights = [0.5, 0.5]
    rho_components = [float(np.exp(-1 / 5)), float(np.exp(-1 / 100))]
    mixture_rows = []
    for c in OVERHEADS_US:
        def g(length: float) -> float:
            return float(sum(w * np.exp(-COUNT_RATE * length * (1 - r)) for w, r in zip(weights, rho_components)))
        ratio = (1 - DELTA_1) * g(FAST_LATENCY_S + c * 1e-6) / g(SLOW_LATENCY_S + c * 1e-6)
        mixture_rows.append({"c_microseconds": c, "fast_to_slow_ratio": ratio})
    return {
        "hidden_survival_example": {
            "survival_probability": survival, "initial_accuracy": initial_accuracy,
            "initial_information_bits": original_information,
            "survival_times_initial_bits": survival * original_information,
            "mixture_accuracy": mixed_accuracy, "mixture_information_bits": 1 - binary_entropy(mixed_accuracy),
        },
        "martingale_direction_counterexample": {
            "regime_probability": 0.5,
            "regime_a": {"up_probability": 0.9, "up_increment": 1, "down_increment": -9, "conditional_mean_increment": 0.9 * 1 + 0.1 * -9},
            "regime_b": {"up_probability": 0.1, "up_increment": 9, "down_increment": -1, "conditional_mean_increment": 0.1 * 9 + 0.9 * -1},
            "direction_information_bits": original_information,
        },
        "two_timescale_poisson": {"weights": weights, "rho_components": rho_components, "rows": mixture_rows},
        "half_life_example": {
            "delays_events": [0, 1, 2, 4, 8], "raw_skill_bits": [0.4, 0.3, 0.32, 0.1, -0.02],
            **half_life(np.array([0, 1, 2, 4, 8]), np.array([0.4, 0.3, 0.32, 0.1, -0.02])),
        },
    }


def main() -> None:
    """Generate the complete JSON evidence, and fail loudly if a check fails."""
    started = time.perf_counter()
    sequence = np.random.SeedSequence(SEED)
    arrival_sequence, crossover_sequence = sequence.spawn(2)
    arrivals = check_arrival_models(np.random.default_rng(arrival_sequence))
    crossover = check_crossover(np.random.default_rng(crossover_sequence))
    lobster = measure_lobster()
    checks = {
        "poisson_pgf": arrivals["poisson_pgf"]["passed"],
        "hawkes_overdispersion": arrivals["overdispersion"]["passed"],
        "overhead_cancellation": arrivals["overhead"]["passed"],
        "crossover": crossover["passed"],
        "runtime_under_one_minute": time.perf_counter() - started < 60,
    }
    output = {
        "schema_version": 1, "seed": SEED, "rng": "numpy PCG64; SeedSequence.spawn for independent arrival/crossover streams",
        "rng_spawn_keys": {"arrivals": list(arrival_sequence.spawn_key), "crossover": list(crossover_sequence.spawn_key)},
        "command": ".venv/bin/python sims/theory_checks.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "polars": pl.__version__, "scipy": scipy_version,
        },
        "units": {"skill": "bits per prediction", "latency": "seconds unless explicitly microseconds", "event_delay": "events", "rate": "events per second"},
        "assumptions": [
            "PGF factorization assumes event-time skill independent of realized counts and anchor regime.",
            "Common-rho comparison uses positive initial skills and exponential event-time decay.",
            "Hawkes windows sample calendar time; market diagnostics sample merged event anchors.",
            "Hawkes prehistory is truncated with the recorded analytic omission bound.",
            "LOBSTER input prices are integer dollar-times-10000 units; types 4,5,6 are trades.",
            "The first book row initializes the event clock; identical nanosecond timestamps merge.",
            "A geometric waiting-time parameter is a plug-in model, not a fitted survival distribution.",
            "The measured rate denominator is the fixed session duration, not the first-to-last observed span.",
            "Supplied literature numbers remain unverified offline; arithmetic checks do not verify publications.",
        ],
        "arrivals": arrivals, "crossover": crossover, "lobster": lobster,
        "h1_tables": make_h1_tables(lobster), "analytic_examples": analytic_examples(),
        "supplied_literature": {
            "source": "docs/theory_spec.md section 7", "status": "unverified offline",
            "moallemi_saglam_year": 2013, "lucchese_pakkanen_veraart_year": 2024,
            "ait_sahalia_fan_xue_zhou": {"ticker": "INTC", "initial_r_squared": 0.14, "delayed_r_squared": 0.025, "feature_delay_seconds": 0.010, "prediction_horizon_seconds": 5.0},
        },
        "experiment_constants": {
            "overheads_microseconds": [1, 5, 40], "gate0_fast_latency_seconds": 100e-9,
            "gate0_slow_latency_seconds": 2e-3, "gate0_bootstrap_block_seconds": 300,
            "bootstrap_resamples": 1000, "bootstrap_seed": 0, "confidence_level": 0.95,
            "gate0_pass_minimum_tickers": 2, "gate0_ticker_count": len(TICKERS),
            "label_horizons_events": [10, 50, 100], "visible_book_levels": 10,
            "half_life_fraction": 0.5, "skill_floor_bits": 0,
            "h1_relative_margin": DELTA_1, "minimum_dummy_price_magnitude": 9999999999,
        },
        "checks": checks, "all_passed": all(checks.values()),
    }
    output["runtime_seconds"] = time.perf_counter() - started
    output["checks"]["runtime_under_one_minute"] = output["runtime_seconds"] < 60
    output["all_passed"] = all(output["checks"].values())
    path = ROOT / "results/theory/checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for name, passed in output["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"runtime_seconds={output['runtime_seconds']:.3f}; wrote {path.relative_to(ROOT)}")
    if not output["all_passed"]:
        raise SystemExit("One or more theory checks failed; inspect results/theory/checks.json")


if __name__ == "__main__":
    main()
