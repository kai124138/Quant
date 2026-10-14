"""Convert wall-clock latency (integer ns) into event counts and grid weights."""

import numpy as np

LATENCY_NS = np.array([100, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000,
                       1_000_000, 2_000_000, 5_000_000, 10_000_000], dtype=np.int64)
LATENCY_NAMES = ["100 ns", "1 us", "10 us", "50 us", "100 us", "250 us", "500 us",
                 "1 ms", "2 ms", "5 ms", "10 ms"]


def event_delays(time_ns: np.ndarray, anchors: np.ndarray, latency_ns: int) -> np.ndarray:
    """Count events in (t_i, t_i + L], including the right endpoint and no ties at t_i."""
    if latency_ns < 0:
        raise ValueError("Latency must be nonnegative")
    times = time_ns[anchors]
    return (np.searchsorted(time_ns, times + latency_ns, side="right")
            - np.searchsorted(time_ns, times, side="right"))


def interpolation_weights(tau: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Weights for linear S interpolation, held constant past the final event delay."""
    tau = np.asarray(tau)
    clipped = np.clip(tau, grid[0], grid[-1])
    high = np.clip(np.searchsorted(grid, clipped, side="right"), 1, len(grid) - 1)
    low = high - 1
    fraction = (clipped - grid[low]) / (grid[high] - grid[low])
    result = np.zeros((len(tau), len(grid)))
    result[np.arange(len(tau)), low] = 1 - fraction
    result[np.arange(len(tau)), high] += fraction
    return result


def delay_statistics(tau: np.ndarray) -> dict:
    """Empirical probabilities and event-count quantiles on the given anchors."""
    values, counts = np.unique(tau, return_counts=True)
    return {"p_at_least_one": float(np.mean(tau >= 1)), "mean_events": float(np.mean(tau)),
            "percentiles_events": dict(zip(["50", "90", "99"],
                                            np.percentile(tau, [50, 90, 99]).tolist())),
            "p_beyond_64": float(np.mean(tau > 64)),
            "pmf": {str(v): float(c / len(tau)) for v, c in zip(values, counts)}}
