"""Whole-day preparation and held-out uncertainty in events, ns, and bits."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import isotonic_regression

from lob_latency.events import Events, merge_events
from lob_latency.features import make_features, standardize
from lob_latency.io_lobster import LobsterData
from lob_latency.labels import TAU_GRID, delayed_labels, next_move, select_anchors
from lob_latency.latency import LATENCY_NS, event_delays

TRAIN_DAYS = ("2025-12-08", "2025-12-09", "2025-12-10")
VALIDATION_DAY = "2025-12-11"
TEST_DAYS = ("2025-12-12", "2026-06-12")
ALL_DAYS = (*TRAIN_DAYS, VALIDATION_DAY, *TEST_DAYS)
SESSION_SECONDS = (35_100, 57_000)


@dataclass
class PreparedDay:
    """Day-local indices/timestamps (ns), anchor features, and binary labels."""

    time_ns: np.ndarray
    next_dir: np.ndarray
    next_end: np.ndarray
    anchors: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    metadata: dict

    def delays(self, latency_ns: int) -> np.ndarray:
        """Count events only in this day's trimmed clock."""
        return event_delays(self.time_ns, self.anchors, latency_ns)

    def direct_target(self, latency_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Realized action-time labels; censor missing within-session endpoints."""
        tau = self.delays(latency_ns)
        action = self.anchors + tau
        labels = delayed_labels(self.next_dir, self.anchors, tau)
        mask = (labels >= 0) & (self.next_end[action] >= 0)
        return labels, mask


def prepare_day(data: LobsterData, max_anchors: int | None = None) -> PreparedDay:
    """Trim first, then construct every rolling window/label/count within a day.

    Include Gate 0's preceding-book guard for the 50-event OFI window. An
    optional deterministic, evenly spaced anchor subset is for fixture runs.
    """
    merged = merge_events(data)
    start, stop = np.array(SESSION_SECONDS, dtype=np.int64) * 10**9
    inside = (merged.time_ns >= start) & (merged.time_ns < stop)
    events = Events(*(getattr(merged, field)[inside] for field in (
        "time_ns", "book", "message_count", "execution", "trade_sign")))
    if not len(events.time_ns):
        raise ValueError("No events in trimmed session")
    direction, end = next_move(events.mid)
    features = make_features(events)
    anchors = select_anchors(events.time_ns, end, np.isfinite(features).all(axis=1),
                             splits={"day": SESSION_SECONDS})["day"]
    candidate_count = len(anchors)
    labels = np.column_stack([delayed_labels(direction, anchors, tau) for tau in TAU_GRID])
    # Avoid partial wall-clock windows at the observed end, including truncated
    # inputs. Labels/features themselves were already constructed day locally.
    keep = np.all(labels >= 0, axis=1) & (
        events.time_ns[anchors] + LATENCY_NS[-1] <= events.time_ns[-1])
    anchors, labels = anchors[keep], labels[keep]
    eligible_count = len(anchors)
    if max_anchors is not None:
        if max_anchors < 2:
            raise ValueError("max_anchors must be at least two")
        pick = np.linspace(0, len(anchors) - 1, min(max_anchors, len(anchors)), dtype=int)
        anchors, labels = anchors[pick], labels[pick]
    if not len(anchors):
        raise ValueError("No common valid anchors in trimmed session")
    metadata = {
        "raw_messages": data.raw_rows, "dropped_halts": data.dropped_halts,
        "sub_ns_timestamps_rounded": data.sub_ns_timestamps_rounded,
        "full_day_merged_events": len(merged.time_ns), "trimmed_events": len(events.time_ns),
        "invalid_mid_events": int((~events.valid_mid).sum()),
        "execution_events": int(events.execution.sum()),
        "before_common_label_filter": candidate_count, "eligible_anchors": eligible_count,
        "count": len(anchors), "first_time_ns": int(events.time_ns[anchors[0]]),
        "last_time_ns": int(events.time_ns[anchors[-1]]),
        "session_seconds": list(SESSION_SECONDS),
    }
    return PreparedDay(events.time_ns, direction, end, anchors, features[anchors], labels, metadata)


def split_arrays(days: dict[str, PreparedDay]) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Pool training anchor statistics; freeze scales for validation/test days."""
    training = np.concatenate([days[day].features for day in TRAIN_DAYS])
    x_train, mean, scale = standardize(training, np.arange(len(training)))
    xs = {"train": x_train, "validation": (days[VALIDATION_DAY].features - mean) / scale}
    ys = {"train": np.concatenate([days[day].labels for day in TRAIN_DAYS]),
          "validation": days[VALIDATION_DAY].labels}
    for day in TEST_DAYS:
        xs[day] = (days[day].features - mean) / scale
        ys[day] = days[day].labels
    return xs, ys, mean, scale


def half_life(skill: np.ndarray, grid: np.ndarray = TAU_GRID) -> dict:
    """Equal-weight decreasing isotonic surrogate and first half crossing (events)."""
    if (len(grid) != len(skill) or len(grid) < 2 or grid[0] != 0
            or np.any(np.diff(grid) <= 0) or not np.isfinite(skill).all()):
        raise ValueError("Half-life requires finite skills on an increasing grid from zero")
    fit = np.maximum(isotonic_regression(skill, increasing=False).x, 0)
    result = {"fit_bits": fit.tolist(), "half_life_events": None}
    if fit[0] <= 0:
        return {**result, "status": "no_positive_initial_skill"}
    crossing = np.flatnonzero(fit <= fit[0] / 2)
    if not len(crossing):
        return {**result, "status": "right_censored", "lower_bound_events": float(grid[-1])}
    j = int(crossing[0])
    fraction = (fit[j - 1] - fit[0] / 2) / (fit[j - 1] - fit[j])
    return {**result, "status": "observed",
            "half_life_events": float(grid[j - 1] + fraction * (grid[j] - grid[j - 1]))}


def half_life_summary(point: np.ndarray, boot: np.ndarray, day: PreparedDay) -> dict:
    """Bootstrap crossing with censored mass retained; elapsed time in seconds.

    Right-censored replicates enter quantiles as infinity, serialized as null
    with explicit unbounded flags. Nonpositive initial skill is unidentified;
    report its mass separately and condition crossing quantiles on positivity.
    Fractional event indices use linear interpolation of event timestamps.
    """
    result = half_life(point)
    fitted = [half_life(curve) for curve in boot]
    censored = sum(r["status"] == "right_censored" for r in fitted)
    unidentified = sum(r["status"] == "no_positive_initial_skill" for r in fitted)
    samples = np.array([np.inf if r["status"] == "right_censored" else r["half_life_events"]
                        for r in fitted if r["status"] != "no_positive_initial_skill"])
    if len(samples):
        # Inverted-CDF percentiles preserve censoring, including an infinite
        # upper endpoint, without infinity-minus-infinity interpolation.
        bounds = np.quantile(samples, [0.025, 0.975], method="inverted_cdf")
        result["ci95_events"] = [float(v) if np.isfinite(v) else None for v in bounds]
        result["ci95_unbounded"] = [bool(np.isinf(v)) for v in bounds]
    else:
        result.update(ci95_events=None, ci95_unbounded=None)
    result.update(bootstrap_right_censored_fraction=censored / len(boot),
                  bootstrap_no_positive_initial_fraction=unidentified / len(boot),
                  bootstrap_positive_initial_resamples=len(samples),
                  ci_condition="positive initial fitted skill; censored crossings retained",
                  wall_clock_seconds=None)
    tau = result["half_life_events"]
    if tau is not None:
        low = int(np.floor(tau))
        fraction = tau - low
        elapsed = (day.time_ns[day.anchors + low] - day.time_ns[day.anchors]).astype(float)
        if fraction:
            elapsed += fraction * (day.time_ns[day.anchors + low + 1] - day.time_ns[day.anchors + low])
        q25, median, q75 = np.percentile(elapsed / 10**9, [25, 50, 75])
        result["wall_clock_seconds"] = {"median": float(median), "q25": float(q25),
                                        "q75": float(q75), "iqr": float(q75 - q25)}
    return result
