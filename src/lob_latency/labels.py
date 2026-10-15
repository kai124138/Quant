"""First future mid-price change labels and split-safe anchors, in events/ns."""

import os

import numpy as np

SPLITS = {"train": (35_100, 48_600), "validation": (48_600, 52_800),
          "test": (52_800, 57_000)}
# Delays in events. LOB_TAU_GRID (comma-separated) extends the grid for fast symbols
# whose millisecond latencies span more than 64 events; the default is the Gate 0 grid.
TAU_GRID = np.array([int(tau) for tau in
                     os.environ.get("LOB_TAU_GRID", "0,1,2,4,8,16,32,64").split(",")],
                    dtype=np.int64)


def next_move(mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return up=1/down=0 and first changed event; undefined entries are -1.

    Missing mids cannot be reference or terminal prices. They retain their
    event indices and are skipped when searching for the next valid change.
    """
    mid = np.asarray(mid)
    direction = np.full(len(mid), -1, dtype=np.int8)
    next_end = np.full(len(mid), -1, dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(mid))
    if not len(valid):
        return direction, next_end
    prices = mid[valid]
    starts = np.r_[0, np.flatnonzero(np.diff(prices) != 0) + 1]
    lengths = np.diff(np.r_[starts, len(valid)])
    ends = np.r_[valid[starts[1:]], -1]
    directions = np.r_[(prices[starts[1:]] > prices[starts[:-1]]).astype(np.int8), -1]
    direction[valid] = np.repeat(directions, lengths)
    next_end[valid] = np.repeat(ends, lengths)
    return direction, next_end


def delayed_labels(next_dir: np.ndarray, anchors: np.ndarray,
                   tau: int | np.ndarray) -> np.ndarray:
    """Label strictly after the action event anchor + tau, with no past target."""
    action = anchors + tau
    result = np.full(action.shape, -1, dtype=np.int8)
    valid = (action >= 0) & (action < len(next_dir))
    result[valid] = next_dir[action[valid]]
    return result


def select_anchors(time_ns: np.ndarray, next_end: np.ndarray,
                   feature_valid: np.ndarray, lookback: int = 50,
                   max_tau: int = int(TAU_GRID[-1]),
                   splits: dict = SPLITS) -> dict[str, np.ndarray]:
    """Require the input and max-delay label to stay within each time split.

    OFI's first difference needs the preceding book, so additionally keep
    event i-lookback in the split (one conservative warmup event).
    """
    i = np.arange(lookback, max(lookback, len(time_ns) - max_tau))
    end = next_end[i + max_tau]
    valid = feature_valid[i] & (end >= 0)
    safe_end = np.maximum(end, 0)
    return {name: i[valid & (time_ns[i - lookback] >= start * 1_000_000_000)
                       & (time_ns[safe_end] < stop * 1_000_000_000)]
            for name, (start, stop) in splits.items()}


def action_mask(time_ns: np.ndarray, next_end: np.ndarray, anchors: np.ndarray,
                tau: np.ndarray, split: str) -> np.ndarray:
    """True when a realized wall-clock action and its label stay in the split."""
    action = anchors + tau
    in_day = action < len(time_ns)
    end = np.full(len(anchors), -1, dtype=np.int64)
    end[in_day] = next_end[action[in_day]]
    start, stop = SPLITS[split]
    return (in_day & (end >= 0)
            & (time_ns[np.maximum(end, 0)] < stop * 1_000_000_000)
            & (time_ns[anchors] >= start * 1_000_000_000))
