"""Causal book features: share OFI, tick spread, and log seconds between events."""

import numpy as np

from lob_latency.events import DUMMY_PRICE, Events

FEATURE_NAMES = ["qi", "depth_3", "depth_5", "ofi_1", "ofi_5", "ofi_10",
                 "ofi_50", "spread_ticks", "log_dt_seconds", "last_trade_10"]
QI_COLUMNS = [0, 7]


def order_flow_imbalance(book: np.ndarray) -> np.ndarray:
    """Cont et al. best-level OFI per event in shares; first event is zero."""
    ask, qa, bid, qb = book[:, :4].T
    result = np.zeros(len(book), dtype=float)
    result[1:] = ((bid[1:] >= bid[:-1]) * qb[1:]
                  - (bid[1:] <= bid[:-1]) * qb[:-1]
                  - (ask[1:] <= ask[:-1]) * qa[1:]
                  + (ask[1:] >= ask[:-1]) * qa[:-1])
    invalid = (np.abs(ask) == DUMMY_PRICE) | (np.abs(bid) == DUMMY_PRICE)
    result[invalid | np.r_[False, invalid[:-1]]] = np.nan
    return result


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum over window events, requiring a complete finite window."""
    finite = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    counts = np.r_[0, np.cumsum(finite)]
    right = np.arange(1, len(values) + 1)
    left = np.maximum(0, right - window)
    result = sums[right] - sums[left]
    result[(right - left < window) | (counts[right] - counts[left] < window)] = np.nan
    return result


def make_features(events: Events) -> np.ndarray:
    """Use only events <= i, including at most 50 OFI increments at anchor i."""
    book = events.book
    columns = []
    for depth in (1, 3, 5):
        ask = book[:, 1:4 * depth:4].sum(axis=1)
        bid = book[:, 3:4 * depth:4].sum(axis=1)
        columns.append(np.divide(bid - ask, bid + ask,
                                 out=np.full(len(book), np.nan), where=(bid + ask) > 0))
    ofi = order_flow_imbalance(book)
    columns.extend(rolling_sum(ofi, k) for k in (1, 5, 10, 50))
    columns.append((book[:, 0] - book[:, 2]) / 100.0)
    dt = np.r_[np.nan, np.diff(events.time_ns) / 1_000_000_000.0]
    columns.append(np.log(dt, where=dt > 0, out=np.full(len(dt), np.nan)))
    last = np.maximum.accumulate(np.where(events.execution, np.arange(len(book)), -1))
    recent = (last >= 0) & (np.arange(len(book)) - last < 10)
    columns.append(np.where(recent, events.trade_sign[np.maximum(last, 0)], 0))
    features = np.column_stack(columns)
    features[~events.valid_mid] = np.nan
    return features


def standardize(features: np.ndarray, train_anchors: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit means/scales on training anchors only; constant columns use scale 1."""
    mean = features[train_anchors].mean(axis=0)
    scale = features[train_anchors].std(axis=0)
    scale[scale == 0] = 1.0
    return (features - mean) / scale, mean, scale
