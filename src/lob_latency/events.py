"""Merge message timestamps into events; integer ns and share quantities."""

from dataclasses import dataclass

import numpy as np

from lob_latency.io_lobster import LobsterData

DUMMY_PRICE = 9_999_999_999


@dataclass
class Events:
    """One final book per timestamp, with the last trade's aggressor sign."""

    time_ns: np.ndarray
    book: np.ndarray
    message_count: np.ndarray
    execution: np.ndarray
    trade_sign: np.ndarray

    @property
    def valid_mid(self) -> np.ndarray:
        return (np.abs(self.book[:, 0]) != DUMMY_PRICE) & (
            np.abs(self.book[:, 2]) != DUMMY_PRICE)

    @property
    def mid(self) -> np.ndarray:
        """Mid-price in dollars; missing best sides yield NaN."""
        mid = (self.book[:, 0] + self.book[:, 2]) / 20_000.0
        return np.where(self.valid_mid, mid, np.nan)


def merge_events(data: LobsterData) -> Events:
    """Merge consecutive equal ns timestamps and use their last book row."""
    if len(data.time_ns) == 0:
        raise ValueError("No retained messages")
    starts = np.r_[0, np.flatnonzero(np.diff(data.time_ns)) + 1]
    ends = np.r_[starts[1:], len(data.time_ns)] - 1
    execution = np.isin(data.event_type, [4, 5, 6])
    last_trade = np.maximum.reduceat(np.where(execution, np.arange(len(execution)), -1),
                                    starts)
    trade_sign = np.zeros(len(starts), dtype=np.int8)
    has_trade = last_trade >= 0
    trade_sign[has_trade] = -data.direction[last_trade[has_trade]]
    return Events(data.time_ns[starts], data.book[ends], ends - starts + 1,
                  np.logical_or.reduceat(execution, starts), trade_sign)
