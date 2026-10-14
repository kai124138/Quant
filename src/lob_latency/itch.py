"""Replay ITCH records into bounded batches of post-message ten-level books.

Times are integer ns internally and decimal seconds in the message Parquet;
prices are dollars * 10000 and sizes are shares. The filter wire dtype is
little endian, unaligned, exactly 40 bytes: offsets 0/8/16/24/28/32/33/34/35.
Type, side and printable contain ASCII byte values; unused fields are zero.
"""

from bisect import bisect_left, insort
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_latency.events import DUMMY_PRICE

RECORD_DTYPE = np.dtype([
    ("ts_ns", "<u8"), ("ref", "<u8"), ("new_ref", "<u8"),
    ("shares", "<u4"), ("price", "<u4"), ("type", "u1"),
    ("side", "u1"), ("printable", "u1"), ("pad", "u1", (5,)),
])
SESSION_START_NS = 34_200_000_000_000
SESSION_END_NS = 57_600_000_000_000
MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]
BOOK_COLUMNS = [f"{field}_{level}" for level in range(1, 11)
                for field in ("ask_price", "ask_size", "bid_price", "bid_size")]
MESSAGE_SCHEMA = pa.schema([
    ("time", pa.decimal128(14, 9)), ("type", pa.int8()),
    ("order_id", pa.uint64()), ("size", pa.uint64()),
    ("price", pa.int64()), ("direction", pa.int8()),
])
BOOK_SCHEMA = pa.schema([(name, pa.int64()) for name in BOOK_COLUMNS])


def load_records(path: str | Path) -> np.ndarray:
    """Memory-map fixed records (ns, shares, price units); reject partial records."""
    path = Path(path)
    if path.stat().st_size % RECORD_DTYPE.itemsize:
        raise ValueError("Partial 40-byte filter record")
    if not path.stat().st_size:
        return np.empty(0, dtype=RECORD_DTYPE)
    return np.memmap(path, dtype=RECORD_DTYPE, mode="r")


class OrderBook:
    """Visible orders and sorted price levels, with integer share quantities."""

    def __init__(self):
        self.orders = {}
        self.levels = {1: {}, -1: {}}
        self.prices = {1: [], -1: []}

    def adjust(self, side: int, price: int, delta: int) -> None:
        """Adjust one level by delta shares and remove empty levels."""
        levels, prices = self.levels[side], self.prices[side]
        old = levels.get(price, 0)
        new = old + delta
        if new < 0:
            raise ValueError("Negative aggregate shares")
        if new == 0:
            if old:
                del levels[price]
                prices.pop(bisect_left(prices, price))
        else:
            if not old:
                insort(prices, price)
            levels[price] = new

    def add(self, ref: int, side: int, price: int, shares: int) -> None:
        """Add a nonempty visible order in shares."""
        if side not in (-1, 1):
            raise ValueError("Add order must have B/S side")
        if shares:
            self.orders[ref] = (side, price, shares)
            self.adjust(side, price, shares)

    def remove(self, ref: int, shares: int) -> tuple[int, int, int]:
        """Remove at most the remaining shares; return side, price, removed shares."""
        side, price, remaining = self.orders[ref]
        removed = min(shares, remaining)
        self.adjust(side, price, -removed)
        if remaining == removed:
            del self.orders[ref]
        else:
            self.orders[ref] = (side, price, remaining - removed)
        return side, price, removed

    def snapshot(self) -> tuple[int, ...]:
        """Return LOBSTER ask/bid price-size columns, with signed dummy prices."""
        asks, bids = self.prices[-1][:10], self.prices[1][-10:][::-1]
        rows = []
        for i in range(10):
            ask = asks[i] if i < len(asks) else DUMMY_PRICE
            bid = bids[i] if i < len(bids) else -DUMMY_PRICE
            rows.extend((ask, self.levels[-1].get(ask, 0),
                         bid, self.levels[1].get(bid, 0)))
        return tuple(rows)

    def market_state(self) -> tuple[bool, bool]:
        """Return crossed and locked flags; a missing side is neither."""
        if not self.prices[1] or not self.prices[-1]:
            return False, False
        bid, ask = self.prices[1][-1], self.prices[-1][0]
        return bid > ask, bid == ask


class StateCounter:
    """Count post-input-message states and contiguous observed durations in ns."""

    def __init__(self):
        self.count = 0
        self.intervals = 0
        self.longest = 0
        self.total = 0
        self.start = None

    def observe(self, ts: int, active: bool) -> None:
        if SESSION_START_NS <= ts <= SESSION_END_NS and active:
            self.count += 1
        if active and self.start is None:
            self.start = max(ts, SESSION_START_NS)
        elif not active:
            self.close(ts)

    def close(self, ts: int) -> None:
        if self.start is not None:
            stop = min(ts, SESSION_END_NS)
            duration = max(0, stop - self.start)
            # A state observed at the close has a counted zero-duration interval.
            if stop >= self.start:
                self.intervals += 1
            self.longest = max(self.longest, duration)
            self.total += duration
            self.start = None

    def result(self) -> dict:
        return {"count": self.count, "intervals": self.intervals,
                "longest_duration_ns": self.longest, "total_duration_ns": self.total}


class ParquetRows:
    """Write aligned post-message rows in bounded batches; times are exact seconds."""

    def __init__(self, message_path: Path, book_path: Path, batch_rows: int):
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self.batch_rows = batch_rows
        self.messages, self.books = [], []
        self.message_writer = pq.ParquetWriter(message_path, MESSAGE_SCHEMA)
        self.book_writer = pq.ParquetWriter(book_path, BOOK_SCHEMA)

    def append(self, row: tuple, snapshot: tuple) -> None:
        ts, *fields = row
        # Construct from decimal digits, avoiding float conversion and context rounding.
        self.messages.append((Decimal(f"{ts // 10**9}.{ts % 10**9:09d}"), *fields))
        self.books.append(snapshot)
        if len(self.messages) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.messages:
            return
        for writer, schema, rows in ((self.message_writer, MESSAGE_SCHEMA, self.messages),
                                     (self.book_writer, BOOK_SCHEMA, self.books)):
            table = pa.Table.from_arrays(
                [pa.array(column, type=field.type)
                 for field, column in zip(schema, zip(*rows))], schema=schema)
            writer.write_table(table)
        self.messages.clear()
        self.books.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.message_writer.close()
            self.book_writer.close()


def reconstruct(bin_path: str | Path, out_dir: str | Path, symbol: str,
                *, batch_rows: int = 50_000,
                observation_end_ns: int | None = None) -> dict:
    """Replay the complete available prefix; emit session rows and integrity JSON.

    Unknown updates are skipped except executions, whose missing side/price use
    zero. Over-removals are counted and clipped to preserve nonnegative depth.
    P and Q directions are zero: modern ITCH does not supply their resting side.
    H rows use zero placeholders; the fixed record does not retain trading state.
    """
    started = time.perf_counter()
    records = load_records(bin_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = OrderBook()
    counts = Counter()
    unknown = Counter()
    negative = duplicates = 0
    state_counts = {name: StateCounter() for name in ("crossed", "locked", "crossed_or_locked")}
    writer = ParquetRows(out_dir / f"{symbol}_message_10.parquet",
                         out_dir / f"{symbol}_orderbook_10.parquet", batch_rows)
    before = book.snapshot()
    previous_ts = None

    def emit(ts, kind, ref, shares, price, side):
        nonlocal before
        snapshot = book.snapshot()
        if SESSION_START_NS <= ts <= SESSION_END_NS and (snapshot != before or kind >= 4):
            writer.append((ts, kind, ref, shares, price, side), snapshot)
            counts[str(kind)] += 1
        before = snapshot

    try:
        for record in records:
            ts, ref, new_ref, shares, price, kind, side, _, _ = record.tolist()
            if previous_ts is not None and ts < previous_ts:
                raise ValueError("ITCH records must be chronological")
            previous_ts = ts
            direction = {ord('B'): 1, ord('S'): -1}.get(side, 0)
            if kind in (ord('A'), ord('F')):
                if ref in book.orders:
                    duplicates += 1
                else:
                    book.add(ref, direction, price, shares)
                    emit(ts, 1, ref, shares, price, direction)
            elif kind in (ord('E'), ord('C'), ord('X'), ord('D'), ord('U')):
                old = book.orders.get(ref)
                if old is None:
                    unknown[chr(kind)] += 1
                    if kind in (ord('E'), ord('C')):
                        emit(ts, 4, ref, shares, price, 0)
                else:
                    direction, old_price, remaining = old
                    if kind == ord('U') and new_ref in book.orders and new_ref != ref:
                        duplicates += 1
                    else:
                        removed = remaining if kind in (ord('D'), ord('U')) else shares
                        negative += int(removed > remaining)
                        book.remove(ref, removed)
                        if kind == ord('U'):
                            emit(ts, 3, ref, remaining, old_price, direction)
                            book.add(new_ref, direction, price, shares)
                            emit(ts, 1, new_ref, shares, price, direction)
                        else:
                            mapped = {ord('E'): 4, ord('C'): 4, ord('X'): 2, ord('D'): 3}[kind]
                            emit(ts, mapped, ref, removed,
                                 price if kind == ord('C') else old_price, direction)
            elif kind == ord('P'):
                emit(ts, 5, ref, shares, price, 0)
            elif kind == ord('Q'):
                emit(ts, 6, 0, shares, price, 0)
            elif kind == ord('H'):
                emit(ts, 7, 0, 0, 0, 0)
            else:
                raise ValueError(f"Unknown filtered record type {kind}")
            crossed, locked = book.market_state()
            for name, active in zip(state_counts, (crossed, locked, crossed or locked)):
                state_counts[name].observe(ts, active)
    finally:
        writer.close()
    end_ns = observation_end_ns if observation_end_ns is not None else previous_ts
    if end_ns is not None:
        if previous_ts is not None and end_ns < previous_ts:
            raise ValueError("Observation end precedes last symbol record")
        for counter in state_counts.values():
            counter.close(end_ns)
    elapsed = time.perf_counter() - started
    result = {
        "symbol": symbol, "input_records": len(records),
        "emitted_rows": sum(counts.values()), "lobster_counts_by_type": dict(sorted(counts.items())),
        "first_timestamp_ns": int(records[0]['ts_ns']) if len(records) else None,
        "last_timestamp_ns": previous_ts, "observation_end_ns": end_ns,
        "unknown_order_refs": sum(unknown.values()), "unknown_order_refs_by_type": dict(unknown),
        "negative_shares": negative, "duplicate_order_refs": duplicates,
        **{name: counter.result() for name, counter in state_counts.items()},
        "final_open_order_count": len(book.orders), "elapsed_seconds": elapsed,
        "messages_per_second": len(records) / elapsed if elapsed else None,
        "configuration": {"seed": None, "batch_rows": batch_rows,
                          "record_bytes": RECORD_DTYPE.itemsize, "book_levels": 10,
                          "session_start_ns": SESSION_START_NS, "session_end_ns": SESSION_END_NS},
    }
    (out_dir / f"{symbol}_integrity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
