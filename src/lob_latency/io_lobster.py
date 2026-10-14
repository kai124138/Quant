"""Load paired LOBSTER rows; times are integer ns, prices are dollars * 10000."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class LobsterData:
    """Aligned messages and post-message books, with share quantities."""

    time_ns: np.ndarray
    event_type: np.ndarray
    direction: np.ndarray
    book: np.ndarray
    raw_rows: int
    dropped_halts: int
    sub_ns_timestamps_rounded: int = 0


def load_lobster(message_path: str | Path, book_path: str | Path) -> LobsterData:
    """Keep message types 1--6, dropping the corresponding halt book rows."""
    messages = pl.read_csv(message_path, has_header=False,
                           new_columns=["time", "type", "order_id", "size", "price", "direction"],
                           schema_overrides={"time": pl.String})
    books = pl.read_csv(book_path, has_header=False).to_numpy().astype(np.int64)
    if messages.width != 6 or books.shape != (messages.height, 40):
        raise ValueError("Expected aligned six-column messages and ten-level books")
    parts = messages["time"].str.split_exact(".", 1)
    seconds = parts.struct.field("field_0").cast(pl.Int64).to_numpy()
    fractions = parts.struct.field("field_1").fill_null("0")
    # Some sample rows have decimal serialization tails below one ns.
    # Round from decimal digits directly, without binary floating-point time.
    extra_precision = fractions.str.len_chars().to_numpy() > 9
    nanos = fractions.str.pad_end(9, "0").str.slice(0, 9).cast(pl.Int64).to_numpy()
    rounding_digit = fractions.str.slice(9, 1).replace("", "0").cast(pl.Int64).to_numpy()
    nanos = nanos + (rounding_digit >= 5)
    times = seconds * 1_000_000_000 + nanos
    if np.any(np.diff(times) < 0):
        raise ValueError("Messages must be chronological")
    types = messages["type"].to_numpy()
    if not np.isin(types, np.arange(1, 8)).all():
        raise ValueError("Unknown message type")
    keep = types != 7
    return LobsterData(times[keep], types[keep], messages["direction"].to_numpy()[keep],
                       books[keep], len(times), int(np.sum(~keep)), int(extra_precision.sum()))
