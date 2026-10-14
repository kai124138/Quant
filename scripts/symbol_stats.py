#!/usr/bin/env python3
"""Summarize converted symbol-days on the Gate 0 event clock (seconds, ticks)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")

import numpy as np
import polars as pl

from lob_latency.events import merge_events
from lob_latency.io_lobster import LobsterData, load_lobster


def symbol_day_statistics(data: LobsterData, symbol: str, day: str, *,
                          start_seconds: int = 35_100, end_seconds: int = 57_000,
                          observation_end_ns: int | None = None,
                          integrity: dict | None = None) -> dict:
    """Measure event averages and observed-time one-tick share, using one-cent ticks.

    Apply the research session trim before counting events; merge equal ns times
    exactly as Gate 0. Spread and mid summaries exclude missing/crossed/locked
    books. Time weights carry each final book only to the next observed event
    or the known observation end, clipped to the trimmed interval.
    """
    if not 34_200 <= start_seconds < end_seconds <= 57_600:
        raise ValueError("Statistics window must be inside the regular session")
    start, stop = start_seconds * 10**9, end_seconds * 10**9
    selected = (data.time_ns >= start) & (data.time_ns < stop)
    result = {"date": day, "symbol": symbol, "raw_messages": data.raw_rows,
              "messages": int(selected.sum()), "dropped_halts": data.dropped_halts,
              "merged_events": 0, "valid_book_events": 0,
              "mean_inter_event_gap_seconds": None, "median_inter_event_gap_seconds": None,
              "mid_changes_per_event": None, "mean_spread_ticks": None,
              "one_tick_time_share": None, "mean_mid_price_dollars": None,
              "observed_book_seconds": 0.0, "tick_size_class": "unclassified",
              "event_rate_hz": 0.0, "crossed_locked_session_share": 0.0,
              "unknown_order_refs": None if integrity is None else integrity.get("unknown_order_refs")}
    if not len(data.time_ns):
        return result
    events = merge_events(data)
    inside = (events.time_ns >= start) & (events.time_ns < stop)
    n_events = int(inside.sum())
    result["merged_events"] = n_events
    result["event_rate_hz"] = n_events / (end_seconds - start_seconds)
    gaps = np.diff(events.time_ns[inside]) / 10**9
    if len(gaps):
        result["mean_inter_event_gap_seconds"] = float(gaps.mean())
        result["median_inter_event_gap_seconds"] = float(np.median(gaps))
    spread = (events.book[:, 0] - events.book[:, 2]) / 100.0
    valid = events.valid_mid & (spread > 0)
    valid_inside = valid & inside
    result["valid_book_events"] = int(valid_inside.sum())
    if valid_inside.any():
        mean_spread = float(spread[valid_inside].mean())
        result["mean_spread_ticks"] = mean_spread
        result["mean_mid_price_dollars"] = float(events.mid[valid_inside].mean())
        result["tick_size_class"] = ("large" if mean_spread <= 1.5 else
                                      "small" if mean_spread >= 3 else "medium")
    if n_events:
        selected_mid = events.mid[inside]
        selected_valid = valid[inside]
        changes = ((np.diff(selected_mid) != 0) & selected_valid[1:] & selected_valid[:-1])
        # The first trimmed event has no within-window predecessor.
        result["mid_changes_per_event"] = float(changes.sum() / n_events)
    observed_stop = int(events.time_ns[-1]) if observation_end_ns is None else observation_end_ns
    if observed_stop < events.time_ns[-1]:
        raise ValueError("Observation end precedes last event")
    interval_ends = np.r_[events.time_ns[1:], observed_stop]
    durations = np.maximum(0, np.minimum(interval_ends, stop) - np.maximum(events.time_ns, start))
    result["crossed_locked_session_share"] = float(
        durations[events.valid_mid & (spread <= 0)].sum() / (stop - start))
    weights = np.where(valid, durations, 0)
    total = int(weights.sum())
    result["observed_book_seconds"] = total / 10**9
    if total:
        result["one_tick_time_share"] = float(weights[spread == 1].sum() / total)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                        help="converted directories or individual message Parquet files")
    parser.add_argument("--out", type=Path, default=ROOT / "results/itch")
    args = parser.parse_args()
    paths = set()
    for source in args.input:
        if source.is_dir():
            paths.update(source.rglob("*_message_10.parquet"))
        elif source.is_file() and source.name.endswith("_message_10.parquet"):
            paths.add(source)
        else:
            parser.error(f"No converted input at {source}")
    if not paths:
        parser.error("No converted symbol-days found")
    rows = []
    for message in sorted(paths):
        symbol = message.name.removesuffix("_message_10.parquet")
        book = message.with_name(f"{symbol}_orderbook_10.parquet")
        manifest_path = message.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        integrity = manifest.get("integrity", {}).get(symbol)
        integrity_path = message.with_name(f"{symbol}_integrity.json")
        if integrity is None and integrity_path.exists():
            integrity = json.loads(integrity_path.read_text())
        data = load_lobster(message, book)
        rows.append(symbol_day_statistics(data, symbol, message.parent.name,
                    observation_end_ns=manifest.get("filter", {}).get("counts", {}).get("last_timestamp_ns"),
                    integrity=integrity))
    args.out.mkdir(parents=True, exist_ok=True)
    # Provide explicit types even when every sample-day metric is undefined.
    schema = {key: (pl.String if isinstance(value, str) else
                    pl.Int64 if isinstance(value, int) else pl.Float64)
              for key, value in rows[0].items()}
    frame = pl.DataFrame(rows, schema=schema)
    frame.write_csv(args.out / "symbol_stats.csv")
    metadata = {"command": "MPLCONFIGDIR=.cache/mpl " + shlex.join(sys.argv),
                "input_paths": [str(path.resolve()) for path in sorted(paths)],
                "symbol_days": len(rows), "seed": None,
                "start_seconds": 35_100, "end_seconds_exclusive": 57_000,
                "tick_dollars": 0.01, "large_max_spread_ticks": 1.5,
              "small_min_spread_ticks": 3.0,
              "integrity_definition": "Crossed/locked time divided by trimmed session seconds; unknown refs from reconstruction",
                "source_venue": "Nasdaq only; not consolidated NBBO"}
    (args.out / "symbol_stats_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    columns = list(schema)
    lines = ["# Converted symbol-day statistics", "",
             "Source: symbol_stats.csv and symbol_stats_metadata.json; undefined metrics are null.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("null" if row[k] is None else str(row[k]) for k in columns) + " |")
    lines += ["", "## Assumptions", "",
              "- The research session trim in the metadata applies before event counts and gap summaries.",
              "- Equal timestamps merge using the final book and the Gate 0 loader drops halt rows.",
              "- Tick size is the fixed one-cent convention recorded in the metadata; classes use event-mean spread.",
              "- Missing, locked and crossed books do not contribute to spread or mid averages or time weights.",
              "- Mid changes count adjacent valid event pairs divided by all trimmed events; the first has no predecessor.",
              "- Time shares use observed intervals, including a carried book at the trim boundary, without extending beyond coverage.",
              "- No random sampling is used; the recorded seed is null."]
    (args.out / "symbol_stats.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} symbol-days to {args.out / 'symbol_stats.csv'}", flush=True)


if __name__ == "__main__":
    main()
