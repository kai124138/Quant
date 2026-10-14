#!/usr/bin/env python3
"""Select symbols deterministically from training-day rates (events/second)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lob_latency.multiday import TRAIN_DAYS


def select_assets(frame: pl.DataFrame) -> dict:
    """Equal-day averages, per-day quality exclusions, lexical tie breaking.

    Tick class follows symbol_stats' thresholds applied to the averaged event
    spread (ticks). Crossed/locked shares are fractions of the trimmed session.
    Missing integrity evidence is excluded rather than assumed clean.
    """
    required = {"date", "symbol", "event_rate_hz", "mean_spread_ticks",
                "crossed_locked_session_share", "unknown_order_refs"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing statistics columns {sorted(missing)}; rerun scripts/symbol_stats.py")
    rows = frame.filter(pl.col("date").is_in(TRAIN_DAYS)).to_dicts()
    symbols = sorted({row["symbol"] for row in rows})
    excluded, eligible = {}, []
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,8}", symbol):
            raise ValueError(f"Invalid symbol {symbol!r}")
        daily = sorted([row for row in rows if row["symbol"] == symbol], key=lambda row: row["date"])
        reasons = []
        if [row["date"] for row in daily] != list(TRAIN_DAYS):
            reasons.append("missing_or_duplicate_training_day")
        for row in daily:
            values = [row[key] for key in required - {"date", "symbol"}]
            if any(v is None or not np.isfinite(v) for v in values):
                reasons.append("missing_or_nonfinite_quality_or_statistics")
                continue
            if row["crossed_locked_session_share"] > 0.01:
                reasons.append("crossed_locked_share_above_limit")
            if row["unknown_order_refs"] > 0:
                reasons.append("unknown_orders")
            if (row["event_rate_hz"] <= 0 or row["mean_spread_ticks"] <= 0
                    or not 0 <= row["crossed_locked_session_share"] <= 1
                    or row["unknown_order_refs"] < 0):
                reasons.append("invalid_statistics")
        if reasons:
            excluded[symbol] = {"reasons": sorted(set(reasons)), "training_rows": daily}
            continue
        numeric = [key for key, value in daily[0].items()
                   if key not in ("date", "symbol") and isinstance(value, (int, float))]
        averages = {key: float(np.mean([row[key] for row in daily])) for key in numeric
                    if all(row[key] is not None and np.isfinite(row[key]) for row in daily)}
        spread = averages["mean_spread_ticks"]
        tick_class = "large" if spread <= 1.5 else "small" if spread >= 3 else "medium"
        eligible.append({"symbol": symbol, "tick_size_class": tick_class,
                         "averages": averages, "training_rows": daily})
    classes = {kind: sorted([r for r in eligible if r["tick_size_class"] == kind],
                            key=lambda r: (-r["averages"]["event_rate_hz"], r["symbol"]))
               for kind in ("large", "small", "medium")}
    if len(classes["large"]) < 2 or not classes["small"]:
        raise ValueError("Selection requires two eligible large-tick symbols and one small-tick symbol")
    median = float(np.median([r["averages"]["event_rate_hz"] for r in classes["large"]]))
    fast = classes["large"][0]
    slow = next(r for r in classes["large"]
                if r["averages"]["event_rate_hz"] <= median and r["symbol"] != fast["symbol"])
    picks = [("large_fast", fast), ("large_at_or_below_median", slow), ("small_fast", classes["small"][0])]
    if classes["medium"]:
        picks.append(("medium_fast", classes["medium"][0]))
    return {"training_days": list(TRAIN_DAYS), "seed": None,
            "rule": {"large_max_mean_spread_ticks": 1.5, "small_min_mean_spread_ticks": 3.0,
                     "max_crossed_locked_session_share": 0.01, "max_unknown_order_refs": 0,
                     "averaging": "equal training-day arithmetic means", "tie_break": "symbol ascending",
                     "large_tick_median_event_rate_hz": median},
            "symbols": [row["symbol"] for _, row in picks],
            "picks": [{"role": role, **row} for role, row in picks],
            "eligible": eligible, "excluded": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=ROOT / "results/itch/symbol_stats.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/itch/selected_assets.json")
    args = parser.parse_args()
    try:
        selected = select_assets(pl.read_csv(args.stats))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Asset selection failed: {error}\n")
    selected.update(command=".venv/bin/python " + shlex.join(["scripts/select_assets.py", *sys.argv[1:]]),
                    stats_path=str(args.stats.resolve()),
                    stats_sha256=hashlib.sha256(args.stats.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
    print(f"Selected {', '.join(selected['symbols'])}; wrote {args.output}")


if __name__ == "__main__":
    main()
