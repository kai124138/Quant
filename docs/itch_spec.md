# ITCH conversion spec: Nasdaq TotalView-ITCH 5.0 -> LOBSTER-format files

Goal: turn full-day ITCH 5.0 files into per-symbol LOBSTER-format message and 10-level
order book data, so the Gate 0 code in `src/lob_latency/` runs on 2025-2026 data unchanged.
Reference: `data/ref/itch50spec.pdf`. Test data: `data/raw/itch_head/S120825_head64MB.gz`
(first 64 MB of the 2025-12-08 file; a truncated gzip stream, so decompression ends with an
error after the last complete block, which the tools must tolerate).

The full days (6 files, 8-18 GB compressed each) live on another machine and are converted
there later; everything here must run from a single CLI command on Linux and macOS.

## 1. `tools/itch_filter.c` (fast pass, C99, no dependencies)
- Read raw ITCH 5.0 from stdin (the caller pipes `gzip -dc`). Each message is a 2-byte
  big-endian length followed by the message. All integer fields are big-endian.
- Args: `itch_filter <out_dir> <SYM1,SYM2,...>`. Map stock locate -> symbol from 'R'
  (Stock Directory) messages; symbols are 8-byte space-padded.
- For requested symbols keep types A, F, E, C, X, D, U, P, Q and H (trading action).
  Write one binary file per symbol, `<out_dir>/<SYM>.bin`, of fixed 40-byte little-endian
  records: `uint64 ts_ns; uint64 ref; uint64 new_ref; uint32 shares; uint32 price;
  uint8 type; uint8 side; uint8 printable; uint8 pad[5]` (document the matching numpy
  dtype in `src/lob_latency/itch.py`). Unused fields are 0.
- Also write `<out_dir>/filter_counts.json`: message counts by type (all symbols) and by
  type per requested symbol, bytes read, and whether input ended mid-message.
- Must stop cleanly at EOF or a truncated final message.

## 2. `src/lob_latency/itch.py` (book reconstruction)
- Load a `.bin` with numpy; replay in order with a dict order_ref -> (side, price, shares)
  and per-side dicts price -> total shares; derive the top 10 levels per side after each
  message (bisect on sorted price lists is fine).
- Message mapping to LOBSTER types: A/F -> 1 submission; X -> 2 partial cancel (a cancel
  that removes all remaining shares is still type 2 unless followed by D); D -> 3 deletion;
  E and C -> 4 visible execution (C at its execution price); P -> 5 hidden execution (no
  book change); Q -> 6 cross trade; H halt -> 7. U (replace) -> a type 3 row for the old
  ref plus a type 1 row for the new ref, same timestamp, same side.
- Direction: +1 buy, -1 sell, of the order on the book (for executions, the resting
  order's side, as in LOBSTER). Price in dollars x 10000 (ITCH prices already have 4
  implied decimals).
- Build the book from the first message of the day; emit rows only for 09:30:00.000 to
  16:00:00.000 (34200-57600 s). Emit a row when the message changes the top-10 snapshot or
  is type 4, 5, 6 or 7 (this matches LOBSTER's "events in the requested price range").
- Output per symbol and day, matching LOBSTER column order and dummy values: message
  table (time s with ns precision, type, order id, size, price, direction) and order book
  table (ask price 1, ask size 1, bid price 1, bid size 1, ... level 10; empty levels
  +/-9999999999 with size 0). Write Parquet; extend `io_lobster.py` so the existing loader
  accepts either LOBSTER CSV or these Parquet files with identical results.
- Integrity counters in a per-symbol JSON: unknown order refs, negative shares, crossed or
  locked books during 09:30-16:00 (count and longest duration), final open-order count.

## 3. `scripts/itch_to_lobster.py`
- `--input <file.gz> --symbols <list> --out <dir> [--keep-bin]`: compiles the filter if
  needed (`cc -O2`), runs `gzip -dc | itch_filter`, then reconstruction per symbol; writes
  `<out>/<DATE>/<SYM>_{message,orderbook}_10.parquet` and `<out>/<DATE>/manifest.json`
  (input path, size, sha256 of the first 64 MiB, date parsed from the filename
  `SMMDDYY-v50.txt.gz`, weekday, counts, integrity counters, elapsed seconds).
- Prints progress; never loads the whole decompressed day into memory.

## 4. `scripts/symbol_stats.py`
- For converted symbol-days: messages, merged events (Gate 0 clock), mean and median
  inter-event gap, mid changes per event, mean spread in ticks, share of time at a
  one-tick spread, mean mid price. Writes `results/itch/symbol_stats.csv` and `.md`.
- Tick-size class per Briola et al.: mean spread <= 1.5 ticks large, >= 3 small, else
  medium.

## 5. Tests (must pass; no network)
- Synthetic ITCH byte streams packed in Python covering every kept type, a replace,
  partial and full executions, a cancel, a delete, a hidden execution, two symbols with one
  not requested, and a truncated final message. Check the filter output and the rebuilt
  book and LOBSTER rows against hand-computed expectations.
- On the 64 MB head sample: the C filter's per-type counts for 5 symbols equal those of a
  slow pure-Python reference parser written for the test.
- Loader parity: a small LOBSTER CSV and its Parquet conversion give identical arrays.

## 6. Run on the head sample and report
Run the CLI on the head sample for the candidate pool below (most will have only
pre-market messages; that is expected). Write `docs/itch_report.md`: commands, counts,
integrity counters, timing (messages per second for filter and rebuild, to estimate a full
day), and an Assumptions section.

Candidate pool for later asset selection (Nasdaq book only; single venue):
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD INTC CSCO CMCSA PEP QCOM SBUX PYPL
MU WBD.
