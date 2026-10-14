# ITCH conversion report

The local head-sample conversion completed, and both the default suite and the additional theory tests passed. The prefix contains no regular-session observations for the candidate pool. Full-day conversion speed and regular-session behavior on real ITCH data remain **unverified**.

## Commands and evidence

Run from the repository root; no packages were installed and no network was used.

```sh
MPLCONFIGDIR=.cache/mpl .venv/bin/python scripts/itch_to_lobster.py \
  --input data/raw/itch_head/S120825_head64MB.gz \
  --symbols AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD INTC CSCO CMCSA PEP QCOM SBUX PYPL MU WBD \
  --out results/itch/head
MPLCONFIGDIR=.cache/mpl .venv/bin/python scripts/symbol_stats.py \
  --input results/itch/head --out results/itch
MPLCONFIGDIR=.cache/mpl .venv/bin/python -m pytest -q --junitxml=results/itch/pytest.xml
MPLCONFIGDIR=.cache/mpl .venv/bin/python -m pytest -q tests sims --junitxml=results/itch/all_pytest.xml
git diff --check
git diff --exit-code -- results/gate0 docs/theory.md
```

- [Conversion log](../results/itch/head_run.log), [manifest](../results/itch/head/2025-12-08/manifest.json), [filter counters](../results/itch/head/2025-12-08/filter_counts.json), and per-symbol integrity JSON beside the manifest.
- [Default suite log](../results/itch/tests.log), [expanded suite log](../results/itch/all_tests.log), [default JUnit](../results/itch/pytest.xml), and [expanded JUnit](../results/itch/all_pytest.xml).
- [Statistics CSV](../results/itch/symbol_stats.csv), [statistics Markdown](../results/itch/symbol_stats.md), [statistics configuration](../results/itch/symbol_stats_metadata.json), and [statistics log](../results/itch/stats_run.log).
- [Review checks](../results/itch/review_checks.json) record successful diff checks and matching current conversion-source hashes.

All measurements below come from those files. Binary intermediates were removed after successful conversion; Parquet files remain locally beside the manifest and are excluded by the existing repository ignore rules.

## Input coverage and counts

Input date: **2025-12-08 (Monday)**. Input size and hashed prefix: **67,108,864 bytes**. SHA-256 of that prefix: `12892d2b9cbb84caa7b976274486c6875b595b2ea8efa2618043224542758568`.

The filter read **164,495,360 decompressed bytes** and counted **5,496,895 complete messages**. The first and last complete message timestamps were **10889045483778 ns** and **19477367957042 ns** since midnight. Both gzip truncation and a partial final ITCH message were recorded; `complete_input` is false. The exact gzip diagnostics are retained in the manifest and conversion log.

Across the candidate pool, **126,266 filter records** were replayed. Every symbol emitted **0 session rows**. The converter still replayed pre-market orders, leaving the open-order counts shown below.

Hardware for every table: reported processor **arm**, machine **arm64**, macOS-26.5-arm64-arm-64bit, Python 3.12.13. The specific CPU model is unverified. The planned research timing platform recorded in [the existing research manifest](../results/gate0/manifest.json) is i5-13600K / RTX 4060 Ti 8 GB; it was not used for this conversion.

Whole-stream message counts, including types not retained in symbol binaries:

| ITCH type | Messages |
| --- | --- |
| A | 2429552 |
| D | 2244347 |
| E | 52799 |
| H | 12093 |
| L | 271181 |
| P | 35999 |
| R | 12102 |
| S | 2 |
| U | 191061 |
| X | 235666 |
| Y | 12093 |

Kept-type counts per requested symbol. Missing entries in the sparse counter JSON are shown as zero. Directory and other administrative counts are available in the filter JSON but do not enter these binary totals.

| Symbol | A | F | E | C | X | D | U | P | Q | H | Records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3973 | 0 | 124 | 0 | 10 | 1371 | 314 | 170 | 0 | 1 | 5963 |
| MSFT | 1560 | 0 | 65 | 0 | 13 | 1219 | 388 | 62 | 0 | 1 | 3308 |
| NVDA | 18762 | 0 | 1066 | 0 | 79 | 5850 | 1673 | 411 | 0 | 1 | 27842 |
| AMZN | 5757 | 0 | 113 | 0 | 18 | 739 | 271 | 154 | 0 | 1 | 7053 |
| GOOGL | 4261 | 0 | 433 | 0 | 43 | 1837 | 348 | 339 | 0 | 1 | 7262 |
| META | 1722 | 0 | 171 | 0 | 16 | 1300 | 415 | 112 | 0 | 1 | 3737 |
| TSLA | 15831 | 0 | 1143 | 0 | 686 | 7508 | 2543 | 951 | 0 | 1 | 28663 |
| AVGO | 2522 | 0 | 169 | 0 | 75 | 2226 | 1 | 223 | 0 | 1 | 5217 |
| COST | 91 | 0 | 3 | 0 | 0 | 39 | 0 | 9 | 0 | 1 | 143 |
| NFLX | 2928 | 0 | 304 | 0 | 65 | 2409 | 938 | 284 | 0 | 1 | 6929 |
| AMD | 4459 | 0 | 214 | 0 | 44 | 1765 | 366 | 198 | 0 | 1 | 7047 |
| INTC | 1747 | 0 | 104 | 0 | 33 | 1432 | 490 | 40 | 0 | 1 | 3847 |
| CSCO | 1098 | 0 | 32 | 0 | 0 | 1060 | 1379 | 15 | 0 | 1 | 3585 |
| CMCSA | 263 | 0 | 1 | 0 | 0 | 232 | 136 | 3 | 0 | 1 | 636 |
| PEP | 511 | 0 | 23 | 0 | 0 | 465 | 547 | 5 | 0 | 1 | 1552 |
| QCOM | 435 | 0 | 17 | 0 | 29 | 317 | 474 | 24 | 0 | 1 | 1297 |
| SBUX | 204 | 0 | 0 | 0 | 0 | 147 | 236 | 2 | 0 | 1 | 590 |
| PYPL | 661 | 0 | 9 | 0 | 8 | 487 | 302 | 17 | 0 | 1 | 1485 |
| MU | 3781 | 0 | 184 | 0 | 67 | 2480 | 859 | 228 | 0 | 1 | 7600 |
| WBD | 997 | 0 | 128 | 0 | 5 | 883 | 452 | 44 | 0 | 1 | 2510 |

## Integrity and reconstruction timing

Counter scope: unknown references and over-removals cover the whole available prefix; crossed/locked counts and durations cover only the regular session. Thus the zero crossed/locked counts here do not validate regular-session market quality. No duplicate references were recorded. Separate crossed and locked counters, interval counts and total durations are in each integrity JSON.

| Symbol | Unknown refs | Negative shares | Crossed/locked | Longest ns | Open orders | Rebuild s | Records/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 0 | 0 | 0 | 0 | 2522 | 0.030691 | 194288 |
| MSFT | 0 | 0 | 0 | 0 | 302 | 0.012756 | 259323 |
| NVDA | 0 | 0 | 0 | 0 | 12259 | 0.068879 | 404215 |
| AMZN | 0 | 0 | 0 | 0 | 4944 | 0.019865 | 355044 |
| GOOGL | 0 | 0 | 0 | 0 | 2191 | 0.021063 | 344777 |
| META | 0 | 0 | 0 | 0 | 339 | 0.013018 | 287071 |
| TSLA | 0 | 0 | 0 | 0 | 7607 | 0.071588 | 400386 |
| AVGO | 0 | 0 | 0 | 0 | 214 | 0.015986 | 326340 |
| COST | 0 | 0 | 0 | 0 | 50 | 0.005690 | 25133 |
| NFLX | 0 | 0 | 0 | 0 | 334 | 0.020799 | 333145 |
| AMD | 0 | 0 | 0 | 0 | 2635 | 0.020295 | 347220 |
| INTC | 0 | 0 | 0 | 0 | 255 | 0.013201 | 291423 |
| CSCO | 0 | 0 | 0 | 0 | 35 | 0.014106 | 254155 |
| CMCSA | 0 | 0 | 0 | 0 | 30 | 0.005912 | 107578 |
| PEP | 0 | 0 | 0 | 0 | 45 | 0.007808 | 198763 |
| QCOM | 0 | 0 | 0 | 0 | 113 | 0.008807 | 147263 |
| SBUX | 0 | 0 | 0 | 0 | 57 | 0.005487 | 107533 |
| PYPL | 0 | 0 | 0 | 0 | 168 | 0.007957 | 186616 |
| MU | 0 | 0 | 0 | 0 | 1196 | 0.022427 | 338879 |
| WBD | 0 | 0 | 0 | 0 | 79 | 0.010602 | 236748 |

The rates below use different denominators: filtering counts all complete feed messages; rebuilding counts only selected-symbol binary records. Filter timing includes the streaming gzip pipeline, and rebuild timing includes writer setup and close.

| Stage | Messages | Elapsed s | Messages/s |
| --- | --- | --- | --- |
| gzip + filter | 5496895 | 0.197279 | 27863600 |
| All candidate rebuilds | 126266 | 0.396939 | 318100 |
| End-to-end, including setup and hashing | - | 0.651920 | - |

For workload-based planning, the recorded timing model is **35.889 seconds per billion feed messages** plus **3.144 seconds per million selected messages**, before additional regular-session output costs and changes in book size or workload. A full-day estimate can substitute the anticipated whole-feed and selected-symbol message counts into these rates. This is an **unverified extrapolation**, not a full-day benchmark: the sample wrote no regular-session rows, and pre-market workload, open-order memory, caching and compression can differ substantially.

## Tests and loader compatibility

- Default discovery: `58 passed, 1 warning in 8.86s` ([log](../results/itch/tests.log)).
- Including theory tests: `65 passed, 1 warning in 8.46s` ([log](../results/itch/all_tests.log)).
- The warning comes from joblib physical-core detection; its fallback to logical cores is recorded in both logs.
- Synthetic fixtures exercise every kept message type, attributed orders, replacements with intermediate snapshots, partial and full executions, full cancellation, deletion, hidden and cross trades, unrequested symbols, session boundaries, out-of-range depth changes, unknown references, over-removal, duration clipping and truncated frames.
- The head test compares all global counts, all per-type counts for AAPL/MSFT/NVDA/AMZN/GOOGL, binary record counts, consumed bytes and partial-frame status with an independent Python framing parser. Both parsers consume the system gzip output. This deliberately does not claim independent verification of damaged-deflate recovery: Python gzip and macOS gzip expose different tails from an incomplete final block.
- Loader tests compare values and array dtypes for CSV and exact decimal-time Parquet, including halt alignment, nanosecond timestamps and unsigned order identifiers. Floating-point Parquet times are rejected because their original nanoseconds may already be lost.
- CLI coverage includes both intermediate-file policies, both supported filename forms, accepted gzip EOF, and rejection of invalid gzip input, checksum failures and overflowing cross quantities. Hand-computed statistics check timestamp merging, gaps, mid changes, time-weighted one-tick shares and classification boundaries.
- The old provenance assertion treated the loader source as immutable. It now verifies the historical loader blob at revision `6a2c3aa` against the unchanged research manifest, while current compatibility is tested separately. The revision and hash comparison are recorded in the review checks. No Gate 0 results or theory documentation were rewritten.

## Symbol selection

The statistics files contain every converted candidate symbol-day. Their regular-session message and event counts are zero, their observed-book durations are zero, and price/gap/spread metrics are null with class `unclassified`. No assets can be selected from this prefix. Selection still requires training-day regular-session observations across the candidate pool; this run establishes conversion evidence only.

## Assumptions

- The stock-directory spin precedes dependent symbol messages, locate assignments remain stable within the day, and replay begins at the available start of the feed. A missing earlier prefix cannot be reconstructed from later updates.
- The filename supplies the exchange calendar date; timestamps are nanoseconds since exchange-local midnight without timezone conversion. The source is the Nasdaq venue, not a consolidated national book.
- The fixed record uses little-endian integers and raw ASCII type/side/printable bytes with zero-filled unused fields. Its share field cannot represent every possible cross quantity; overflow is rejected instead of wrapped.
- Visible execution direction uses the resting order. Modern hidden-trade direction and cross direction are unavailable and therefore use zero. The raw hidden-trade buy indicator is not interpreted as an observed resting side.
- All trading-action messages become halt-category rows with zero placeholders. The prescribed record has no field for trading state or reason, so it does not distinguish halt, quotation-only and resumption states. The research loader drops these rows.
- Execution-with-price messages remove depth at the original resting price and report the execution price. Non-printable executions are retained for book/event reconstruction; this output is not a deduplicated time-and-sales volume series.
- Replacements are replayed as deletion and submission at the same timestamp and side. Each logical row is tested for a changed snapshot; executions, hidden trades, crosses and trading actions are emitted even without a snapshot change.
- Unknown modifications are counted and skipped when their resting side cannot be recovered; unknown executions still emit with unknown side and, where unavailable, price represented by zero. Over-removals are counted and clipped in the book; the message row retains the reported removal size. Duplicate additions/replacement targets are counted and skipped rather than overwriting live orders.
- Conversion uses inclusive regular-session endpoints recorded in each integrity configuration. Statistics apply the separate research trim recorded in the statistics metadata. Reconstruction continues outside the emission window.
- Missing depth uses the signed dummy prices specified by the format and zero shares. Message seconds use exact decimal storage; loader arrays use integer nanoseconds. Binary records are memory-mapped and Parquet rows are batched; live-order and price-level state still grows with the book.
- Integrity state counts refer to the final state after each input message, so a replacement does not add an artificial intermediate crossed/locked count. Durations use contiguous state intervals clipped to the session and available whole-feed observation end, rather than extrapolating an incomplete feed to the close.
- Statistics merge timestamps using the existing event clock and discard halt rows. Spread and mid means are event averages on valid positive-spread books. Tick size and class thresholds are the fixed values in the statistics metadata. Adjacent valid mid changes are divided by all trimmed events; the first has no predecessor.
- The one-tick share is weighted by observed book dwell time, excluding missing/crossed/locked intervals. A prior book can carry into the trim boundary; no interval is extended beyond the known observation end. No observations produce null metrics rather than invented estimates.
- Gzip premature-EOF diagnostics are accepted, including the extra generic macOS failure line only when accompanied by that EOF diagnostic. Format and checksum failures remain errors. Complete messages after the decoder's last returned bytes are not inferred or recovered.
- No random sampling is used; output seeds are null. Timings are local wall-clock measurements with cached compilation and are not inference-latency measurements. Linux execution, full-day peak memory and real regular-session conversion throughput remain unverified.
