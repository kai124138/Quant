# Gate 0b spec: the Gate 0 decay test on multi-day 2025-2026 Nasdaq data

Goal: rerun the Gate 0 analysis (`docs/gate0_spec.md`, code in `scripts/run_gate0.py` and
`src/lob_latency/`) on ITCH-derived LOBSTER-format Parquet files, with whole-day splits.
Reuse the Gate 0 modules; generalize rather than fork them. Gate 0 results must stay
reproducible from the existing command.

## Inputs
- Data root (CLI `--data-root`): `<root>/<YYYY-MM-DD>/<SYM>_{message,orderbook}_10.parquet`
  plus `<root>/<YYYY-MM-DD>/manifest.json`, as written by `scripts/itch_to_lobster.py`.
- Days: train 2025-12-08, 2025-12-09, 2025-12-10; validation 2025-12-11; test 2025-12-12;
  out-of-time test 2026-06-12 (scored with the same fitted models, no refit).

## 1. Asset selection: `scripts/select_assets.py`
- Read `results/itch/symbol_stats.csv`, keep training days only (Dec 8-10), and average
  per symbol. Deterministic rule, written to `results/itch/selected_assets.json` with the
  numbers behind each pick:
  (a) the large-tick symbol with the highest event rate;
  (b) a large-tick symbol with event rate at or below the large-tick median (the highest
      such one);
  (c) the small-tick symbol with the highest event rate;
  (d) if a medium-tick symbol exists, the one with the highest event rate.
- Exclude symbols with any training day missing, a crossed/locked share above 1% of the
  session, or unknown-order counts above zero.

## 2. Multi-day pipeline: `scripts/run_gate0b.py`
- Everything per day: merged-event clock, mids, labels, features, latency counts. No
  window, label or latency count crosses a day. Session trim 09:45-15:50 on every day;
  an anchor is valid only if its input window and its label at tau = 64 lie in the same
  day's trimmed session.
- Feature standardization: statistics from the training days only.
- Models, tau grid, temperature scaling, metrics and the latency grid as in Gate 0. Fit on
  the three training days; select `max_iter` and temperatures on the validation day.
- Bootstrap: moving 30-minute blocks within each test day (1000 resamples, seed 0, paired
  as in Gate 0). Also report 5-minute blocks as a sensitivity check. State in the report
  that within-day blocks understate day-to-day variation.
- Half-life (theory.md section 4): fit a non-increasing isotonic curve to S(tau) on the tau
  grid, report tau_1/2 in events (linear interpolation on the fit) with a bootstrap
  interval, and in wall-clock as the median and IQR of t[i + tau_1/2] - t[i] over test
  anchors.
- Direct wall-clock check and burst split as in Gate 0, on the 2025-12-12 test day.
- Verdict per test day: PASS if Delta = A_best(100 ns) - A_best(2 ms) has a 30-minute-block
  interval excluding zero for at least 2 of the selected symbols.
- Outputs: `results/gate0b/<SYM>/metrics.json`, `results/gate0b/summary.{json,md}`,
  figures in `results/gate0b/figures/`, and `docs/gate0b_report.md` (commands, numbers,
  both verdicts, comparison with the 2012 Gate 0 numbers, and an Assumptions section).

## 3. Tests (no network; real ITCH days are not available here)
- Build a fake multi-day root by converting the five LOBSTER 2012 CSV samples to the
  Parquet layout under a temp dir, assigning them to synthetic dates, and run the whole
  pipeline end to end at reduced settings (fewer resamples, one HGB seed).
- Day isolation: anchors near the end of a day never take labels, features or latency
  counts from the next day.
- Asset selection on a hand-built stats table picks the expected symbols, including the
  exclusions.

The full run happens later on another machine with the real days; here, make the code and
tests complete and leave a one-line command in the report for that run.
