# Gate 0 spec: does skill decay over the delays that real latencies produce?

Goal: a cheap go/no-go test of H2 on the five free LOBSTER samples (AAPL, AMZN, GOOG, INTC,
MSFT; 2012-06-21; 10 levels; files in `data/raw/lobster/`, format in
`data/raw/lobster/LOBSTER_SampleFiles_ReadMe.txt`). Small models only. No deep learning.

## Deliverables
- `src/lob_latency/`: `io_lobster.py`, `events.py`, `labels.py`, `features.py`,
  `latency.py`, `models.py`, `metrics.py`.
- `scripts/run_gate0.py`: runs everything for all five tickers; one command, fixed seeds.
- `tests/`: unit tests listed below; all must pass.
- `results/gate0/<TICKER>/metrics.json`, `results/gate0/summary.md`, figures in
  `results/gate0/figures/`.
- `docs/gate0_report.md`: what was run (exact commands), the numbers, the Gate 0 verdict,
  and an "Assumptions" section.

## 1. Loading
- Message file columns: time (seconds after midnight, ns precision), type (1 submission,
  2 partial cancel, 3 deletion, 4 visible execution, 5 hidden execution, 6 cross trade,
  7 halt), order id, size, price (dollars x 10000), direction.
- Order book file: one row per message, after the message: ask price 1, ask size 1, bid
  price 1, bid size 1, ... to level 10. Empty levels use dummy prices (+/-9999999999).
- Drop type 7 rows. Keep types 1-6.

## 2. Event clock (decision 5)
- Merge consecutive messages with identical timestamps into one event. The event's book
  state is the order-book row of the last message in the group. Record per event: time,
  number of messages merged, whether any message was an execution (types 4, 5, 6).
- Mid-price m = (ask1 + bid1) / 2 from the event's book state; undefined if either side is
  a dummy level (drop such events from anchors and labels).
- Event index i runs over merged events within the day.

## 3. Labels (decision 4, L-next; action-time rule)
- `next_dir[j]`: +1 if the first event k > j with m_k != m_j has m_k > m_j, else 0 (down);
  undefined if no change before the session ends. `next_end[j]`: that k.
- Label for anchor i at delay tau: y(i, tau) = next_dir[i + tau]. The reference price is the
  mid at the action event i + tau. Nothing at or before i + tau may enter the target.
- tau grid: 0, 1, 2, 4, 8, 16, 32, 64 (events).

## 4. Anchors and splits (single day, so within-day chronological)
- Trimmed session: 09:45:00-15:50:00 (decision 12).
- Splits by anchor time: train 09:45-13:30, validation 13:30-14:40, test 14:40-15:50.
- An anchor i belongs to a split only if its whole input window and its label at tau = 64
  lie inside that split: t[i - W + 1] >= split start and t[next_end[i + 64]] < split end.
  This gives one anchor set used for every tau, and no leakage across boundaries.
- Lookback W = 50 events (features below need at most 50).

## 5. Features at anchor i (causal: only events <= i)
- Queue imbalance QI = (bid1 size - ask1 size) / (bid1 size + ask1 size).
- Depth imbalance over levels 1-3 and 1-5 (same formula, summed sizes).
- OFI (Cont, Kukanov & Stoikov 2014) per event from best bid/ask price and size changes:
  e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
        - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}.
  Rolling sums over the last K = 1, 5, 10, 50 events.
- Spread in ticks (tick = $0.01); log time since previous event; signed last-execution
  indicator (+1 buy-initiated, -1 sell-initiated, 0 none in last 10 events). LOBSTER's
  direction field on an execution is the side of the resting order, so trade sign =
  -direction (an executed sell limit order, direction -1, is a buy-initiated trade).
- Standardize with training-split statistics only.

## 6. Models (retrain per tau, decision 1)
- M1 `qi_logit`: logistic regression on QI and spread (Gould & Bonart style).
- M2 `ofi_logit`: logistic regression on all features.
- M3 `hgb`: sklearn HistGradientBoostingClassifier on all features, `early_stopping=False`,
  learning_rate 0.1; pick `max_iter` from {50, 100, 200, 400} by validation log-loss;
  5 seeds (0-4).
- For each tau: fit on train labels y(., tau); temperature-scale on validation (fit one
  temperature T > 0 on validation logits by minimizing log-loss); score on test.
- Naive variant: the tau = 0 model scored against y(., tau) for every tau.

## 7. Metrics
- S(tau) in bits per prediction: H(train prior) - mean test cross-entropy (base 2).
- Accuracy and MCC at threshold 0.5.
- Uncertainty: moving-block bootstrap over the test window with 5-minute blocks, 1000
  resamples, seed 0; 95% percentile intervals for S(tau) and for differences. The bootstrap
  is paired: on each resample, recompute P_hat(tau | L), every S(tau), every A(L) and every
  difference from the same resampled anchors. Never subtract two separate intervals. For M3 also
  report the spread across seeds.

## 8. Latency conversion
- L grid: 100 ns, 1, 10, 50, 100, 250, 500 us, 1, 2, 5, 10 ms.
- tau_i(L) = number of events with time in (t_i, t_i + L] (np.searchsorted on event times).
- Report per ticker on test anchors: P_hat(tau >= 1 | L), mean tau, and the 50/90/99th
  percentiles of tau.
- A_m(L) = sum_tau P_hat(tau | L) S_m(tau), with S linearly interpolated between grid points
  and held at S(64) beyond 64; report the probability mass beyond 64.
- Direct check at L = 10 us and 1 ms: retrain M2 on labels y(i, tau_i(L)) (the realized
  action event) and compare its test S with A_M2(L). Report both with intervals. tau_i(L)
  can exceed 64 in bursts: drop anchors whose label at tau_i(L) leaves their split, and
  report the fraction dropped per ticker.
- Burst split: S_M2(tau) on test anchors with tau_i(1 ms) = 0 vs >= 1, reported separately.

## 9. Gate 0 verdict (state it in the report, do not soften it)
- Primary quantity per ticker: Delta = A_best(100 ns) - A_best(2 ms), where best = the model
  with the highest S(0) on validation.
- PASS if Delta's 95% interval excludes zero for at least 2 of the 5 tickers.
- Otherwise FAIL: report that, at these timescales and on this day, latency up to 2 ms does
  not measurably cost skill. That is a valid result.

## 10. Tests (must pass)
- Event merge: synthetic messages with tied timestamps give the expected events and states.
- Labels: on a hand-built mid sequence, y(i, tau) equals the direction of the first change
  after i + tau. Changing mids strictly before index i + tau never changes y(i, tau).
  Changing mids after next_end[i + tau] never changes y(i, tau).
- No look-ahead in features: perturbing events after i leaves features at i unchanged.
- OFI: matches a hand-computed example covering each indicator case.
- tau_i(L): synthetic times with ties and an event exactly at t_i + L (counted) and at t_i
  (not counted).
- Metrics: a constant prior predictor gives S = 0 (within 1e-12 on the training prior); a
  perfect predictor gives S = H(prior).
- Anchor validity: no train anchor's label end falls in the validation window.

## Runtime
Should finish in minutes on an Apple M5. Print progress per ticker.
