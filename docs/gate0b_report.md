# Gate 0b report

Run kind: **real_days_subsampled**.

Both held-out days use the same training and validation fits. Anchors are evenly subsampled to at most 500000 per day; the event clock, features, labels and latency counts use every event.

## Commands and evidence

Executed pipeline command:

```sh
MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/run_gate0b.py --data-root /home/kaimoe/quant-data/lobster --max-anchors-per-day 500000
```

Evidence: manifest.json, summary.json, per-symbol metrics.json and bootstrap artifacts under results/gate0b.
All reported values below come from those result JSON files; bootstrap intervals condition on fixed fits.

Later full run, from the repository root after placing complete converter output in data/raw/itch_lobster:

```sh
MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/symbol_stats.py --input data/raw/itch_lobster --out results/itch && .venv/bin/python scripts/select_assets.py && MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/run_gate0b.py --data-root data/raw/itch_lobster --assets results/itch/selected_assets.json --output results/gate0b --report docs/gate0b_report.md
```

## Configuration

```json
{
  "symbols": [
    "NVDA",
    "CSCO",
    "TSLA",
    "GOOGL"
  ],
  "train_days": [
    "2025-12-08",
    "2025-12-09",
    "2025-12-10"
  ],
  "validation_day": "2025-12-11",
  "test_days": [
    "2025-12-12",
    "2026-06-12"
  ],
  "session_seconds": [
    35100,
    57000
  ],
  "tau_events": [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512
  ],
  "latency_ns": [
    100,
    1000,
    10000,
    50000,
    100000,
    250000,
    500000,
    1000000,
    2000000,
    5000000,
    10000000
  ],
  "lookback_events": 50,
  "ofi_preceding_book_guard_events": 1,
  "direct_check_latency_ns": [
    10000,
    1000000
  ],
  "hgb_seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "hgb_iterations": [
    50,
    100,
    200,
    400
  ],
  "logit_seed": 0,
  "bootstrap_seed": 0,
  "resamples": 1000,
  "primary_block_seconds": 1800,
  "sensitivity_block_seconds": 300,
  "ci_percentiles": [
    2.5,
    97.5
  ],
  "max_anchors_per_day": 500000,
  "out_of_time_refit": false,
  "required_significant_symbols": 2
}
```

Execution hardware: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 / x86_64. Reference: i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

## Verdicts and primary results

Skill is in bits/prediction: zero means test cross-entropy equals training-prior entropy.
Delta is fast minus slow adjusted skill. Brackets are paired percentile intervals.

**2025-12-12: PASS**; 4 selected symbols exclude zero.

**2026-06-12: PASS**; 4 selected symbols exclude zero.

| Day | Symbol | Validation best | Fast A | Slow A | Delta, primary | Delta, sensitivity |
|---|---|---|---:|---:|---:|---:|
| 2025-12-12 | NVDA | hgb | 0.10321843 | 0.03142704 | 0.071791 [0.065976, 0.076404] | 0.071791 [0.066912, 0.072239] |
| 2026-06-12 | NVDA | hgb | 0.06178335 | 0.02274270 | 0.039041 [0.036223, 0.040197] | 0.039041 [0.035727, 0.040998] |
| 2025-12-12 | CSCO | hgb | 0.09334116 | 0.04817500 | 0.045166 [0.042734, 0.047081] | 0.045166 [0.042784, 0.048125] |
| 2026-06-12 | CSCO | hgb | 0.01848845 | 0.00748006 | 0.011008 [0.004810, 0.015721] | 0.011008 [0.005227, 0.011969] |
| 2025-12-12 | TSLA | hgb | 0.10451101 | 0.04720309 | 0.057308 [0.053523, 0.060696] | 0.057308 [0.056411, 0.060570] |
| 2026-06-12 | TSLA | hgb | 0.01773018 | 0.00361406 | 0.014116 [0.010911, 0.018442] | 0.014116 [0.012405, 0.016509] |
| 2025-12-12 | GOOGL | hgb | 0.07501151 | 0.01830957 | 0.056702 [0.054162, 0.059286] | 0.056702 [0.055764, 0.059302] |
| 2026-06-12 | GOOGL | hgb | 0.03466904 | 0.01110474 | 0.023564 [0.020973, 0.025477] | 0.023564 [0.021434, 0.025147] |

## Half-life

The raw curves and isotonic fits for every model are in metrics.json. These are fitted skill surrogates.
A null upper interval endpoint with an unbounded flag denotes a right-censored crossing.

| Day | Symbol | Selected model | Status | Events | Event interval | Clock median seconds | Clock quartiles seconds | Censored mass | Unidentified mass |
|---|---|---|---|---:|---|---:|---|---:|---:|
| 2025-12-12 | NVDA | hgb | observed | 2.208146595121752 | [1.8911934002692405, 2.2825202959571183] | 0.0001749309698496716 | [3.340828350886908e-05, 0.0017652191881917722] | 0.0 | 0.0 |
| 2026-06-12 | NVDA | hgb | observed | 1.7275048469358496 | [1.5325957872086904, 2.102701650454444] | 0.00014296392154854247 | [2.4145604912926485e-05, 0.002362447410201764] | 0.0 | 0.0 |
| 2025-12-12 | CSCO | hgb | observed | 3.520257493317601 | [2.8057606322452124, 3.657594180613739] | 0.002621992698494374 | [9.467722387763807e-05, 0.08792867214370897] | 0.0 | 0.0 |
| 2026-06-12 | CSCO | hgb | observed | 2.1310640935241363 | [1.6458822560653408, 3.5242637773422354] | 0.0012900311985718992 | [6.172953890096045e-05, 0.03631084924877366] | 0.0 | 0.0 |
| 2025-12-12 | TSLA | hgb | observed | 2.622728018983428 | [2.4277307571219717, 3.1208961586969566] | 0.00046289806444495653 | [4.833017292577063e-05, 0.009189993231990773] | 0.0 | 0.0 |
| 2026-06-12 | TSLA | hgb | observed | 1.3991118837099399 | [0.7492937904337097, 2.380542197076647] | 0.00010307053797349048 | [1.7005793177277843e-05, 0.0036157478570855617] | 0.0 | 0.001 |
| 2025-12-12 | GOOGL | hgb | observed | 0.8265378608798798 | [0.7926720859631331, 0.8574467473378087] | 3.498734765104531e-05 | [1.1380599806455065e-05, 0.00039660964658497547] | 0.0 | 0.0 |
| 2026-06-12 | GOOGL | hgb | observed | 1.0683905395473094 | [0.9798391265199857, 1.3523535680257006] | 9.917465197605056e-05 | [1.6757669787052893e-05, 0.0016119554221403721] | 0.0 | 0.0 |

## Direct wall-clock check and burst split

The December test's direct checks and full OFI subgroup curves, paired differences, boundary drops and finite bootstrap counts are stored in each metrics.json.

| Symbol | Latency ns | Direct S | Mixture on same anchors | Direct minus mixture |
|---|---:|---:|---:|---:|
| NVDA | 10000 | 0.073875 [0.065812, 0.079277] | 0.071178 [0.063471, 0.075920] | 0.002697 [0.002239, 0.003412] |
| NVDA | 1000000 | 0.009914 [0.007162, 0.010656] | 0.027444 [0.022804, 0.028818] | -0.017531 [-0.018470, -0.015407] |
| CSCO | 10000 | 0.068855 [0.061943, 0.071161] | 0.067308 [0.060414, 0.069397] | 0.001547 [0.001180, 0.002277] |
| CSCO | 1000000 | 0.015234 [0.012803, 0.018158] | 0.040367 [0.034770, 0.041879] | -0.025133 [-0.025414, -0.020166] |
| TSLA | 10000 | 0.062984 [0.061287, 0.067485] | 0.064335 [0.062912, 0.068637] | -0.001351 [-0.001935, -0.000923] |
| TSLA | 1000000 | 0.012787 [0.011705, 0.014882] | 0.033789 [0.033229, 0.036899] | -0.021002 [-0.023138, -0.020390] |
| GOOGL | 10000 | 0.047388 [0.044808, 0.049280] | 0.047580 [0.045242, 0.049123] | -0.000192 [-0.000576, 0.000288] |
| GOOGL | 1000000 | 0.004379 [0.003109, 0.004895] | 0.014856 [0.013666, 0.016377] | -0.010477 [-0.012859, -0.009111] |

## Historical Gate 0 comparison

Source: the unchanged results/gate0/summary.json, also copied into this run's summary.json. Historical verdict: **PASS**.
Historical within-day splits and block duration differ from this whole-day experiment. Synthetic-date repetitions cannot establish a modern-market change in skill.

| Historical symbol | Selected model | Historical Delta bits | Historical interval |
|---|---|---:|---|
| AAPL | hgb | 0.00813037 | [0.006806000560621302, 0.01052442349658353] |
| AMZN | ofi_logit | 0.00466346 | [0.0020655543341899616, 0.0069391471658048895] |
| GOOG | hgb | 0.00730596 | [0.0019941820046019804, 0.008575612842936372] |
| INTC | qi_logit | 0.01291462 | [0.0084872117082769, 0.020437029043059652] |
| MSFT | hgb | 0.03007661 | [0.023695139170256358, 0.03737567485671913] |

## Assumptions

- Input timestamps are exchange-local seconds after midnight, with exact decimal parsing and equal-timestamp merging as in Gate 0.
- Each day is trimmed before features, labels and counts; the preceding-book OFI guard and common valid labels apply. Complete maximum-latency windows are required through the last observed event.
- Training days are pooled by anchor count for feature statistics, class priors and fitting; validation alone selects iterations, temperatures and the best model.
- Out-of-time scores reuse the same fitted models, temperatures, training priors and feature scales without refitting or reselection.
- Asset selection uses equal-day averaged event rates and event-mean spreads; per-day quality exclusions use trimmed-session time fractions. Missing integrity evidence is excluded. Ties use ascending symbol order, and picks must be distinct.
- The symbol-statistics producer was extended to fix missing event-rate and integrity fields required by selection; unknown references come from reconstruction manifests or integrity JSON, never assumed zero.
- HGB metrics average individual calibrated seed losses and classification metrics, rather than probability-ensemble scores. Naive scoring freezes the zero-delay model and temperature.
- Bootstrap blocks start uniformly at eligible anchors, retain variable event counts and use the same draws for all losses, distributions, differences and diagnostics within each day. Fits and training priors stay fixed.
- Within-day blocks understate day-to-day variation; these intervals do not include training, calibration or asset-selection uncertainty.
- Latency mixtures interpolate event-delay skills and hold the last grid value beyond the grid; burst dependence can violate this approximation. Direct checks use matching surviving anchors.
- Half-life uses equal-weight non-increasing isotonic skill clipped at zero as a surrogate, with first-crossing interpolation. Censored bootstrap crossings retain unbounded endpoints; crossing intervals condition on positive initial skill and report unidentified mass.
- Fractional half-life event indices interpolate adjacent timestamps; median and IQR describe elapsed times across the common test anchors, rather than a confidence interval for clock conversion.
- Latencies are assumed scenarios; inference timing, quantization, FPGA behavior and profitability are unverified. Tables name the planned reference hardware, whose timing is unmeasured.
- The prescribed gate accepts an interval excluding zero in either direction, without an added multiple-comparison correction; a negative Delta is not evidence of a latency penalty.
- Synthetic dates repeat historical sample streams and reduced fixture settings; their verdicts exercise code and provide no evidence of modern-market or across-day generalization.
- The later full-run command assumes complete converter output under data/raw/itch_lobster and regeneration of statistics before selection. Real-day results remain unverified until that command runs.
