# Gate 0 report

**Verdict: PASS.** The paired interval for the prescribed primary loss is strictly positive for 5 of 5 tickers. The rule requires at least 2. On this sample day, the specified latency-adjusted score measurably loses skill between the fast and slow latency scenarios.

The primary loss is `Delta = A_best(100 ns) - A_best(2 ms)`. The best model is chosen only by calibrated validation `S(0)`, before looking at test performance. The values and intervals below come from the completed run, not from the failed initial attempt.

## Commands and evidence

Working directory: the repository root.

The corrected full run used exactly:

```sh
set -o pipefail
MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python scripts/run_gate0.py 2>&1 | tee results/gate0/run.log
```

After the run, the final unit and artifact integration checks used exactly:

```sh
set -o pipefail
MPLCONFIGDIR=.cache/mpl OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q | tee results/gate0/tests.log
```

Evidence: [run.log](../results/gate0/run.log) ends with `Gate 0 PASS: 5/5 tickers exclude zero`; [tests.log](../results/gate0/tests.log) records `36 passed in 4.23s`. Both commands exited successfully. The integration tests reconstruct the raw-data anchors and every delay's split boundaries, verify input/source hashes and selected models, and independently reconstruct adjusted scores with `numpy.interp` and the stored empirical distributions.

The completed pipeline recorded 904.876828 seconds. Python and platform were `3.12.13` and `macOS-26.5-arm64-arm-64bit`. The processor brand is **unverified**: the sandbox denied the local brand query, so the manifest records only `arm`. The timing reference is i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here). No inference latency or FPGA performance was measured.

The initial full attempt stopped on AMZN because the first parser rejected extra decimal digits below nanosecond precision. That failed attempt is preserved in [initial_run.log](../results/gate0/initial_run.log) and [initial_manifest.json](../results/gate0/initial_manifest.json). The corrected parser uses explicit CSV column names and rounds decimal timestamp tails directly to integer nanoseconds. Synthetic regression tests cover both cases. The unit-suite result before the corrected full run is preserved in [pre_full_tests.log](../results/gate0/pre_full_tests.log).

## Data and configuration

Data date: `2012-06-21`. Full-sample event counts below precede the session trim; the anchor counts are after split, warmup, and common-label guards. Halts are dropped and equal timestamps are merged using the final book row. Only common anchors with a defined action-time label at every delay are scored.

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Raw messages | Merged events | Train anchors | Validation anchors | Test anchors | Rounded timestamp rows |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | 400391 | 384329 | 203282 | 58581 | 85792 | 0 |
| AMZN | 269748 | 261265 | 143626 | 42258 | 56898 | 2 |
| GOOG | 147916 | 139379 | 76910 | 22946 | 26298 | 0 |
| INTC | 624040 | 558154 | 303045 | 60865 | 110735 | 5 |
| MSFT | 668765 | 611562 | 335034 | 76591 | 128378 | 3 |

Lookback: 50 events, with 1 extra preceding book event inside the split for OFI. Delay grid (events): `[0, 1, 2, 4, 8, 16, 32, 64]`. The time splits and feature standardization statistics are recorded in each `metrics.json`; split endpoints and the full latency grid are also in [manifest.json](../results/gate0/manifest.json).

The logistic models use seed 0; HGB uses seeds `[0, 1, 2, 3, 4]`, learning rate 0.1, and validation selection from iteration counts `[50, 100, 200, 400]`, with early stopping disabled. Each trained model has its own validation-fitted positive temperature. HGB tables average individual seed losses and metrics. The naive variant keeps the zero-delay model and temperature fixed.

Uncertainty uses 1000 moving-block resamples of 300 seconds, seed 0. The same blocks resample all losses, latency distributions, adjusted scores, direct checks, subgroup scores, and differences. Training priors remain fixed. Intervals are the percentiles `[2.5, 97.5]`; no interval is formed by subtracting separately computed bounds. The actual block draws and test anchor indices are saved per ticker in `bootstrap_blocks.npz`.

## Primary result and model selection

All skill values are bits per prediction: `S = H(training prior) - mean test cross-entropy in bits`. Zero means test cross-entropy equals training-prior entropy. Positive skill means test cross-entropy is below that entropy; negative skill means it is above. Brackets below are paired 95% percentile intervals. Tables round the persisted values; JSON retains full precision.

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Selected model | A(100 ns) [interval] | A(2 ms) [interval] | Delta [interval] |
| --- | --- | --- | --- | --- |
| AAPL | hgb | 0.04614553 [0.03827577, 0.05743218] | 0.03801516 [0.03050566, 0.04754227] | 0.00813037 [0.00680600, 0.01052442] |
| AMZN | ofi_logit | 0.06106651 [0.04411352, 0.08684576] | 0.05640305 [0.04108844, 0.08051812] | 0.00466346 [0.00206555, 0.00693915] |
| GOOG | hgb | 0.05150557 [0.03192652, 0.06382838] | 0.04419961 [0.02713012, 0.05752014] | 0.00730596 [0.00199418, 0.00857561] |
| INTC | qi_logit | 0.06278902 [0.03038520, 0.11720411] | 0.04987440 [0.02068991, 0.09767228] | 0.01291462 [0.00848721, 0.02043703] |
| MSFT | hgb | 0.12887328 [0.09733092, 0.14583983] | 0.09879667 [0.07205823, 0.10936049] | 0.03007661 [0.02369514, 0.03737567] |

Selection evidence: calibrated validation skill at zero event delay.

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | qi_logit validation S(0) | ofi_logit validation S(0) | hgb validation S(0) |
| --- | --- | --- | --- |
| AAPL | 0.01224372 | 0.03212899 | 0.03397971 |
| AMZN | 0.01566737 | 0.04251438 | 0.03715994 |
| GOOG | 0.01085996 | 0.03637141 | 0.04491109 |
| INTC | 0.07435386 | 0.06870817 | 0.06526931 |
| MSFT | 0.08764385 | 0.09842536 | 0.10538813 |

Endpoint event-delay behavior of the selected model; accuracy and MCC refer to zero delay. The complete curves for every model and both training strategies are in [summary.md](../results/gate0/summary.md).

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Retrained S(0) | Retrained S(64) | Naive S(64) | Retrained S(0)-S(64) | Accuracy(0) | MCC(0) |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | 0.04614553 [0.03827577, 0.05743218] | 0.00154142 [0.00015868, 0.00462102] | -0.01583619 [-0.02450173, -0.00363335] | 0.04460411 [0.03738610, 0.05338217] | 0.59978087 | 0.19834607 |
| AMZN | 0.06106651 [0.04411352, 0.08684576] | 0.01060575 [0.00431801, 0.01697128] | 0.00706478 [-0.01021646, 0.02410828] | 0.05046076 [0.03708252, 0.07246663] | 0.61580372 | 0.23148681 |
| GOOG | 0.05150557 [0.03192652, 0.06382838] | -0.00376081 [-0.00899594, 0.00284100] | -0.01988504 [-0.03727594, 0.00325718] | 0.05526638 [0.03679330, 0.06451744] | 0.61594038 | 0.22803266 |
| INTC | 0.06278902 [0.03038520, 0.11720411] | 0.02440276 [0.00260799, 0.05994238] | 0.01159131 [-0.02113937, 0.06430021] | 0.03838626 [0.02649455, 0.05859082] | 0.62053551 | 0.25217074 |
| MSFT | 0.12887328 [0.09733092, 0.14583983] | 0.05079316 [0.02700672, 0.05535669] | 0.04929152 [0.01282263, 0.05962335] | 0.07808013 [0.06076831, 0.09680504] | 0.66228949 | 0.31334040 |

## Latency conversion and the tail

The primary slow-latency distribution on common test anchors is shown below. The complete distributions, including empirical PMFs, all requested quantiles, and beyond-grid mass at every latency, are in each ticker JSON and the complete table in [summary.md](../results/gate0/summary.md).

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker at 2 ms | P(tau >= 1) | Mean tau (events) | p50 | p90 | p99 | P(tau > 64) |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | 0.65530586 | 2.30399105 | 1.0 | 6.0 | 16.0 | 0.00000000 |
| AMZN | 0.63218391 | 2.78892052 | 1.0 | 7.0 | 20.0 | 0.00405990 |
| GOOG | 0.55236140 | 2.12153015 | 1.0 | 6.0 | 19.0 | 0.00000000 |
| INTC | 0.73004922 | 32.79804037 | 3.0 | 125.0 | 200.0 | 0.20978914 |
| MSFT | 0.77524186 | 35.81185250 | 6.0 | 123.0 | 181.0 | 0.23890386 |

The beyond-grid mass is substantial for INTC and MSFT. Their adjusted scores use the prescribed constant tail, not models trained at those larger delays. The sign and size of any additional tail decay are **unverified**. This approximation is part of the stated Gate criterion and does not change the PASS verdict computed above.

## Direct wall-clock checks

These retrain `ofi_logit` on each realized action-event label. Direct skill and the paired mixture score use the same retained test anchors. The mixture recomputes both its delay-specific skills and its latency weights on each resample. Direct training uses its own training-label prior. The full-common-anchor mixture scores are separately stored as `A_full_bits` in JSON.

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Latency | Direct S [interval] | A on same anchors [interval] | Direct - A [interval] | Fraction dropped train/val/test |
| --- | --- | --- | --- | --- | --- |
| AAPL | 10000 ns | 0.04391673 [0.03802887, 0.05476348] | 0.04374865 [0.03747094, 0.05449009] | 0.00016808 [-0.00012877, 0.00072975] | 0.00000000/0.00000000/0.00000000 |
| AAPL | 1000000 ns | 0.04016811 [0.03388084, 0.04992863] | 0.03905874 [0.03307466, 0.04844514] | 0.00110938 [-0.00077943, 0.00297634] | 0.00000000/0.00000000/0.00000000 |
| AMZN | 10000 ns | 0.06105040 [0.04401156, 0.08686286] | 0.06089429 [0.04407531, 0.08656438] | 0.00015611 [-0.00010253, 0.00042876] | 0.00000000/0.00000000/0.00000000 |
| AMZN | 1000000 ns | 0.05636075 [0.03933878, 0.08084160] | 0.05763829 [0.04211875, 0.08183017] | -0.00127753 [-0.00420818, 0.00136211] | 0.00000000/0.00000000/0.00000000 |
| GOOG | 10000 ns | 0.04696992 [0.02998574, 0.06525659] | 0.04712404 [0.02986892, 0.06556146] | -0.00015412 [-0.00058498, 0.00033032] | 0.00000000/0.00000000/0.00000000 |
| GOOG | 1000000 ns | 0.04292634 [0.02433900, 0.06477940] | 0.04298672 [0.02625424, 0.06197583] | -0.00006038 [-0.00354993, 0.00553850] | 0.00000000/0.00000000/0.00000000 |
| INTC | 10000 ns | 0.07493615 [0.04426488, 0.12593378] | 0.07502334 [0.04433471, 0.12597556] | -0.00008720 [-0.00034271, 0.00016232] | 0.00000000/0.00000000/0.00000000 |
| INTC | 1000000 ns | 0.06018876 [0.03515452, 0.10129523] | 0.06342644 [0.03713975, 0.10903591] | -0.00323769 [-0.01562975, 0.00647153] | 0.00000000/0.00062433/0.00029801 |
| MSFT | 10000 ns | 0.12906284 [0.10545451, 0.15453254] | 0.12824007 [0.10496605, 0.15361529] | 0.00082276 [0.00025565, 0.00109812] | 0.00000000/0.00000000/0.00000000 |
| MSFT | 1000000 ns | 0.10837641 [0.07759148, 0.12527617] | 0.10703037 [0.08608788, 0.12882616] | 0.00134604 [-0.01334080, 0.00316336] | 0.00000000/0.00000000/0.00001558 |

MSFT at the shorter direct-check latency has a positive direct-minus-mixture interval excluding zero. The other listed difference intervals include zero; that is not proof of equivalence. The mixture is therefore an approximate latency conversion, not an exact substitute for direct retraining.

## Burst split and seed spread

The burst split uses `ofi_logit`: quiet anchors have `tau_i(1 ms) = 0`, and active anchors have `tau_i(1 ms) >= 1`. Below are its zero-event-delay skills. Complete subgroup curves and intervals are in [summary.md](../results/gate0/summary.md); JSON also retains paired active-minus-quiet differences at every delay. The same full-training prior is used in both groups.

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Quiet anchors | Quiet S(0) | Active anchors | Active S(0) | Active - quiet [interval] |
| --- | --- | --- | --- | --- | --- |
| AAPL | 35016 | 0.02865239 [0.02444317, 0.03696890] | 50776 | 0.05479219 [0.04457193, 0.06820723] | 0.02613981 [0.01549437, 0.03458836] |
| AMZN | 24141 | 0.04088255 [0.02711496, 0.06419296] | 32757 | 0.07594153 [0.05316844, 0.10495622] | 0.03505898 [0.01882387, 0.04947838] |
| GOOG | 13466 | 0.02831816 [0.00741919, 0.05206864] | 12832 | 0.06696475 [0.04968032, 0.08540038] | 0.03864659 [0.01994441, 0.05572036] |
| INTC | 34235 | 0.04506911 [0.01587710, 0.11021751] | 76500 | 0.08897518 [0.05065862, 0.13871525] | 0.04390607 [-0.01574961, 0.07313425] |
| MSFT | 32851 | 0.10738063 [0.07362983, 0.14843470] | 95527 | 0.13665024 [0.11160814, 0.16220630] | 0.02926961 [-0.00312753, 0.05905988] |

HGB seed spread at zero event delay, reported separately from time-block uncertainty:

Hardware for this table: macOS arm64 execution; i5-13600K / RTX 4060 Ti 8 GB (planned timing platform; no hardware timing here).

| Ticker | Minimum seed S(0) | Maximum seed S(0) | Seed standard deviation |
| --- | --- | --- | --- |
| AAPL | 0.04490402 | 0.04710320 | 0.00094275 |
| AMZN | 0.05583289 | 0.05583289 | 0.00000000 |
| GOOG | 0.05150557 | 0.05150557 | 0.00000000 |
| INTC | 0.05220169 | 0.05420552 | 0.00089527 |
| MSFT | 0.12791004 | 0.13011406 | 0.00086185 |

## Artifacts and numerical provenance

- [manifest.json](../results/gate0/manifest.json): command, environment, configuration, source hashes, and elapsed time.
- [summary.json](../results/gate0/summary.json): complete ticker list, primary quantities, and the exact verdict rule result.
- [summary.md](../results/gate0/summary.md): complete model, latency, direct-check, burst, and seed-spread tables, with figures.
- [tests.log](../results/gate0/tests.log) and [run.log](../results/gate0/run.log): final executed test and experiment evidence.
- [AAPL/metrics.json](../results/gate0/AAPL/metrics.json): data and anchor counts, training priors, standardization, fit selection/calibration, scores, PMFs, paired intervals, and direct-check drops. Figure: [AAPL_skill.png](../results/gate0/figures/AAPL_skill.png).
- [AMZN/metrics.json](../results/gate0/AMZN/metrics.json): data and anchor counts, training priors, standardization, fit selection/calibration, scores, PMFs, paired intervals, and direct-check drops. Figure: [AMZN_skill.png](../results/gate0/figures/AMZN_skill.png).
- [GOOG/metrics.json](../results/gate0/GOOG/metrics.json): data and anchor counts, training priors, standardization, fit selection/calibration, scores, PMFs, paired intervals, and direct-check drops. Figure: [GOOG_skill.png](../results/gate0/figures/GOOG_skill.png).
- [INTC/metrics.json](../results/gate0/INTC/metrics.json): data and anchor counts, training priors, standardization, fit selection/calibration, scores, PMFs, paired intervals, and direct-check drops. Figure: [INTC_skill.png](../results/gate0/figures/INTC_skill.png).
- [MSFT/metrics.json](../results/gate0/MSFT/metrics.json): data and anchor counts, training priors, standardization, fit selection/calibration, scores, PMFs, paired intervals, and direct-check drops. Figure: [MSFT_skill.png](../results/gate0/figures/MSFT_skill.png).

Table provenance: count/configuration tables use `data`, `anchors`, and the manifest configuration; selection and primary tables use `gate` and `latency_adjusted`; delay-endpoint tables use `retrained` and `naive`; clock conversion uses `latency_distribution`; direct checks use `direct_checks`; burst and seed tables use `burst_split` and `hgb_seed_spread`. No report value is an extrapolated measurement or a hand-entered experiment result.

## Assumptions

- Split times use the files' own seconds-after-midnight clock, without timezone conversion. The study is a chronological split of a single historical day, not an across-day validation.
- Each retained equal-timestamp message group remains an event even if its final book returns to the preceding state. This follows the specified message-merging clock rather than removing groups with no net change.
- Decimal timestamp serialization tails are rounded to the nearest integer nanosecond directly from digits, using half-up rounding. All subsequent endpoint comparisons use integer nanoseconds.
- Invalid best-side mids keep their event indices but cannot be anchor/action references or label endpoints; the next valid changed mid supplies the label. Complete finite feature windows and defined labels at every grid delay are required.
- OFI includes the preceding book state needed for its first difference, so that state must also be inside the split. This conservatively removes an additional warmup event beyond the specified input window.
- The last execution message in a merged timestamp group determines that event's execution sign; the latest execution in the trailing indicator window determines the feature. Resting-order direction is negated.
- Feature scaling uses only the common training anchors, including for direct models after boundary drops. Depth imbalance includes occupied shares, with dummy levels contributing their zero volume.
- Unspecified estimator settings use the installed sklearn defaults; the material tree and logistic parameters and package versions are recorded in the manifest. Candidate selection uses uncalibrated validation loss before fitting temperature.
- Temperature optimization searches log(T) in `[-8.0, 8.0]`. The finite bounds approximate an unrestricted positive-temperature search and are part of this implementation.
- HGB statistics average independently calibrated seed losses, skills, and classification metrics. They are not scores of a probability ensemble. Validation model selection uses the seed-mean zero-delay skill.
- The naive model freezes both weights and temperature at zero delay. Its skill at each scored delay uses the same delay-specific training prior as the retrained comparison.
- Training priors are estimated separately for each event-delay label, and directly trained wall-clock models use their own priors. Priors, fitted models, standardization, and validation choices are held fixed in the test bootstrap. Burst subgroups keep the full-training priors.
- Complete overlapping bootstrap blocks start uniformly at eligible test anchors; each replicate draws enough full blocks to cover the observed duration after rounding up. Variable event counts and subgroup denominators are preserved. Intervals describe uncertainty under this within-day time-block resampling scheme.
- Global delay-specific skill is used for anchors realizing each wall-clock delay. Activity-dependent skill can violate that mixture approximation; direct checks and burst stratification assess the discrepancy.
- Skill is interpolated linearly across the event-delay grid and held constant beyond its last point. Beyond-grid mass is reported rather than treated as newly measured skill.
- Direct checks start with the common grid anchor population and drop anchors whose realized action label leaves its split. Paired mixture comparisons use the same surviving test anchors, with both skills and weights recomputed on each common resample.
- The prescribed per-ticker significance rule is applied without an added multiple-testing correction. The verdict is conditional on the fixed day, features, models, grids, and assumptions above.
- Wall-clock latencies are assumed scenarios. Inference timing, quantization losses, hardware acceleration benefits, across-day generalization, and trading profitability remain **unverified**.
