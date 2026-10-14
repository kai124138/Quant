# Decisions (2026-10-13)

Full plan: research plan v1 (private document). Kai accepted every recommendation on the
12 open questions. This file is the in-repo record.

## Thesis and hypotheses
A small ternary-weight transformer on an FPGA (~100 ns) may keep more predictive power at the
moment it can act than a slower, more accurate CPU/GPU model. Falsifiable:
- H1: quantization costs little skill at zero delay (margin: 10% of FP32 skill).
- H2: skill decays measurably as the input is delayed by tau events.
- H3: a crossover latency exists where the fast model wins; it shrinks as event rate rises.

## The 12 decisions
| # | Decision | Adopted |
|---|---|---|
| 1 | H3 framing | Retrain per tau (latency-aware) primary; train-once-at-tau-0 ("naive") secondary |
| 2 | Data route | Free data now (LOBSTER 2012 samples, Nasdaq ITCH sample days); faculty access requested in parallel |
| 3 | Weight precision | Ternary (W2) primary; HGQ-heterogeneous variant as a third point; strict binary only via custom quantizer |
| 4 | Primary label | L-next: sign of the first nonzero mid-price change after the action event i+tau. Secondary L-h at h = 10, 50, 100 events |
| 5 | Event clock | Messages changing the top-10 visible book plus trades; messages sharing a timestamp merged into one event |
| 6 | H1 margin | 10% of FP32 skill, fixed before training |
| 7 | Pipeline overhead c | Swept: 1, 5, 40 microseconds |
| 8 | Assets | >= 3 stocks spanning tick size and event rate, chosen on training-day statistics |
| 9 | Crypto arm | Later, as a cross-asset check |
| 10 | Compute | Full tau sweep for small models; deep baseline reduced-size |
| 11 | Timing hardware | Home PC: i5-13600K and RTX 4060 Ti 8 GB, named in every table |
| 12 | Session trim | Drop the first 15 and last 10 minutes of each session |

## Core definitions
- Skill S_m(tau): information gain over the training class prior, in bits per prediction:
  S = H(prior) - mean cross-entropy (base 2) on test anchors. Temperature scaling fit on
  validation for every model before computing S.
- Label at action time: the model sees events up to i, acts at event i+tau, and is scored on
  the move after i+tau. Never on the move from i.
- tau_i(L): number of events with timestamp in (t_i, t_i + L].
- Latency-adjusted score A_m(L) = sum_tau P_hat(tau | L) S_m(tau); checked against direct
  retraining on wall-clock labels at 2-3 latencies.

## Constraints
- No synthesis and no use of the Duarte-lab machines: this is a separate project.
- Work so far: Gate 0 (quick H2 test on LOBSTER samples) and the theoretical foundations.
