# Latency-aware limit order book prediction with ternary-weight transformers

Does a small ternary-weight transformer running in ~100 ns on an FPGA keep more
predictive power at the moment it can act than a slower, more accurate CPU/GPU
model? This repo tests that by measuring how predictive skill decays as a
model's input is delayed by τ book events, converting τ to wall-clock time per
asset, and scoring each model at its own measured latency.

Status: research plan v1 written; awaiting approval before any code.

Hardware numbers come from bit-exact emulation (HGQ2 + hls4ml C-simulation) and
Vitis HLS synthesis reports, not physical boards.
