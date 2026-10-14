# Agent rules for this repository

Research project: latency-aware limit order book (LOB) prediction. The research plan is
summarized in `docs/decisions.md`. Read it before starting.

## Evidence
- Never state that something works, is correct, or is verified without the evidence: the
  command you ran and the numbers or log lines it produced. Otherwise write "unverified".
- Every number in a report must trace to a file under `results/` produced by code in this repo.
- List every assumption you make in the report you write, under a heading "Assumptions".

## Environment
- Python: `.venv/bin/python` (3.12). Run tests with `.venv/bin/python -m pytest -q`.
- Do not install packages and do not use the network. All data is already in `data/raw/`.
- Set `MPLCONFIGDIR=.cache/mpl` before anything that imports matplotlib.

## Scope limits
- No FPGA synthesis, no Vitis/Vivado, no hls4ml builds, no SSH to any machine.
- Do not run `git commit`, `git push`, or change branches. Leave changes in the working tree
  for review.
- Do not modify files outside the paths your task names, except to fix a bug you document.

## Style
- Small, readable modules with docstrings that state units (events, seconds, bits).
- Prefer vectorized numpy/polars; no premature abstraction.
- Fixed random seeds, recorded in outputs.
