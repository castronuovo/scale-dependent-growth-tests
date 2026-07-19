# ST1–ST2 semi-numerical benchmark

This directory contains the reproducibility materials for the benchmark in
*Testing Single-Scale Growth in Large-Scale Structure: A Conditional
Classification Framework*.

The calculation is an indicative, semi-numerical validation of the declared
classification logic. It is not a catalogue analysis, a survey forecast, a
Boltzmann-code validation, or a nonlinear large-scale-structure pipeline.

## Environment

The supplied output was generated with:

- Python 3.9.6
- NumPy 2.0.2
- Matplotlib 3.9.4

Install the pinned Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Reproduction

Run from this directory:

```bash
python3 benchmark_st1_st2.py
```

The command overwrites the machine-readable result and creates four figures:

- `benchmark_st1_st2_results.json`
- `figs/st1_st2_end_to_end_benchmark.png`
- `figs/st1_st2_operating_diagnostics.png`
- `figs/st1_st2_end_to_end_benchmark_ita.png`
- `figs/st1_st2_operating_diagnostics_ita.png`

The fitting bank contains 128 fixed-seed causal histories. The ST1 threshold is
calibrated on the declared power-law cells and on 16 causal histories drawn
independently of that bank. A further 16 causal histories, generated with a
different seed, provide out-of-calibration validation. These tests establish
operating characteristics only for the finite sampled families; they do not
prove uniform coverage of a continuous function space.

The displayed evaluation set additionally contains a held-out monotonic
non-rational transition,
\(\Delta\mu=\beta[1-\exp(-(k/q)^2)]\), with
\((\beta,q)=(0.040,0.105a\,h\,{\rm Mpc}^{-1})\). It is absent from the fitting
bank and every threshold-calibration cell and has no assigned microscopic
interpretation. Under the fixed contract, 93.5% of its 400 mocks reach the
complete ST1--ST2-pass category, 6.5% fail ST2, and none fail ST1. This is one
off-template transition-shape validation, not coverage of arbitrary one-scale
responses.

The piecewise-causal histories use left-closed, right-open intervals (with the
final interval closed at the upper endpoint). The RK4 grid inserts every causal
edge and output epoch exactly and evaluates step endpoints one-sidedly within
the current interval. The JSON also records an 80--160--320 base-step
convergence check for representative constant-scale, evolving-scale,
non-rational, and discontinuous causal histories.

The manuscript cites the fixed
[GitHub release v1.0.1](https://github.com/castronuovo/scale-dependent-growth-tests/releases/tag/v1.0.1).
This versioned release, rather than the moving `main` branch, is the intended
reproducibility target.

## Expected SHA-256 hashes

```text
3abd68ebc48806f406001ceb03d34673928a2e29279acd421b284866623bdefc  benchmark_st1_st2.py
0e0569687e1c91ffbc838abba2ff968e03ab4bebb2f7860500be024e3b4483b7  benchmark_st1_st2_results.json
3fec6b9992caa9cec43f881c54e0b20d31031c06bd07c9a5d56a6c71a6568fbf  figs/st1_st2_end_to_end_benchmark.png
a716ca7c8be47fff7444af229c3e55f2aa2d74ac37b1a0e3ccbb54d27cc5e7ae  figs/st1_st2_operating_diagnostics.png
7f21ac2ed6908a70f30804100f6c723c03b3e71ab20db465a0883939a65aceb7  figs/st1_st2_end_to_end_benchmark_ita.png
f04202ff9612f51601dcbd335b46a3fac795aacc797514f9fec81ddebcd65ee3  figs/st1_st2_operating_diagnostics_ita.png
```

Small rendering differences may occur across Matplotlib or font-stack
versions; the JSON should remain byte-identical under the pinned numerical
environment.

## Citation

Please cite the manuscript above and the foundational theory article:

V. Castronuovo, *Scale-Dependent Growth of Large-Scale Structure from Infrared
Modifications of Gravity*, Ann. Phys. (Berlin) **538**, e70230 (2026),
<https://doi.org/10.1002/andp.70230>.
