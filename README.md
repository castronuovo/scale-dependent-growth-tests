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

The piecewise-causal histories use left-closed, right-open intervals (with the
final interval closed at the upper endpoint). The RK4 grid inserts every causal
edge and output epoch exactly and evaluates step endpoints one-sidedly within
the current interval. The JSON also records an 80--160--320 base-step
convergence check for representative constant-scale, evolving-scale, and
discontinuous causal histories.

The manuscript will cite a fixed GitHub release URL once the public repository
and its versioned release have been created. The release, rather than the
moving `main` branch, is the intended reproducibility target.

## Expected SHA-256 hashes

```text
8f15403a4abaab46ef9ca8f6a0ec386f47459516f5b60a41c692064782a4b9e4  benchmark_st1_st2.py
201de495e6a5b549bebfe07155246d635fc26509165d6099edaec554a149439c  benchmark_st1_st2_results.json
dc2d296b5c1b67a3f4b5500748659656dfcf3351caa4134785a10651da90dd09  figs/st1_st2_end_to_end_benchmark.png
a716ca7c8be47fff7444af229c3e55f2aa2d74ac37b1a0e3ccbb54d27cc5e7ae  figs/st1_st2_operating_diagnostics.png
e5eff9b31e04c8c24d278f189b60f1d31a3175d645ea6dcd2df8d7c85db0fba0  figs/st1_st2_end_to_end_benchmark_ita.png
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
