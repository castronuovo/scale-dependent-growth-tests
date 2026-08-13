# ST1–ST2 semi-numerical benchmark

This directory contains the reproducibility materials for the benchmark in
*Testing a Single Physical Infrared Scale in the Growth of Cosmic Structure:
A Conditional Classification Framework*.

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

Run the fixed publication contract from this directory:

```bash
python3 benchmark_st1_st2.py
```

The default command intentionally exposes no options that alter the physical
contract, bank, seeds, thresholds, or mock counts. Useful engineering options
are:

```bash
python3 benchmark_st1_st2.py --version
python3 benchmark_st1_st2.py --no-plots --quiet
python3 benchmark_st1_st2.py --verify-existing
```

`--verify-existing` does not rerun the mocks. It validates the stored JSON
schema and probability invariants, then checks every SHA-256 entry declared
below. The script also fails explicitly on non-finite responses, malformed
causal histories, inconsistent array dimensions, a non-positive covariance,
missing output epochs, or classification fractions that do not sum to unity.
The JSON file is replaced atomically only after strict serialization succeeds.

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

The frozen reproducibility target is
[GitHub release v1.0.5](https://github.com/castronuovo/scale-dependent-growth-tests/releases/tag/v1.0.5),
rather than the moving `main` branch. This release synchronizes the stored
machine-readable results and benchmark figures with the held-out non-rational
one-scale validation and the published SHA-256 manifest. Version `v1.0.5`
also records the code and content licenses and corrects the frozen-release
reference in this README; the scientific outputs are unchanged from `v1.0.3`.

## Expected SHA-256 hashes

```text
8dd1363cc468d6198208c58e4bbf0d6f7c19d976087b708e7cf5c8459da7d9e9  benchmark_st1_st2.py
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

## License

- `benchmark_st1_st2.py` is licensed under the
  [BSD 3-Clause License](LICENSE).
- The documentation, JSON output, and figures are licensed under
  [Creative Commons Attribution 4.0 International](LICENSE-CONTENT).

Copyright (c) 2026 Vitantonio Castronuovo.
