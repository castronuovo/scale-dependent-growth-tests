# Reproduction of the additional synthetic checks

This directory accompanies the revised manuscript *Testing a Single Physical
Infrared Scale in Cosmic Growth*. The parent directory contains
`reproducibility.zip`, the frozen benchmark and its results. Its original
README gives the code and data licenses and execution instructions.

Use Python with NumPy and Matplotlib. The additional calculations were run
with Python 3.9.6, NumPy 2.0.2, and Matplotlib 3.9.4.

```bash
MPLBACKEND=Agg python3 stress_test/run_stress_test.py
```

The script loads the released forward model from the sibling ZIP, checks the
nominal cuts against the archived results, and writes
`stress_test/stress_test_results.json`. The JSON records the source hashes,
settings, thresholds, counts, denominators, and intervals. The simulated
realizations are reproducible from the labels in the code and the frozen
benchmark's random-number generator.

`clean_validation` reports pointwise synthetic operating characteristics for
each fixed truth and decision rule. `redshift_smearing_sigma_v_km_s` holds the
fitting family, covariance, and cuts fixed and changes only the data-generating
mean. Wilson intervals condition on the realized cuts and each rate's
denominator. An empty denominator yields null. The paired drift comparison
reports the change in the raw fraction exiting at ST2 under the nominal rule.
All realizations enter ST2 in both of those cases, so that difference also
applies to conditional rejection. The figure script uses the frozen diagnostic results and the schematic
response, Green-function, and free-streaming formulas specified in the texts.

To recreate the updated figures in a copy of the submission source tree,
place `stress_test` and `reproducibility.zip` next to the `main`, `supplement`, and `ita`
directories and run:

```bash
MPLBACKEND=Agg python3 stress_test/render_revision_figures.py
```

## Partial ST2 entry and selective mass intervals

The supplementary archive `st2_selective_validation.zip` contains the
additional runner and results. Use Python 3.9.6, NumPy 2.0.2,
SciPy 1.13.1 and Matplotlib 3.9.4 to reproduce the numerical environment.
From the extracted archive's root, run:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 stress_test/run_selective_validation.py
```

The runner uses the archived nominal cuts. It selects partial-entry
constant-scale pairs in an independent pilot based on the first two
decisions and evaluates each pair with 20,000 new Gaussian draws. ST2
false rejection is counted among non-null, ST1-compatible draws. The
JSON records stage counts and simultaneous 95% Clopper–Pearson intervals
over the 16 tested pairs for the conditional false-rejection rate.

For mass inference, the report event also requires ST2 non-rejection.
Calibration uses the first 1,999 reports per truth on the nine-mass,
eleven-amplitude grid. For each mass, it takes the maximum over amplitude
of the cell-specific upper order statistic. The reported interval
spans the mass nodes retained by the profiled residual test. The JSON
contains the reference thresholds and a validation with the first
4,000 reports per grid truth from independent streams. Coverage
intervals quantify validation Monte Carlo uncertainty conditional on
the realized reference thresholds. The finite-grid calibration applies
to the exact recorded truth cells. Off-grid inclusion rates in the
boundary sample are separate diagnostics, conditional on final reports.

For one mock with the archived forward model, import
`selective_mass_interval` from `run_selective_validation.py`. Pass the
nominal `Benchmark` instance, its raw mock vector, and the
`thresholds` and `mass` entries from
`selective_validation_results.json`. The function returns closed
endpoints in h Mpc\(^{-1}\) on a raw final pass and `None` otherwise.
