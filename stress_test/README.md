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
