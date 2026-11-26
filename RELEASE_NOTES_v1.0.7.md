# ST1–ST2 selective validation for the major revision

Version `v1.0.7` adds two fixed-seed synthetic studies to the reproducibility
materials for *Testing a Single Physical Infrared Scale in Cosmic Growth*.

- The partial-entry study reports ST2 false rejection conditional on non-null
  entry and ST1 compatibility across 16 constant-scale injection pairs.
- The selective inference study calibrates mass intervals conditional on the
  final pass on the exact 99-cell mass–amplitude grid. It also reports
  independent coverage validation and off-grid diagnostics.

The release asset `st2_selective_validation.zip` accompanies the major
revision. It includes the frozen benchmark, the earlier
revision checks and renderer, and the new runner and results in
`stress_test/run_selective_validation.py` and
`stress_test/selective_validation_results.json`. The new JSON includes the
pilot design, calibration thresholds, event counts, denominators, and Monte
Carlo intervals. Installation and execution instructions are in
`stress_test/README.md` and the repository README.

```text
2e6eb236fce472bf208ed8e12b67e9951031b38e1b5d4ecff69cf82cd0652ff8  st2_selective_validation.zip
```
