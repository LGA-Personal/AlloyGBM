# Benchmark Regression Report (2026-03-03)

## Status
This report replaces the earlier provisional benchmark narrative and captures the full rerun results after:
- `v0.9.4` runtime provenance guards were added,
- the local editable `alloygbm` package was installed,
- full benchmark matrices were executed again top-to-bottom.

## Scope
Re-ran benchmark suite end-to-end and interpreted results against the current local runtime.

## Environment
- git commit: `b35903ce1ef4753dece58daaca55aa31d5b4dbb6`
- OS: `Darwin 24.6.0 arm64`
- Python: `3.12.10`
- Rust/Cargo: `1.92.0`
- Alloy runtime path:
  - Python module: `bindings/python/alloygbm/__init__.py`
  - native extension: `bindings/python/alloygbm/_alloygbm.abi3.so`

## Commands Executed
1. `python3 -m pip install -e .`
2. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29`
3. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`
4. `bash scripts/benchmark_avx2_compare.sh --runs 3`

## Produced Artifacts
- Full matrix (4 scenarios, 3 profiles, 3 seeds):
  - `benchmarks/results/model_comparison_20260303T094707Z.csv`
  - `benchmarks/results/model_comparison_20260303T094707Z.json`
  - `benchmarks/results/model_comparison_20260303T094707Z.md`
  - `benchmarks/results/model_comparison_profile_summary_20260303T094707Z.csv`
  - `benchmarks/results/model_comparison_profile_summary_20260303T094707Z.json`
- Ultra constrained run (2 scenarios, default+ultra profiles, seed 7):
  - `benchmarks/results/model_comparison_20260303T094739Z.csv`
  - `benchmarks/results/model_comparison_20260303T094739Z.json`
  - `benchmarks/results/model_comparison_20260303T094739Z.md`
  - `benchmarks/results/model_comparison_profile_summary_20260303T094739Z.csv`
  - `benchmarks/results/model_comparison_profile_summary_20260303T094739Z.json`

## Pass/Fail Summary
- Full matrix records (`20260303T094707Z`): `72 PASS / 36 FAIL`
  - `alloygbm`: `36 FAIL`
  - `lightgbm`: `36 PASS`
  - `xgboost`: `36 PASS`
- Ultra run records (`20260303T094739Z`): `16 PASS / 8 FAIL`
  - `alloygbm`: `8 FAIL`
  - `lightgbm`: `8 PASS`
  - `xgboost`: `8 PASS`

## Blocking Issue Observed
All Alloy benchmark training attempts failed with native input validation errors:
- `ValueError: row 0 feature 0 must be an integer-valued bin`
- `ValueError: row 0 feature 1 must be an integer-valued bin`

This is consistent with the current native training contract in `bindings/python/src/lib.rs`:
- features are required to be non-negative integer-valued bins,
- benchmark datasets currently provide continuous floating-point features.

## Interpretation
1. Runtime provenance hardening is working as designed:
   - stale runtime imports are blocked,
   - benchmark runner now executes against the intended local package.
2. The current blocker is no longer "wrong package loaded"; it is native trainer capability mismatch with continuous features.
3. Because all Alloy rows fail, no valid Alloy-vs-LightGBM speed/quality competitiveness conclusions can be drawn from this rerun.

## AVX2 Script Result
`bash scripts/benchmark_avx2_compare.sh --runs 3` summary:
- runtime target arch: `aarch64`
- runtime AVX2 enabled: `false`
- reported median delta: `n/a (runtime AVX2 unavailable)`

Interpretation: on this host, AVX2 acceleration cannot be validated and is not a useful competitiveness signal.

## Required Follow-up
1. Prioritize native training support for continuous features in upcoming `v0.9.x` slices.
2. Re-run full benchmark matrix only after continuous-feature support is available in Alloy native training path.
3. Preserve this report as the formal issue note: current Alloy benchmark failures are capability-bound, not harness-bound.
