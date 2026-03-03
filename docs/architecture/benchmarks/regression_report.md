# Benchmark Regression Report (2026-03-03)

## Status
PASS for benchmark execution and runtime compatibility.  
Decision: keep `linear` as default; keep `rank` as configurable opt-in variant.

## Scope
- Added and benchmarked configurable Alloy continuous binning strategies (`linear`, `rank`).
- Ran full benchmark matrices for both strategies.
- Compared strategy deltas and competitiveness against `lightgbm` and `xgboost`.

## Environment
- git commit: `6b912650969987a06b99aa72b48921acdefc4338`
- OS: `Darwin 24.6.0 arm64`
- Python: `3.12.10`
- Rust/Cargo: `1.92.0`
- Alloy runtime path:
  - Python module: `bindings/python/alloygbm/__init__.py`
  - native extension: `bindings/python/alloygbm/_alloygbm.abi3.so`

## Commands Executed
1. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --alloy-continuous-binning-strategy linear --output-dir benchmarks/results/strategy_linear`
2. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial --alloy-continuous-binning-strategy linear --output-dir benchmarks/results/strategy_linear`
3. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --alloy-continuous-binning-strategy rank --output-dir benchmarks/results/strategy_rank`
4. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial --alloy-continuous-binning-strategy rank --output-dir benchmarks/results/strategy_rank`
5. `bash scripts/benchmark_avx2_compare.sh --runs 3 > benchmarks/results/avx2_compare_20260303_rank_linear_eval.log 2>&1`

## Produced Artifacts
- `linear` default matrix:
  - `benchmarks/results/strategy_linear/model_comparison_20260303T183621Z.csv`
  - `benchmarks/results/strategy_linear/model_comparison_profile_summary_20260303T183621Z.csv`
- `linear` default_ultra matrix:
  - `benchmarks/results/strategy_linear/model_comparison_20260303T183706Z.csv`
  - `benchmarks/results/strategy_linear/model_comparison_profile_summary_20260303T183706Z.csv`
- `rank` default matrix:
  - `benchmarks/results/strategy_rank/model_comparison_20260303T184847Z.csv`
  - `benchmarks/results/strategy_rank/model_comparison_profile_summary_20260303T184847Z.csv`
- `rank` default_ultra matrix:
  - `benchmarks/results/strategy_rank/model_comparison_20260303T184935Z.csv`
  - `benchmarks/results/strategy_rank/model_comparison_profile_summary_20260303T184935Z.csv`
- AVX2 compare log:
  - `benchmarks/results/avx2_compare_20260303_rank_linear_eval.log`

## Pass/Fail Summary
- `linear` default: `108 PASS / 0 FAIL`
- `linear` default_ultra: `24 PASS / 0 FAIL`
- `rank` default: `108 PASS / 0 FAIL`
- `rank` default_ultra: `24 PASS / 0 FAIL`

## Strategy Delta (Rank vs Linear, Alloy Only)
Regression threshold used for triage: `>5%` slowdown in fit/predict median.

### Default matrix (`4 scenarios x 3 profiles x 3 seeds`)
- Fit median delta: `+11.61%` (rank slower), better in `2/12` scenario/profile cells.
- Predict median delta: `+36.17%` (rank slower), better in `1/12` cells.
- RMSE median delta: `+1.13%` (mixed), better in `6/12` cells.
- MAE median delta: `-2.17%` (slight improvement), better in `7/12` cells.
- Largest regression cluster: `histogram_stress` (`+31.90%` fit, `+81.61%` predict, `+6.94%` RMSE).

### Default_ultra matrix (`2 scenarios x 4 profiles x 1 seed`)
- Fit median delta: `-9.23%` (rank faster), better in `8/8` cells.
- Predict median delta: `-4.04%` (rank slightly faster), better in `5/8` cells.
- RMSE median delta: `+0.05%` (near parity), better in `4/8` cells.

Interpretation: `rank` is unstable across the full matrix and regresses heavily in the histogram-heavy scenario despite occasional wins.

## Competitiveness Snapshot
Using the full default matrix:
- Best RMSE by scenario/profile is still dominated by `xgboost` (`11/12` cells; `lightgbm` wins `1/12`; `alloygbm` wins `0/12`).
- Fastest fit is `lightgbm` in all `12/12` cells.
- Fastest predict is `xgboost` in all `12/12` cells.

Using default_ultra:
- Best RMSE remains `xgboost` (`8/8` cells).
- Fastest fit remains `lightgbm` (`8/8` cells).
- Fastest predict remains `xgboost` (`8/8` cells).

## AVX2 Script Result
`scripts/benchmark_avx2_compare.sh --runs 3` summary:
- runtime target arch: `aarch64`
- runtime AVX2 enabled (default): `false`
- runtime AVX2 enabled (forced scalar): `false`
- median delta: `n/a (runtime AVX2 unavailable)`

Interpretation: AVX2 tuning cannot be validated on this host; performance conclusions should rely on non-AVX2 metrics here.

## Keep/Drop Decision
1. Keep `continuous_binning_strategy` as a configurable variant (`linear` default, `rank` optional).
2. Keep benchmark-runner configurability (`--alloy-continuous-binning-strategy`) for repeatable A/B testing.
3. Do not promote `rank` to default in `v0.9.7` due to full-matrix speed regressions above threshold.

## Required Follow-up
1. Implement true capped quantile histogram binning (for example, fixed `<=256` bins) to reduce `rank` lookup overhead while preserving distribution awareness.
2. Add optional strategy-specific calibration knobs (for example, quantile bin count / sketch epsilon) behind config flags.
3. Re-run this exact matrix on a native AVX2-capable `x86_64` host for SIMD-path evidence.
