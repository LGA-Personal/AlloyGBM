# Benchmark Regression Report (2026-03-03)

## Status
PASS for benchmark execution and runtime compatibility.  
Decision: keep `linear` as default; keep `rank` and `quantile` as configurable opt-in variants.

## Scope
- Added and benchmarked configurable Alloy continuous binning strategies (`linear`, `rank`).
- Implemented and benchmarked capped quantile binning (`quantile`, `max_bins` configurable).
- Ran full benchmark matrices for both strategies.
- Ran additional full matrices for quantile and a focused quantile bin-cap sweep.
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
6. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --alloy-continuous-binning-strategy quantile --alloy-continuous-binning-max-bins 256 --output-dir benchmarks/results/strategy_quantile`
7. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial --alloy-continuous-binning-strategy quantile --alloy-continuous-binning-max-bins 256 --output-dir benchmarks/results/strategy_quantile`
8. Focused sweep (`histogram_stress`, seed `7`, default profiles): `linear`, `quantile@256`, `quantile@128`, `quantile@64` with outputs in `benchmarks/results/quantile_bin_sweep/`

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
- `quantile` default matrix:
  - `benchmarks/results/strategy_quantile/model_comparison_20260303T190907Z.csv`
  - `benchmarks/results/strategy_quantile/model_comparison_profile_summary_20260303T190907Z.csv`
- `quantile` default_ultra matrix:
  - `benchmarks/results/strategy_quantile/model_comparison_20260303T190956Z.csv`
  - `benchmarks/results/strategy_quantile/model_comparison_profile_summary_20260303T190956Z.csv`
- Quantile bin-cap sweep:
  - `benchmarks/results/quantile_bin_sweep/model_comparison_profile_summary_20260303T191319Z.csv` (`linear`)
  - `benchmarks/results/quantile_bin_sweep/model_comparison_profile_summary_20260303T191559Z.csv` (`quantile@256`)
  - `benchmarks/results/quantile_bin_sweep/model_comparison_profile_summary_20260303T191839Z.csv` (`quantile@128`)
  - `benchmarks/results/quantile_bin_sweep/model_comparison_profile_summary_20260303T192110Z.csv` (`quantile@64`)
- AVX2 compare log:
  - `benchmarks/results/avx2_compare_20260303_rank_linear_eval.log`

## Pass/Fail Summary
- `linear` default: `108 PASS / 0 FAIL`
- `linear` default_ultra: `24 PASS / 0 FAIL`
- `rank` default: `108 PASS / 0 FAIL`
- `rank` default_ultra: `24 PASS / 0 FAIL`
- `quantile` default: `108 PASS / 0 FAIL`
- `quantile` default_ultra: `24 PASS / 0 FAIL`

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

## Strategy Delta (Quantile vs Linear, Alloy Only)
Regression threshold used for triage: `>5%` slowdown in fit/predict median.

### Default matrix (`4 scenarios x 3 profiles x 3 seeds`)
- Fit median delta: `+0.02%` (near parity), better in `5/12` scenario/profile cells.
- Predict median delta: `-0.08%` (near parity), better in `7/12` cells.
- RMSE median delta: `+3.05%` (mixed), better in `6/12` cells.
- Largest regression cluster: `histogram_stress` (`+9.78%` fit, `+1.47%` predict, `+16.76%` RMSE).

### Default_ultra matrix (`2 scenarios x 4 profiles x 1 seed`)
- Fit median delta: `-12.03%` (faster), better in `7/8` cells.
- Predict median delta: `-8.03%` (faster), better in `6/8` cells.
- RMSE median delta: `+0.08%` (near parity), better in `3/8` cells.

Interpretation: quantile mode improves speed in constrained ultra runs but is not robust on full-matrix accuracy due to histogram-stress degradation.

## Quantile Bin-Cap Sweep (`histogram_stress`, default profiles, seed `7`)
Average percent delta vs `linear` baseline:
- `quantile@256`: fit `+8.98%`, predict `+3.09%`, RMSE `+17.04%`
- `quantile@128`: fit `+3.19%`, predict `+18.77%`, RMSE `+34.18%`
- `quantile@64`: fit `-1.72%`, predict `+12.92%`, RMSE `+45.59%`

Interpretation: lower bin caps reduce/flatten fit cost in some profiles but materially harm regression quality on this stress scenario.

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
4. Keep `quantile` configurable for controlled experimentation, but do not promote it to default in `v0.9.7`.
5. Keep `linear` as baseline for now.

## Required Follow-up
1. Investigate per-feature adaptive quantile bins (or weighted sketch) instead of a single global bin cap.
2. Add calibration guardrails for quantile mode (for example, scenario-aware fallback to linear when RMSE regression exceeds threshold).
3. Re-run this exact matrix on a native AVX2-capable `x86_64` host for SIMD-path evidence.

---

## Candidate Experiment: Parallel Histogram Tiles (2026-03-04)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default backend improvement candidate (no behavior/accuracy contract change observed).

### Scope
- Implemented deterministic parallelization of CPU histogram building across feature tiles in `alloygbm-backend-cpu`.
- Added workload gate to avoid small-workload overhead.
- Preserved feature-histogram materialization order to keep split behavior deterministic.

### Commands Executed
1. Kernel microbench (before/after capture):
   - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels -- --nocapture`
2. Focused benchmark run (shallow profile, all scenarios, seed `7`):
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-par-tiles/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/tile_parallel_candidate_shallow`
3. A/B isolation with parallelism effectively disabled:
   - `RAYON_NUM_THREADS=1 PYTHONPATH=/tmp/alloygbm-bench-runtime-par-tiles/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/tile_parallel_candidate_shallow_rayon1`
4. Heavy stress profile spot check:
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-par-tiles/site-packages python3 -B benchmarks/run_model_comparison.py --profile mid_balanced:0.05:6:1200 --profile-seeds 7 --scenarios histogram_stress --output-dir benchmarks/results/tile_parallel_candidate_mid_hist`

### Produced Artifacts
- `benchmarks/results/tile_parallel_candidate_shallow/model_comparison_20260304T005348Z.csv`
- `benchmarks/results/tile_parallel_candidate_shallow_rayon1/model_comparison_20260304T005509Z.csv`
- `benchmarks/results/tile_parallel_candidate_mid_hist/model_comparison_20260304T010532Z.csv`

### Histogram Kernel Microbench Delta
`histogram_build_medium_backend` improved from ~`420,426 ns/iter` to ~`201,669 ns/iter` (about `52%` faster) on this host.

### Alloy A/B Delta (Parallel vs `RAYON_NUM_THREADS=1`, shallow profile)
Across `dense_numeric`, `panel_time_series`, `histogram_stress`, `dow_jones_financial`:
- Median fit delta: `-16.72%` (parallel faster).
- Largest fit improvement: `histogram_stress` `-38.87%`.
- RMSE/MAE unchanged in all compared cells.

### Heavy Scenario Spot Check
`histogram_stress` + `mid_balanced` (seed `7`):
- Alloy fit time: `198.584s` (parallel-enabled run).
- RMSE/MAE unchanged from prior baseline values.

### Notes
- This candidate targets training throughput only; it does not change split math or objective behavior.
- Current inference timings remain dominated by predictor-path behavior unrelated to this histogram candidate.
