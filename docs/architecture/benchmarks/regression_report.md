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

---

## Candidate Experiment: Predictor Tree Traversal Inference Path (2026-03-04)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default predictor inference path.

### Scope
- Replaced per-row per-stump ancestor validation (`HashMap` + path checks) with prebuilt per-tree node tables and direct tree-path traversal.
- Tree structures are built once at predictor load time and reused for all row predictions.
- No objective/split math changes; inference-only execution-path optimization.

### Commands Executed
1. Validation:
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
   - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Focused shallow profile benchmark (all scenarios, seed `7`):
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-opt/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/predictor_treepath_candidate_shallow`
3. Heavy scenario spot check:
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-opt/site-packages python3 -B benchmarks/run_model_comparison.py --profile mid_balanced:0.05:6:1200 --profile-seeds 7 --scenarios histogram_stress --output-dir benchmarks/results/predictor_treepath_candidate_mid_hist`

### Produced Artifacts
- `benchmarks/results/predictor_treepath_candidate_shallow/model_comparison_20260304T023959Z.csv`
- `benchmarks/results/predictor_treepath_candidate_mid_hist/model_comparison_20260304T024306Z.csv`

### Alloy Delta vs Prior Candidate Build (Shallow Profile)
Compared against `tile_parallel_candidate_shallow`:
- Median fit delta: `-2.58%` (small improvement).
- Median predict delta: `-97.78%` (major improvement).
- RMSE/MAE deltas: `0.00%` in all compared cells.

Per-scenario predict deltas:
- `dense_numeric`: `-97.84%`
- `panel_time_series`: `-98.23%`
- `histogram_stress`: `-97.71%`
- `dow_jones_financial`: `-97.46%`

### Heavy Scenario Spot Check
`histogram_stress` + `mid_balanced` (seed `7`) compared to prior candidate build:
- Fit delta: `-15.98%`
- Predict delta: `-99.50%`
- RMSE/MAE: unchanged

### Notes
- This directly addresses the current predictor bottleneck identified in prior runs.
- Accuracy parity held across all comparison runs.

---

## Candidate Experiment: Cache Parsed Predictor Per Fitted Model (2026-03-04)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default regressor inference optimization (large wins for repeated small-batch inference; no accuracy impact observed).

### Scope
- Added `NativePredictorHandle` Python binding to parse/load predictor once and reuse it across `predict()` calls.
- Updated `GBMRegressor.fit()` to cache a strict parsed predictor handle when available.
- Updated `GBMRegressor.predict()` to use cached handle first, with canonical bridge fallback on handle runtime failure.
- Added contract tests for both handle-fast-path and fallback behavior.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
   - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Runtime build for benchmark isolation:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-bench-runtime-predictor-cache/wheelhouse -q`
   - `python3 -m pip install --no-deps --no-cache-dir --target /tmp/alloygbm-bench-runtime-predictor-cache/site-packages <wheel>`
3. Focused benchmark run (shallow profile, all scenarios, seed `7`):
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-cache/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/predictor_handle_cache_candidate_shallow`
4. Heavy scenario spot check:
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-cache/site-packages python3 -B benchmarks/run_model_comparison.py --profile mid_balanced:0.05:6:1200 --profile-seeds 7 --scenarios histogram_stress --output-dir benchmarks/results/predictor_handle_cache_candidate_mid_hist`
5. Repeated-predict microbench A/B (cached handle vs canonical parse-each-call path):
   - Results captured in `benchmarks/results/predictor_handle_cache_candidate_microbench_20260304.json`

### Produced Artifacts
- `benchmarks/results/predictor_handle_cache_candidate_shallow/model_comparison_20260304T054157Z.csv`
- `benchmarks/results/predictor_handle_cache_candidate_mid_hist/model_comparison_20260304T054517Z.csv`
- `benchmarks/results/predictor_handle_cache_candidate_microbench_20260304.json`

### Alloy Delta vs Prior Candidate Build (Shallow Profile)
Compared against `predictor_treepath_candidate_shallow`:
- Median fit delta: `+3.46%` (noise-level regression in this single-seed run).
- Median predict delta: `-8.91%` (small improvement in one-shot benchmark path).
- RMSE/MAE deltas: `0.00%` in all compared cells.

Per-scenario predict deltas:
- `dense_numeric`: `-19.56%`
- `panel_time_series`: `+1.74%`
- `histogram_stress`: `+2.24%`
- `dow_jones_financial`: `-28.37%`

### Heavy Scenario Spot Check
`histogram_stress` + `mid_balanced` (seed `7`) compared to prior candidate build:
- Fit delta: `+9.54%`
- Predict delta: `-1.46%`
- RMSE/MAE: unchanged

### Repeated-Predict Microbench (A/B)
From `benchmarks/results/predictor_handle_cache_candidate_microbench_20260304.json`:
- `rows=4000` (`80` loops): `-0.29%` delta (near parity; traversal dominates).
- `rows=32` (`2000` loops): `-66.88%` delta (substantial improvement).
- `rows=1` (`4000` loops): `-98.61%` delta (very large improvement).

### Notes
- Full benchmark harness usually performs one predict per fit, so this candidate's value is underrepresented there.
- The improvement scales with repeated prediction calls per fitted model, especially for low-latency small-batch inference.
- Accuracy parity held across all benchmark comparisons in this run.

---

## Candidate Experiment: Predictor Batch Row Parallelization (2026-03-04)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default predictor-path improvement.

### Scope
- Added guarded Rayon row-parallel execution in `alloygbm-predictor` `predict_batch`.
- Added a workload gate to avoid parallel overhead on small batches.
- Preserved deterministic output ordering and existing input validation behavior.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
   - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Runtime build for benchmark isolation:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-bench-runtime-predictor-rowpar/wheelhouse -q`
   - `python3 -m pip install --no-deps --no-cache-dir --target /tmp/alloygbm-bench-runtime-predictor-rowpar/site-packages <wheel>`
3. Focused benchmark run (shallow profile, all scenarios, seed `7`):
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-rowpar/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/predictor_row_parallel_candidate_shallow`
4. Heavy scenario spot check:
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-predictor-rowpar/site-packages python3 -B benchmarks/run_model_comparison.py --profile mid_balanced:0.05:6:1200 --profile-seeds 7 --scenarios histogram_stress --output-dir benchmarks/results/predictor_row_parallel_candidate_mid_hist`

### Produced Artifacts
- `benchmarks/results/predictor_row_parallel_candidate_shallow/model_comparison_20260304T060246Z.csv`
- `benchmarks/results/predictor_row_parallel_candidate_mid_hist/model_comparison_20260304T060555Z.csv`

### Alloy Delta vs Prior Candidate Build
Compared against `predictor_handle_cache_candidate_*`:

Shallow profile (`4` scenarios, seed `7`):
- Median fit delta: `-2.37%`.
- Median predict delta: `-53.89%`.
- RMSE/MAE deltas: `0.00%` in all compared cells.

Per-scenario predict deltas:
- `dense_numeric`: `-57.90%`
- `panel_time_series`: `-61.86%`
- `histogram_stress`: `-49.88%`
- `dow_jones_financial`: `+8.24%`

Heavy spot check (`histogram_stress` + `mid_balanced`, seed `7`):
- Fit delta: `-5.98%`
- Predict delta: `-76.86%`
- RMSE/MAE: unchanged

### Notes
- This candidate materially improves one-shot benchmark predict latency, unlike parse-cache-only optimization.
- Accuracy parity held across all comparison runs.

---

## Candidate Experiment: Histogram Subtraction In Depth Expansion (2026-03-04)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default training-path improvement.

### Scope
- Updated engine depth expansion to:
  - build root histograms once,
  - build histograms only for the smaller child partition per accepted split,
  - derive sibling child histograms by subtracting from parent histograms,
  - carry child histograms forward to the next depth level.
- Added histogram-subtraction contract checks (feature/bin alignment and count underflow guards).
- Added unit coverage for subtraction correctness.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
   - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Runtime build for benchmark isolation:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-bench-runtime-histsub/wheelhouse -q`
   - `python3 -m pip install --no-deps --no-cache-dir --target /tmp/alloygbm-bench-runtime-histsub/site-packages <wheel>`
3. Focused benchmark run (shallow profile, all scenarios, seed `7`):
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-histsub/site-packages python3 -B benchmarks/run_model_comparison.py --profile shallow_high_lr:0.20:4:200 --profile-seeds 7 --output-dir benchmarks/results/hist_subtraction_candidate_shallow`
4. Heavy scenario spot check:
   - `PYTHONPATH=/tmp/alloygbm-bench-runtime-histsub/site-packages python3 -B benchmarks/run_model_comparison.py --profile mid_balanced:0.05:6:1200 --profile-seeds 7 --scenarios histogram_stress --output-dir benchmarks/results/hist_subtraction_candidate_mid_hist`

### Produced Artifacts
- `benchmarks/results/hist_subtraction_candidate_shallow/model_comparison_20260304T072310Z.csv`
- `benchmarks/results/hist_subtraction_candidate_mid_hist/model_comparison_20260304T072500Z.csv`

### Alloy Delta vs Prior Candidate Build
Compared against `predictor_row_parallel_candidate_*`:

Shallow profile (`4` scenarios, seed `7`):
- Median fit delta: `-30.49%`.
- Median predict delta: `+2.25%`.
- RMSE/MAE deltas: near zero (largest observed RMSE delta `+0.0077%`).

Per-scenario fit deltas:
- `dense_numeric`: `-20.50%`
- `panel_time_series`: `-42.29%`
- `histogram_stress`: `-40.48%`
- `dow_jones_financial`: `-9.44%`

Heavy spot check (`histogram_stress` + `mid_balanced`, seed `7`):
- Fit delta: `-46.30%`
- Predict delta: `+1.99%`
- RMSE delta: `-0.0114%`
- MAE delta: `-0.0015%`

### Notes
- This materially reduces training time in the current high-cost profiles.
- Predict-time impact is small and slightly regressive in this run; training gain dominates.
- Observed accuracy drift was negligible in this benchmark slice.

---

## Candidate Experiment: Predictor Repacked Tree Layout (2026-03-05)

### Status
PASS for compile/test/benchmark execution.  
Decision: reject candidate and revert changes.

### Scope
- Reworked predictor tree storage from sparse `Option` node vectors to repacked per-tree arrays (`feature`, `threshold`, `left_leaf`, `right_leaf`) with sentinel slots.
- Updated predictor row traversal loop to consume repacked arrays directly.
- No training objective/split/leaf math changes.

### Commands Executed
1. Validation:
   - `cargo fmt`
   - `cargo test -p alloygbm-predictor`
2. Full paired benchmark matrix:
   - Baseline (clean worktree at `41f00ac`):  
     `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate8_predictor_repack_baseline`
   - After candidate change:  
     `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate8_predictor_repack_after`

### Produced Artifacts
- Baseline:
  - `benchmarks/results/v097_candidate8_predictor_repack_baseline/model_comparison_20260305T030228Z.csv`
  - `benchmarks/results/v097_candidate8_predictor_repack_baseline/model_comparison_profile_summary_20260305T030228Z.csv`
- After:
  - `benchmarks/results/v097_candidate8_predictor_repack_after/model_comparison_20260305T031449Z.csv`
  - `benchmarks/results/v097_candidate8_predictor_repack_after/model_comparison_profile_summary_20260305T031449Z.csv`

### Alloy Delta vs Baseline (Full Matrix, 36 Alloy Runs)
- Fit sum delta: `+5.35%` (regression)
- Predict sum delta: `-6.00%` (improvement)
- RMSE/MAE/R2 deltas: `0.00%` (no behavior change)

Profile aggregates:
- `shallow_high_lr`: fit `-10.42%`, predict `-8.76%`
- `mid_balanced`: fit `-1.72%`, predict `-0.48%`
- `deep_low_lr`: fit `+7.03%`, predict `-6.80%`

### Rejection Rationale
- For current `v0.9.7` priorities, fit-time improvements are weighted higher than predict-time-only gains.
- Mid/deep training profiles dominate runtime cost and are the primary target for competitiveness progression.
- Because deep fit regressed (`+7.03%`) and full-matrix fit regressed (`+5.35%`), this candidate is not acceptable as default and was reverted.

---

## Candidate Experiment: Release-Hotloop Gradient Validation Gating (2026-03-05)

### Status
PASS for compile/test/benchmark execution.  
Decision: keep as default training-path improvement.

### Scope
- Kept initial fit-contract gradient validation unchanged.
- In iterative training hot-loop, switched to:
  - always validate gradient vector length,
  - run full per-element finite/hessian validation only in debug builds.
- No objective math, split scoring, or model serialization changes.

### Commands Executed
1. Validation:
   - `cargo fmt`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
2. Full benchmark matrix (after candidate):
   - `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate9_grad_validation_after`
3. Baseline comparison source:
   - `benchmarks/results/v097_candidate8_predictor_repack_baseline/model_comparison_20260305T030228Z.csv`
   - (same pre-candidate runtime code state; predictor-repack candidate had been reverted)

### Produced Artifacts
- `benchmarks/results/v097_candidate9_grad_validation_after/model_comparison_20260305T034000Z.csv`
- `benchmarks/results/v097_candidate9_grad_validation_after/model_comparison_profile_summary_20260305T034000Z.csv`

### Alloy Delta vs Baseline (Full Matrix, 36 Alloy Runs)
- Fit sum delta: `-8.71%` (improvement)
- Predict sum delta: `-11.16%` (improvement)
- RMSE/MAE/R2 deltas: `0.00%` (no behavior change)

Profile aggregates:
- `shallow_high_lr`: fit `-20.87%`, predict `-19.41%`
- `mid_balanced`: fit `-11.99%`, predict `-7.90%`
- `deep_low_lr`: fit `-7.80%`, predict `-11.37%`

### Notes
- This directly improves the mid/deep fit-time priority while preserving deterministic output metrics.
- Safety contract remains enforced in full during debug/test builds and at initial fit-contract validation in all builds.

---

## Candidate Experiment: High-Resolution Continuous Binning for Linear/Rank (2026-03-05)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: reject candidate and revert code changes.

### Scope
- Extended `continuous_binning_max_bins` handling so `linear`, `rank`, and `quantile` strategies all honor the configured cap.
- Raised supported bin-cap range to `2..4096` for controlled high-resolution experiments.
- Added Python contract tests to verify `linear` and `rank` quantization paths obey non-default bin caps.
- No engine/backend split math changes.

### Commands Executed
1. Validation:
   - `python3 -m unittest bindings.python.tests.test_regressor_contract`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate27/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate27/wheelhouse/*.whl`
3. Focused benchmark run (high-res candidate):
   - `python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.05:6:1200 --profile-seeds 7,17,29 --alloy-continuous-binning-max-bins 1024 --output-dir benchmarks/results/v097_candidate27_highres_bins_1024_focus_after`
4. Baseline comparison source:
   - `benchmarks/results/v097_candidate17_round_buffer_reuse_after/model_comparison_20260305T125838Z.csv` (filtered to same profiles/seeds/scenarios)

### Produced Artifacts
- `benchmarks/results/v097_candidate27_highres_bins_1024_focus_after/model_comparison_20260305T181214Z.csv`
- `benchmarks/results/v097_candidate27_highres_bins_1024_focus_after/model_comparison_profile_summary_20260305T181214Z.csv`

### Alloy Delta vs Candidate17 Baseline (Focused Slice, 24 Alloy Runs)
- Median fit delta: `+2621.92%` (major regression)
- Median predict delta: `+217.66%` (regression)
- Median RMSE delta: `-1.00%` (improvement)
- Median MAE delta: `-1.28%` (improvement)
- Median R2 delta: `+0.00417` (improvement)
- Wins: RMSE `15/24`, MAE `18/24`, R2 `15/24`

### Notes
- This candidate confirms a real accuracy-vs-speed tradeoff lever exists in bin resolution.
- Runtime cost at `1024` bins is too high for default use, especially on mid/deep fits.
- Candidate code changes were reverted; benchmark artifacts are retained for future reference if we revisit a lower-cost high-resolution path.

---

## Candidate Experiment: Leafwise Best-First Expansion (Env-Gated) (2026-03-05)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: reject candidate and revert code changes.

### Scope
- Added an experimental leafwise/best-first split expansion path in engine training, gated by `ALLOYGBM_EXPERIMENT_LEAFWISE`.
- Candidate built and benchmarked in strict A/B mode against same code/runtime with env flag off.
- No objective/split-score math changes; only node expansion order differed.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate28/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate28/wheelhouse/*.whl`
3. Baseline focused run (env off):
   - `python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate28_leafwise_focus_baseline`
4. Candidate focused run (env on):
   - `ALLOYGBM_EXPERIMENT_LEAFWISE=1 python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate28_leafwise_focus_after`

### Produced Artifacts
- `benchmarks/results/v097_candidate28_leafwise_focus_baseline/model_comparison_20260305T183004Z.csv`
- `benchmarks/results/v097_candidate28_leafwise_focus_after/model_comparison_20260305T183409Z.csv`

### Alloy Delta vs Baseline (Focused Slice, 24 Alloy Runs)
- Median fit delta: `+84.51%` (regression)
- Median predict delta: `+0.78%` (regression)
- Median RMSE delta: `0.00%` (no change)
- Median MAE delta: `0.00%` (no change)
- Median R2 delta: `0.00000` (no change)
- Wins: RMSE `0/24`, MAE `0/24`, R2 `0/24`

### Notes
- Candidate increased training cost materially without any measurable accuracy benefit.
- Engine code changes were reverted after benchmarking; only artifacts/reporting retained.

---

## Candidate Experiment: Tree-Semantics Delta Updates Per Node (2026-03-05)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: keep as default training-path accuracy improvement.

### Scope
- Updated per-round depth expansion updates to use tree-semantics deltas:
  - track each active node's current absolute leaf output,
  - compute child absolute outputs from gradient/hessian,
  - apply row updates as `child_output - parent_output` deltas.
- This makes multi-depth rounds behave like standard tree leaf replacement rather than additive path stacking of full child outputs.
- No histogram kernel, split gain formula, or predictor artifact contract changes.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
   - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate29/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate29/wheelhouse/*.whl`
3. Baseline comparison source (same profile slice and seeds):
   - `benchmarks/results/v097_candidate28_leafwise_focus_baseline/model_comparison_20260305T183004Z.csv`
4. Candidate run:
   - `python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate29_tree_semantics_focus_after`

### Produced Artifacts
- `benchmarks/results/v097_candidate29_tree_semantics_focus_after/model_comparison_20260305T193649Z.csv`
- `benchmarks/results/v097_candidate29_tree_semantics_focus_after/model_comparison_profile_summary_20260305T193649Z.csv`

### Alloy Delta vs Baseline (Focused Slice, 24 Alloy Runs)
- Median fit delta: `+3.32%`
- Median predict delta: `+1.88%`
- Median RMSE delta: `-5.03%`
- Median MAE delta: `-4.80%`
- Median R2 delta: `+0.01994`
- Wins: RMSE `18/24`, MAE `18/24`, R2 `18/24`

### Notes
- This is the first tested candidate in the recent sequence with consistent and material metric-quality improvement across the focused slice.
- Runtime regression is small relative to quality gain and aligns with current `v0.9.7` priority shift toward fitting quality.

---

## Candidate Experiment: Engine-Side Regularized Split Re-Scoring (Env-Gated) (2026-03-06)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: reject candidate and revert code changes.

### Scope
- Added engine-side split re-scoring path gated by:
  - `ALLOYGBM_EXPERIMENT_SPLIT_L2`
  - `ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS`
- Applied regularization during candidate selection and leaf denominator in trainer.
- Kept backend default split search as fallback.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate31/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate31/wheelhouse/*.whl`
3. Baseline focused run (env off):
   - `python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate31_regsplit_focus_baseline`
4. Candidate focused run (env on):
   - `ALLOYGBM_EXPERIMENT_SPLIT_L2=1.0 ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS=16 python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate31_regsplit_focus_after`

### Produced Artifacts
- `benchmarks/results/v097_candidate31_regsplit_focus_baseline/model_comparison_20260306T005619Z.csv`
- `benchmarks/results/v097_candidate31_regsplit_focus_after/model_comparison_20260306T005914Z.csv`

### Alloy Delta vs Baseline (Focused Slice, 24 Alloy Runs)
- Median fit delta: `+39.14%`
- Median predict delta: `+4.99%`
- Median RMSE delta: `-1.32%`
- Median MAE delta: `-0.20%`
- Median R2 delta: `+0.00534`
- Wins: RMSE `17/24`, MAE `13/24`, R2 `17/24`

### Rejection Rationale
- Quality improved, but fit-time regression is too large for the current v0.9.7 tradeoff target.
- Candidate was reverted before next attempt.

---

## Candidate Experiment: Backend-Integrated Regularized Split Scoring (Env-Gated) (2026-03-06)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: keep candidate code and continue from this baseline.

### Scope
- Added split-selection options to backend contract:
  - `l2_lambda`
  - `min_child_hessian`
- Implemented regularized gain directly in CPU backend split scan (single pass, no duplicate histogram walk).
- Wired trainer to pass env-controlled options:
  - `ALLOYGBM_EXPERIMENT_SPLIT_L2`
  - `ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS`
- Leaf value denominator now includes configured `l2_lambda`.
- Added backend tests covering:
  - gain shrinkage under L2 regularization,
  - split pruning under high `min_child_hessian`.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate32/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate32/wheelhouse/*.whl`
3. Quick tuning run (seed `7`):
   - Baseline: `benchmarks/results/v097_candidate32_tune_baseline/model_comparison_20260306T011442Z.csv`
   - `lambda=1,min_child_hess=0`: `benchmarks/results/v097_candidate32_tune_l2_1_h0/model_comparison_20260306T011537Z.csv`
   - `lambda=1,min_child_hess=8`: `benchmarks/results/v097_candidate32_tune_l2_1_h8/model_comparison_20260306T011626Z.csv`
4. Full focused A/B (selected setting `lambda=1,min_child_hess=0`):
   - Baseline:  
     `python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate32_regsplit_backend_focus_baseline`
   - Candidate:  
     `ALLOYGBM_EXPERIMENT_SPLIT_L2=1 ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS=0 python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile-seeds 7,17,29 --output-dir benchmarks/results/v097_candidate32_regsplit_backend_focus_after`

### Produced Artifacts
- Baseline: `benchmarks/results/v097_candidate32_regsplit_backend_focus_baseline/model_comparison_20260306T011932Z.csv`
- Candidate: `benchmarks/results/v097_candidate32_regsplit_backend_focus_after/model_comparison_20260306T012201Z.csv`

### Alloy Delta vs Baseline (Focused Slice, 24 Alloy Runs)
- Median fit delta: `+0.80%`
- Median predict delta: `+0.36%`
- Median RMSE delta: `-0.26%`
- Median MAE delta: `-0.06%`
- Median R2 delta: `+0.00119`
- Wins: RMSE `13/24`, MAE `12/24`, R2 `13/24`

### Notes
- This preserves the quality direction of regularization without the severe fit-time penalty seen in candidate31.
- Tuning indicated `min_child_hessian=8` reduced quality in this slice; `min_child_hessian=0` is currently preferred.

---

## Candidate Experiment: L1 + L2 Regularized Split/Leaf Scoring (Env-Gated) (2026-03-06)

### Status
PASS for implementation/tests/benchmark execution.  
Decision: keep candidate code and treat as accepted v0.9.7 improvement.

### Scope
- Extended split regularization with L1 term:
  - `ALLOYGBM_EXPERIMENT_SPLIT_L1`
  - paired with existing `ALLOYGBM_EXPERIMENT_SPLIT_L2` and `ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS`
- Applied L1 soft-thresholding in both:
  - backend split gain evaluation,
  - trainer leaf-value calculation.
- Added backend unit test for L1 gain shrink behavior.

### Commands Executed
1. Validation:
   - `cargo fmt --all`
   - `cargo test -p alloygbm-engine -p alloygbm-backend-cpu`
2. Runtime build/install:
   - `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --interpreter python3 --out /tmp/alloygbm-v097-candidate33/wheelhouse -q`
   - `python3 -m pip install --force-reinstall /tmp/alloygbm-v097-candidate33/wheelhouse/*.whl`
3. Focused A/B (`shallow_high_lr`, `mid_balanced`, seeds `7,17,29`):
   - Baseline:
     - `benchmarks/results/v097_candidate33_l1_reg_focus_baseline/model_comparison_20260306T013749Z.csv`
   - Candidate:
     - `ALLOYGBM_EXPERIMENT_SPLIT_L2=1 ALLOYGBM_EXPERIMENT_SPLIT_L1=0.1 ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS=0`
     - `benchmarks/results/v097_candidate33_l1_reg_focus_after/model_comparison_20260306T014027Z.csv`
4. Deep low-lr finance/time-series supplemental A/B (`deep_low_lr`, seeds `7,17,29`, scenarios `panel_time_series` + `dow_jones_financial`):
   - Baseline:
     - `benchmarks/results/v097_candidate33_l1_reg_deep_finance_baseline/model_comparison_20260306T022549Z.csv`
   - Candidate:
     - `benchmarks/results/v097_candidate33_l1_reg_deep_finance_after/model_comparison_20260306T023048Z.csv`

### Focused Slice Delta (24 Alloy Runs)
- Median fit delta: `-0.37%`
- Median predict delta: `+0.76%`
- Median RMSE delta: `-0.88%`
- Median MAE delta: `-0.29%`
- Median R2 delta: `+0.00419`
- Wins: RMSE `15/24`, MAE `16/24`, R2 `15/24`

### Deep Low-LR Supplemental Delta (6 Alloy Runs)
- Median fit delta: `-20.57%`
- Median predict delta: `-15.14%`
- Median RMSE delta: `+0.08%`
- Median MAE delta: `-1.87%`
- Median R2 delta: `-0.00088` (sub-0.5% absolute effect)

### Acceptance Note
- For the deep low-SNR slice, this is accepted as a strong tradeoff: ~20% fit-time reduction with very small overall quality movement (sub-0.5% R2 effect, slight RMSE increase, MAE improvement).
