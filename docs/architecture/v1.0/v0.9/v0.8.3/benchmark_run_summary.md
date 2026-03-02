# AlloyGBM v0.8.3 Benchmark Run Summary

## Scope
- Layer: `docs/architecture/v1.0/v0.9/v0.8.3`
- Date: 2026-03-02
- Purpose: record executed benchmark commands and key runtime measurements for reproducibility evidence.

## Executed Benchmark Commands
1. `python3 -B benchmarks/dense_numeric/prepare.py --force-download --output-dir benchmarks/data/dense_numeric`
2. `python3 -B benchmarks/panel_time_series/prepare.py --force-download --max-rows 50000 --output-dir benchmarks/data/panel_time_series`
3. `python3 -B benchmarks/histogram_stress/prepare.py --rows 10 --features 4 --output-dir /tmp/alloy_hist_smoke`
4. `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
5. `bash scripts/benchmark_avx2_compare.sh --runs 1`
6. `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 80`

## Dataset Preparation Outcomes
- Dense numeric:
  - output: `benchmarks/data/dense_numeric/prepared/prepared.csv`
  - rows (including header): `1600`
- Panel time series:
  - output: `benchmarks/data/panel_time_series/prepared/prepared.csv`
  - rows (including header): `7025`
  - source switched to UCI Air Quality archive for a stable direct download endpoint.
- Histogram stress:
  - output: `/tmp/alloy_hist_smoke/prepared/prepared.csv`
  - smoke shape: 10 rows, 4 features, deterministic seed path.

## Key Measurements

### `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
- `runtime_target_arch: aarch64`
- `runtime_avx2_enabled: false`
- `histogram_build_tiny_backend: ns_per_iter=4629.35`
- `histogram_build_small_backend: ns_per_iter=30724.41`
- `histogram_build_medium_backend: ns_per_iter=1134514.59`
- `best_split_small: ns_per_iter=3311.00`
- `best_split_medium: ns_per_iter=112484.42`

### `scripts/benchmark_avx2_compare.sh --runs 1`
- `runtime_avx2_enabled(default): false`
- `runtime_avx2_enabled(forced_scalar): false`
- `runtime_avx2_override(default): unset`
- `runtime_avx2_override(forced_scalar): 1`
- `medium_ns_per_iter(default_runs): 1071879.18`
- `medium_ns_per_iter(forced_scalar_runs): 776020.82`
- `medium_ns_per_iter(default_median): 1071879.18`
- `medium_ns_per_iter(forced_scalar_median): 776020.82`
- `medium_delta_vs_forced_scalar_median: 38.13%`

### `run_model_comparison.py --force-prepare --rounds 80`
Output artifacts:
- `benchmarks/results/model_comparison_latest.csv`
- `benchmarks/results/model_comparison_latest.json`
- `benchmarks/results/model_comparison_latest.md`

Best RMSE by scenario:
- `dense_numeric`: `xgboost` (`rmse=0.548168`)
- `panel_time_series`: `lightgbm` (`rmse=4.723634`)
- `histogram_stress`: `lightgbm` (`rmse=0.366104`)

Fastest fit by scenario:
- `dense_numeric`: `alloygbm` (`fit_seconds=0.003681`)
- `panel_time_series`: `alloygbm` (`fit_seconds=0.007363`)
- `histogram_stress`: `alloygbm` (`fit_seconds=0.212366`)

## Notes
- All requested benchmark flows were executed in this pass, including live UCI-backed preparation for both dense and panel scenarios.
- Timing variance is expected in shared local environments; this layer captures reproducibility workflow and command evidence rather than fixed performance thresholds.
- Panel source currently uses UCI Air Quality archive (`00360/AirQualityUCI.zip`) for a stable direct-download endpoint.
