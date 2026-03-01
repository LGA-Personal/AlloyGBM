# AlloyGBM v0.4.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.5/v0.4.2`
- Date: 2026-03-01

## Acceptance Criteria Matrix
- Criterion: (1) `histogram_kernels` benchmark matrix is expanded and continues to report same-run baseline/backend deltas for histogram and split hot paths.
- Evidence:
  - Benchmark harness in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs) includes:
    - `histogram_build_tiny_baseline_ref`
    - `histogram_build_tiny_backend`
    - `histogram_build_small_baseline_ref`
    - `histogram_build_small_backend`
    - `histogram_build_medium_baseline_ref`
    - `histogram_build_medium_backend`
    - `best_split_small`
    - `best_split_medium`
  - Command: `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` -> PASS.
- Status: PASS

- Criterion: (2) `build_histograms` receives an additive scalar optimization pass that preserves API/contract behavior.
- Evidence:
  - Hybrid scalar routing implemented in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
    - `build_tile_histograms_per_feature(...)`
    - `build_tile_histograms_row_first(...)`
    - workload routing via `SMALL_TILE_WORKLOAD_THRESHOLD`.
  - No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Status: PASS

- Criterion: (3) Parity tests confirm no drift in histogram aggregates, split selection behavior, or tile partition invariance.
- Evidence:
  - Backend tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
    - `build_histograms_aggregates_bins`
    - `best_split_returns_high_gain_candidate`
    - `build_histograms_is_tile_partition_invariant`
    - `histogram_tile_strategies_are_equivalent`
  - Command: `cargo test -p alloygbm-backend-cpu` -> PASS (`9 passed`).
- Status: PASS

- Criterion: (4) Benchmark evidence shows small regression no worse than `+10%` vs baseline and medium at least `10%` faster than baseline.
- Evidence:
  - Repeated benchmark command evidence (3 runs) from `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`:
    - small delta (`backend` vs `baseline_ref`): `+5.11%`, `-16.38%`, `+5.09%` (median `+5.09%`)
    - medium delta (`backend` vs `baseline_ref`): `-15.48%`, `-23.54%`, `-12.77%` (median `-15.48%`)
  - Threshold check:
    - small median `+5.09%` <= `+10%` threshold
    - medium median `-15.48%` >= `10%` faster threshold
- Status: PASS

- Criterion: (5) `cargo fmt -- --check` passes.
- Evidence:
  - Command run in this verification pass: `cargo fmt -- --check` -> PASS.
- Status: PASS

- Criterion: (6) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence:
  - Command run in this verification pass: `cargo clippy --workspace --all-targets -- -D warnings` -> PASS.
- Status: PASS

- Criterion: (7) `cargo test --workspace` passes.
- Evidence:
  - Command run in this verification pass: `cargo test --workspace` -> PASS.
- Status: PASS

- Criterion: (8) `cargo doc --workspace --no-deps` passes.
- Evidence:
  - Command run in this verification pass: `cargo doc --workspace --no-deps` -> PASS.
- Status: PASS

- Criterion: (9) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence:
  - Command run in this verification pass: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`).
- Status: PASS

- Criterion: (10) Layer artifacts are created with command outputs and delta table.
- Evidence:
  - [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.5/v0.4.2/implementation_notes.md) exists.
  - This verification report includes command outcomes and benchmark delta evidence.
- Status: PASS

## Test Gap Mapping
- Gap check result:
  - No additional test gaps remained after mapping criteria to existing backend tests and benchmark coverage.
  - Existing/new tests were sufficient to cover parity and strategy-equivalence criteria.

## Tests Added or Updated
- No new tests added in this verification pass (tests were added during `v0.4.2` implementation).

## Commands Executed
- `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Benchmark variance remains non-zero across runs; median-based evidence was used to reduce risk of single-run noise.
- SIMD dispatch and runtime feature-path verification remain deferred to `v0.4.3` by design.

## Final Readiness
- Ready: Yes (`v0.4.2` acceptance criteria satisfied with explicit evidence mapping).
- Required follow-up before merge/release: update layer state index and proceed to `v0.4.3` planning/implementation.
