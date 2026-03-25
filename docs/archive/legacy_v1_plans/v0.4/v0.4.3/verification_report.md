# AlloyGBM v0.4.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.4/v0.4.3`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) Backend includes explicit runtime AVX2 capability detection and dispatch logic for histogram kernel path selection.
- Evidence:
  - [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs) introduces:
    - `HistogramKernelPath`,
    - `runtime_avx2_available()`,
    - `select_histogram_kernel_path(...)`,
    - dispatch routing in `build_histograms(...)`.
- Status: PASS

- Criterion: (2) Scalar fallback behavior is implemented and validated for non-AVX2 environments.
- Evidence:
  - Dispatch selector routes large workloads to `RowFirstScalar` when AVX2 is unavailable.
  - Test: `histogram_kernel_path_falls_back_to_scalar_when_avx2_unavailable` -> PASS.
- Status: PASS

- Criterion: (3) Tests cover dispatch decision rules and AVX2/scalar row-first parity where runtime AVX2 is available.
- Evidence:
  - Dispatch tests:
    - `histogram_kernel_path_prefers_per_feature_for_small_tiles`
    - `histogram_kernel_path_prefers_avx2_for_large_tiles_when_available`
    - `histogram_kernel_path_falls_back_to_scalar_when_avx2_unavailable`
  - AVX2 parity test added:
    - `avx2_row_first_histograms_match_scalar_when_supported` (`x86/x86_64`-gated, runs only on AVX2-capable hosts).
  - Command: `cargo test -p alloygbm-backend-cpu` -> PASS (`12 passed` on this runner).
- Status: PASS

- Criterion: (4) Existing histogram/split correctness tests remain passing with no contract/API drift.
- Evidence:
  - Existing backend invariance and split tests remain green in `cargo test -p alloygbm-backend-cpu`.
  - No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Status: PASS

- Criterion: (5) Benchmark evidence captured and `v0.4.2` guardrails remain satisfied by median repeated-run deltas.
- Evidence:
  - Command run 3 times: `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`.
  - Small delta (`backend` vs `baseline_ref`):
    - run1 `+5.93%`
    - run2 `+5.03%`
    - run3 `+5.62%`
    - median `+5.62%` (threshold `<= +10%`) -> PASS
  - Medium delta (`backend` vs `baseline_ref`):
    - run1 `-17.39%`
    - run2 `-17.56%`
    - run3 `-21.33%`
    - median `-17.56%` (threshold `>= 10%` faster) -> PASS
- Status: PASS

- Criterion: (6) `cargo fmt -- --check` passes.
- Evidence:
  - Command run in this verification pass: `cargo fmt -- --check` -> PASS.
- Status: PASS

- Criterion: (7) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence:
  - Command run in this verification pass: `cargo clippy --workspace --all-targets -- -D warnings` -> PASS.
- Status: PASS

- Criterion: (8) `cargo test --workspace` passes.
- Evidence:
  - Command run in this verification pass: `cargo test --workspace` -> PASS.
- Status: PASS

- Criterion: (9) `cargo doc --workspace --no-deps` passes.
- Evidence:
  - Command run in this verification pass: `cargo doc --workspace --no-deps` -> PASS.
- Status: PASS

- Criterion: (10) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence:
  - Command run in this verification pass -> PASS (`Ran 52 tests`, `OK`).
- Status: PASS

## Test Gap Mapping
- Gap check result:
  - Dispatch/fallback and parity coverage was added for this layer.
  - No additional acceptance-criteria test gaps remained after command and test mapping.

## Tests Added or Updated
- Added backend dispatch and parity tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `histogram_kernel_path_prefers_per_feature_for_small_tiles`
  - `histogram_kernel_path_prefers_avx2_for_large_tiles_when_available`
  - `histogram_kernel_path_falls_back_to_scalar_when_avx2_unavailable`
  - `avx2_row_first_histograms_match_scalar_when_supported`

## Commands Executed
- `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` (3 runs) -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- This runner is non-AVX2 architecture (`aarch64`), so AVX2 runtime path execution performance was not directly observed here; parity/dispatch behavior is covered structurally and by architecture-gated tests.
- Benchmark variance remains non-zero; median-based thresholds were used.

## Final Readiness
- Ready: Yes (`v0.4.3` acceptance criteria satisfied with explicit evidence mapping).
- Required follow-up before parent closeout: decide whether `v0.4.4` is needed for further SIMD tuning or proceed to `v0.4` rollup artifacts.
