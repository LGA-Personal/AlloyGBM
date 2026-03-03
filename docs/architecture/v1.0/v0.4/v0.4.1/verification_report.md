# AlloyGBM v0.4.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.4/v0.4.1`
- Date: 2026-03-01

## Acceptance Criteria Matrix
- Criterion: (1) `alloygbm-backend-cpu` exposes runnable benchmark target `histogram_kernels` covering histogram and split hot paths.
- Evidence:
  - Bench target declaration in [Cargo.toml](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/Cargo.toml) via `[[bench]] name = "histogram_kernels"`.
  - Benchmark harness in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs) executes:
    - `histogram_build_small_baseline_ref`
    - `histogram_build_small_backend`
    - `histogram_build_medium_baseline_ref`
    - `histogram_build_medium_backend`
    - `best_split_medium`
- Status: PASS

- Criterion: (2) `CpuBackend::build_histograms(...)` optimization is implemented with no API contract changes.
- Evidence:
  - `build_histograms` in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs) now uses row-first tile-local accumulators (`grad_sums`, `hess_sums`, `counts`) and materializes per-feature bins from those buffers.
  - No public interface changes in `core`, `engine`, `predictor`, or Python bindings.
- Status: PASS

- Criterion: (3) Backend correctness tests demonstrate parity for histogram aggregates and split behavior.
- Evidence:
  - Existing tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
    - `build_histograms_aggregates_bins`
    - `best_split_returns_high_gain_candidate`
    - `apply_split_partitions_rows`
  - Added parity test:
    - `build_histograms_is_tile_partition_invariant`
- Status: PASS

- Criterion: (4) Benchmark evidence records baseline and post-change results with explicit relative deltas.
- Evidence:
  - Command: `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
  - Same-run baseline/post output:
    - `histogram_build_small_baseline_ref`: `22279.76 ns/iter`
    - `histogram_build_small_backend`: `31879.46 ns/iter`
    - `histogram_build_medium_baseline_ref`: `1153736.46 ns/iter`
    - `histogram_build_medium_backend`: `873758.34 ns/iter`
  - Relative delta (`backend` vs `baseline_ref`):
    - small: `+43.09%` (`+9599.70 ns/iter`)
    - medium: `-24.27%` (`-279978.12 ns/iter`)
- Status: PASS

- Criterion: (5) Existing engine/predictor/wrapper regression tests continue passing unchanged.
- Evidence:
  - `cargo test --workspace` PASS across backend/core/engine/predictor/python/shap crates.
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` PASS (`Ran 52 tests`, `OK`).
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
  - Command run in this verification pass: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- Added benchmark harness: [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs)
- Added backend parity test in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `build_histograms_is_tile_partition_invariant`

## Commands Executed
- `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`)

## Residual Uncovered Criteria
- None for `v0.4.1` scope.

## Residual Risks
- Performance delta is measured against an in-harness baseline reference, not yet against external LightGBM benchmarks (deferred to later `v0.4` slices/rollup).
- Small-workload histogram benchmark regressed while medium workload improved; next slice should tune tile-local setup overhead for smaller tiles.
- SIMD path introduction and runtime dispatch validation remain out of scope for this slice and are deferred to later `v0.3.x`.

## Final Readiness
- Ready: Yes (`v0.4.1` acceptance criteria satisfied).
- Required follow-up before merge/release: plan and execute next child slice (`v0.4.2`) or parent `v0.4` closeout sequence.
