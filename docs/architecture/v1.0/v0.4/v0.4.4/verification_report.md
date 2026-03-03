# AlloyGBM v0.4.4 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.4/v0.4.4`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) Backend no longer relies on unsafe AVX2 implementation constructs that break `x86_64` target builds under workspace lint settings.
- Evidence:
  - [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs) contains no `unsafe` blocks/functions.
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels --target x86_64-apple-darwin` -> PASS (previous `unsafe_code` failure resolved).
  - `cargo test -p alloygbm-backend-cpu --target x86_64-apple-darwin` -> PASS (`13 passed`).
- Status: PASS

- Criterion: (2) Runtime AVX2 dispatch remains explicit and supports deterministic forced-scalar override via environment variable for benchmarking.
- Evidence:
  - Backend dispatch still uses `HistogramKernelPath` with AVX2 route selection in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs).
  - `ALLOYGBM_DISABLE_AVX2` override added and consumed via cached runtime probe in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs).
  - Benchmark output shows override state:
    - default run: `runtime_avx2_override: unset`
    - forced run: `runtime_avx2_override: 1`
- Status: PASS

- Criterion: (3) Benchmark output includes runtime AVX2 context signal for traceability.
- Evidence:
  - [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs) now prints:
    - `runtime_target_arch`
    - `runtime_avx2_enabled`
    - `runtime_avx2_override`
  - Verified in host and `x86_64` benchmark command outputs.
- Status: PASS

- Criterion: (4) Added AVX2-vs-scalar comparison script executes and reports deltas for `x86_64` target runs.
- Evidence:
  - Script added: [benchmark_avx2_compare.sh](/Users/lashby/Projects/AlloyGBM/scripts/benchmark_avx2_compare.sh).
  - Command: `bash scripts/benchmark_avx2_compare.sh --target x86_64-apple-darwin` -> PASS.
  - Summary output:
    - default medium runs: `508788.01`, `505186.46`, `501021.35`
    - forced scalar medium runs: `506727.60`, `499893.24`, `531251.56`
    - default median: `505186.46`
    - forced scalar median: `506727.60`
    - delta vs forced scalar median: `-0.30%`
- Status: PASS

- Criterion: (5) Existing histogram/split correctness tests remain passing with no API drift.
- Evidence:
  - `cargo test -p alloygbm-backend-cpu` -> PASS (`12 passed`).
  - `cargo test -p alloygbm-backend-cpu --target x86_64-apple-darwin` -> PASS (`13 passed`, includes x86-specific path-parity test).
  - No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Status: PASS

- Criterion: (6) `cargo fmt -- --check` passes.
- Evidence:
  - Command run in this verification pass -> PASS.
- Status: PASS

- Criterion: (7) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence:
  - Command run in this verification pass -> PASS.
- Status: PASS

- Criterion: (8) `cargo test --workspace` passes.
- Evidence:
  - Command run in this verification pass -> PASS.
- Status: PASS

- Criterion: (9) `cargo doc --workspace --no-deps` passes.
- Evidence:
  - Command run in this verification pass -> PASS.
- Status: PASS

- Criterion: (10) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence:
  - Command run in this verification pass -> PASS (`Ran 52 tests`, `OK`).
- Status: PASS

## Test Gap Mapping
- Gap check result:
  - Added x86 target verification path and AVX2 override benchmark comparison script evidence.
  - No acceptance-criteria gaps remained after command mapping.

## Tests Added or Updated
- Updated backend x86 route parity test in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `x86_row_first_histograms_match_scalar_when_avx2_supported`.

## Commands Executed
- `rustup target add x86_64-apple-darwin` -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo test -p alloygbm-backend-cpu` -> PASS
- `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` -> PASS
- `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels --target x86_64-apple-darwin` -> PASS
- `bash scripts/benchmark_avx2_compare.sh --target x86_64-apple-darwin` -> PASS
- `cargo test -p alloygbm-backend-cpu --target x86_64-apple-darwin` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- On this machine, `x86_64` benchmark runs report `runtime_avx2_enabled: false`; AVX2-on-hardware speedup characterization remains to be collected on a native AVX2-capable runner.
- Benchmark variance remains non-zero; script uses repeated runs and median comparison to reduce noise.

## Final Readiness
- Ready: Yes (`v0.4.4` acceptance criteria satisfied with explicit evidence mapping).
- Required follow-up before parent closeout: capture native AVX2-enabled comparison results or proceed with documented caveat in `v0.4` rollup.
