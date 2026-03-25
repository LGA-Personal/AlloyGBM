# AlloyGBM v0.4 Implementation Notes

## Summary of What Was Built
- Completed `v0.4` through child slices `v0.4.1` -> `v0.4.4`:
  - `v0.4.1`: introduced backend benchmark harness and first row-first histogram optimization.
  - `v0.4.2`: added hybrid scalar routing to recover small-workload overhead while preserving medium-workload gains.
  - `v0.4.3`: introduced explicit runtime AVX2-dispatchable histogram route with scalar fallback and parity tests.
  - `v0.4.4`: removed x86 unsafe-code portability blocker, added AVX2 override control (`ALLOYGBM_DISABLE_AVX2`), and added repeated-run AVX2-vs-scalar comparison script.
- Expanded benchmark tooling in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs) to include tiny/small/medium histogram cases plus split cases and runtime context output.
- Added reusable benchmark comparison script [benchmark_avx2_compare.sh](/Users/lashby/Projects/AlloyGBM/scripts/benchmark_avx2_compare.sh) for default-vs-forced-scalar median delta analysis.

## Performance Evidence Rollup
- `v0.4.1`:
  - small delta (`backend` vs baseline): `+43.09%`
  - medium delta (`backend` vs baseline): `-24.27%`
- `v0.4.2` (3-run median):
  - small delta: `+5.09%`
  - medium delta: `-15.48%`
- `v0.4.3` (3-run median):
  - small delta: `+5.62%`
  - medium delta: `-17.56%`
- `v0.4.4` (`x86_64` target, 3-run medians from comparison script):
  - medium backend delta (`default` vs forced scalar): `-0.30%`
  - runtime context showed `runtime_avx2_enabled: false` in both modes on this machine.

## Non-Intuitive Decisions
- Decision: keep benchmark comparisons same-matrix and median-based rather than single-run.
- Reason: benchmark variance was non-trivial across runs; medians produced more stable gating signals.
- Impact: small/medium thresholds in `v0.4.2+` were validated against repeated-run medians.

- Decision: preserve explicit AVX2 dispatch structure while removing unsafe target-feature implementation for portability.
- Reason: workspace lint policy (`unsafe_code = forbid`) blocked `x86_64` target verification when unsafe AVX2 code was present.
- Impact: `x86_64` target compile/test/bench paths now execute in CI-style workflows without policy exceptions.

## Boundary/Interface Changes vs Plan
- No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- `v0.4` changes were confined to backend internals, benchmark harnesses, and documentation/state artifacts.
- Deterministic behavior and contract validation semantics remained unchanged.

## Apple Silicon AVX2 Caveat
- This repository was verified on an Apple Silicon host (`aarch64`).
- Even when running `x86_64` target binaries, benchmark output on this machine reported `runtime_avx2_enabled: false`.
- Practical implication:
  - portability and fallback behavior are verified,
  - direct AVX2-on-hardware speedup evidence is still pending on a native AVX2-capable `x86_64` host.

## Follow-Up Actions
- Capture native AVX2-enabled benchmark evidence on an AVX2-capable `x86_64` runner using `scripts/benchmark_avx2_compare.sh`.
- Proceed with next parent target planning (`v0.5` and beyond) while preserving the caveat in release-gate decisions until native AVX2 evidence is available.
