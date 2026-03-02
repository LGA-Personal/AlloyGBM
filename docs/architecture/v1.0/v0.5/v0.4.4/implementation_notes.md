# AlloyGBM v0.4.4 Implementation Notes

## Summary of What Was Built
- Updated backend SIMD-path internals in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - removed `unsafe` AVX2-target-feature implementation path that blocked `x86_64` target builds under workspace `unsafe_code` lint policy,
  - retained explicit runtime dispatch semantics via `HistogramKernelPath::RowFirstAvx2`,
  - kept `x86` chunked row-first route with safe Rust-only implementation.
- Added deterministic runtime AVX2 override in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `ALLOYGBM_DISABLE_AVX2` support for forced-scalar fallback during benchmark A/B runs,
  - one-time environment/capability resolution via `OnceLock` to avoid repeated probe overhead.
- Extended benchmark observability in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs):
  - emits `runtime_target_arch`,
  - emits `runtime_avx2_enabled`,
  - emits `runtime_avx2_override`.
- Added comparison automation script [benchmark_avx2_compare.sh](/Users/lashby/Projects/AlloyGBM/scripts/benchmark_avx2_compare.sh):
  - runs benchmark matrix in two modes (`default` and forced scalar),
  - executes repeated runs per mode,
  - computes median medium-workload delta.

## Non-Intuitive Decisions
- Decision: replace unsafe target-feature code with safe x86 chunked route while preserving AVX2 dispatch structure.
- Reason: workspace lint policy forbids unsafe code for target configurations where x86 path compiles, so portability had to be restored before meaningful x86 benchmarking.
- Impact: `x86_64` target benchmarks/tests now compile and execute; dispatch scaffolding remains intact for future SIMD-specialized tuning.

- Decision: add environment override for AVX2 route selection (`ALLOYGBM_DISABLE_AVX2`) and cache runtime probe state.
- Reason: AVX2-vs-scalar A/B benchmarking requires deterministic per-process route control without introducing public API churn.
- Impact: benchmark script can compare default path against forced scalar fallback on the same target command shape.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Backend changes are internal implementation and benchmark tooling only.
- Existing deterministic and contract behavior remains unchanged at interface level.

## Known Gaps Deferred to Next Layer
- On this machine, `x86_64` benchmarks run via target invocation but report `runtime_avx2_enabled: false`, so AVX2-on-hardware speedup evidence is still pending on an AVX2-capable native `x86_64` host.
- Parent `v0.5` rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/verification_report.md`

## Follow-Up Actions
- Execute `scripts/benchmark_avx2_compare.sh` on native AVX2-capable `x86_64` runner and record results for parent rollup.
- Decide whether further kernel tuning is required (`v0.4.5` style optional slice) or proceed directly to parent `v0.5` closeout.
- Update `docs/architecture/state/layer_index.yaml` to mark `v0.4.4` as `verified`.
