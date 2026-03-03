# AlloyGBM v0.4.3 Implementation Notes

## Summary of What Was Built
- Added explicit histogram kernel dispatch in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `HistogramKernelPath` (`PerFeatureScalar`, `RowFirstScalar`, `RowFirstAvx2`),
  - `runtime_avx2_available()` runtime capability detection for `x86/x86_64`,
  - `select_histogram_kernel_path(...)` dispatch selector used by `build_histograms(...)`.
- Added AVX2-targeted row-first route in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `build_tile_histograms_row_first_avx2(...)` with 8-row chunk processing under `#[target_feature(enable = "avx2")]`,
  - scalar remainder handling and shared histogram materialization.
- Refactored shared histogram materialization into:
  - `materialize_tile_histograms(...)`, reused by scalar and AVX2 row-first paths.
- Expanded backend tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `histogram_kernel_path_prefers_per_feature_for_small_tiles`,
  - `histogram_kernel_path_prefers_avx2_for_large_tiles_when_available`,
  - `histogram_kernel_path_falls_back_to_scalar_when_avx2_unavailable`,
  - `avx2_row_first_histograms_match_scalar_when_supported` (`x86/x86_64` only).

## Non-Intuitive Decisions
- Decision: keep AVX2 dispatch as runtime-selected but preserve scalar implementations as first-class paths.
- Reason: `v0.4` requires explicit SIMD readiness without sacrificing portability and deterministic behavior on non-AVX2 hosts/CI environments.
- Impact: backend behavior remains stable across architectures, while AVX2-capable `x86/x86_64` hosts can take a dedicated path.

- Decision: implement AVX2 path as chunked row-first accumulation with shared materialization logic.
- Reason: this keeps parity and maintenance risk low while introducing explicit SIMD-route structure and dispatch points needed for future intrinsics-heavy tuning.
- Impact: AVX2 and scalar routes stay behavior-equivalent, and follow-on tuning can focus on the AVX2 route in isolation.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Changes are internal to backend CPU histogram kernel routing and tests.
- Existing contract-validation and deterministic behavior expectations are preserved.

## Known Gaps Deferred to Next Layer
- AVX2 path was introduced with runtime dispatch and parity checks, but architecture-specific benchmark evidence on AVX2-capable hardware is still needed for stronger SIMD-only performance attribution.
- Optional `v0.4.4` remains available if additional SIMD tuning/benchmark stabilization is required.
- Parent `v0.4` rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/verification_report.md`

## Follow-Up Actions
- Use an AVX2-capable `x86_64` runner to collect dispatch-path-specific benchmark evidence.
- Proceed with next child slice only if further SIMD tuning is required; otherwise prepare `v0.4` parent rollup artifacts.
- Update `docs/architecture/state/layer_index.yaml` to mark `v0.4.3` as `verified`.
