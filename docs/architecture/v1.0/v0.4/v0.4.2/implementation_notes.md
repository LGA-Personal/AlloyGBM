# AlloyGBM v0.4.2 Implementation Notes

## Summary of What Was Built
- Implemented a hybrid scalar histogram strategy in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - added `build_tile_histograms_per_feature(...)` for smaller tile workloads,
  - kept `build_tile_histograms_row_first(...)` for larger workloads,
  - added tile-workload routing in `build_histograms(...)` with `SMALL_TILE_WORKLOAD_THRESHOLD`.
- Expanded benchmark matrix in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs):
  - added tiny workload baseline/backend comparison,
  - kept small and medium baseline/backend comparisons,
  - added `best_split_small` alongside `best_split_medium`.
- Added parity test in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `histogram_tile_strategies_are_equivalent` verifies per-feature and row-first tile accumulation produce identical histogram outputs.
- Ran required validation commands and benchmark command from `v0.4.2` plan; all gates passed.

## Non-Intuitive Decisions
- Decision: route histogram strategy by tile workload rather than globally using a single accumulation approach.
- Reason: `v0.4.1` evidence showed medium workload wins with row-first accumulation but small-workload overhead regressions; a hybrid route retains medium gains while reducing small-workload penalties.
- Impact: latest benchmark run shows small regression reduced to `+3.45%` vs baseline and medium remains improved at `-13.05%` vs baseline.

- Decision: keep strategy threshold as a fixed constant (`SMALL_TILE_WORKLOAD_THRESHOLD`) in `v0.4.2`.
- Reason: bounded tuning keeps this layer decision-complete without introducing additional runtime/config surface.
- Impact: behavior is deterministic and easy to audit; threshold sensitivity tuning can be revisited in later slices if needed.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Changes are confined to backend scalar kernel internals and backend benchmark harness coverage.
- Deterministic training behavior and existing contracts remain unchanged.

## Known Gaps Deferred to Next Layer
- `v0.4.2` verification artifact remains to be produced:
  - `docs/architecture/v1.0/v0.4/v0.4.2/verification_report.md`
- SIMD/runtime feature dispatch remains deferred to `v0.4.3`.
- External dataset/LightGBM comparative benchmarking remains deferred to later `v0.4` closeout work.

## Follow-Up Actions
- Run layer verification workflow and publish `v0.4.2/verification_report.md` with acceptance-criteria mapping and benchmark delta table.
- If benchmark variance grows, increase iteration counts or add repeated-run summary reporting inside `histogram_kernels`.
- Proceed to `v0.4.3` planning for AVX2 runtime dispatch and scalar fallback validation.
