# AlloyGBM v0.4.1 Implementation Notes

## Summary of What Was Built
- Added stable benchmark target wiring for backend CPU kernels in [Cargo.toml](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/Cargo.toml):
  - `[[bench]]`
  - `name = "histogram_kernels"`
  - `harness = false`
- Added deterministic benchmark harness in [histogram_kernels.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/benches/histogram_kernels.rs):
  - histogram-build benchmark for small and medium dense fixtures,
  - split-scan benchmark (`best_split_medium`),
  - same-run baseline reference path (`build_histograms_baseline_reference`) to reduce cross-run noise.
- Applied first low-risk `build_histograms` optimization in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - switched to row-first, tile-local accumulation buffers (`grad_sums`, `hess_sums`, `counts`) to avoid repeatedly loading gradients for every feature scan.
  - materializes per-feature histogram bins from contiguous tile-local accumulators after each tile pass.
- Added parity guard test in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs):
  - `build_histograms_is_tile_partition_invariant` ensures histogram outputs and split selection remain unchanged when equivalent feature tiles are partitioned differently.

## Non-Intuitive Decisions
- Decision: use an in-benchmark baseline reference implementation instead of comparing across separate benchmark runs only.
- Reason: short benchmark windows showed thermal/scheduling variance; same-run baseline/backend comparison gives tighter evidence for the optimization delta.
- Impact: verification artifacts include reproducible relative deltas under identical runtime conditions.

- Decision: keep optimization bounded to histogram kernel internals and avoid API or training-policy changes while still introducing row-first tile-local processing.
- Reason: `v0.4.1` requires measurable backend hot-path movement but must preserve deterministic correctness and contract behavior.
- Impact: correctness parity remained intact; benchmark results show a meaningful gain on the medium histogram workload while exposing a small-workload regression to revisit in `v0.4.2`.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No public API changes in `core`, `engine`, `predictor`, or Python bindings.
- Backend changes were internal to `CpuBackend::build_histograms(...)`.
- Benchmark surface was added as planned under `crates/backend_cpu/benches/`.

## Known Gaps Deferred to Next Layer
- AVX2/runtime SIMD dispatch remains deferred to later `v0.4.x` slices.
- Broader histogram/split-loop restructuring and memory-layout tuning remain deferred to `v0.4.2+`.
- Parent `v0.5` rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/verification_report.md`

## Follow-Up Actions
- Open `docs/architecture/v1.0/v0.5/v0.4.2/plan.md` for deeper histogram/split-path optimization.
- Keep `best_split` and histogram benchmark tracking in `histogram_kernels` for regression monitoring across `v0.4.x`.
- Update `docs/architecture/state/layer_index.yaml` to mark `docs/architecture/v1.0/v0.5/v0.4.1` as `verified`.
