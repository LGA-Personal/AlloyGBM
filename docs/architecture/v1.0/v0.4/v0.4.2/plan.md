# AlloyGBM v0.4.2 Plan (Histogram/Small-Workload Recovery + Split-Path Tuning)

## Summary
- Goal: execute the second `v0.4` child slice by reducing `build_histograms` small-workload overhead while preserving or improving the medium-workload gain established in `v0.4.1`, and tighten split-path benchmark coverage.
- Success criteria:
  - histogram kernel shows improved balance across small and medium benchmark fixtures,
  - correctness parity remains unchanged under deterministic fixtures,
  - full workspace and Python verification gates remain green.
- Audience: engineers iterating CPU backend kernel performance and reviewers validating scalar-path optimization evidence before SIMD introduction.

## Scope
### In Scope
- Additional scalar `build_histograms` optimization in `crates/backend_cpu/src/lib.rs` focused on reducing setup/conversion overhead for smaller feature-tile workloads.
- Split-path and histogram benchmark coverage expansion in `crates/backend_cpu/benches/histogram_kernels.rs` to better isolate:
  - small-workload overhead,
  - medium-workload throughput,
  - split-scan behavior under representative histogram sizes.
- Additive parity and determinism tests for any new accumulation strategy branching.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.4.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.4.2/verification_report.md`

### Out of Scope
- AVX2 intrinsics/runtime dispatch (reserved for `v0.4.3`).
- Multithreaded training-loop redesign.
- Objective or tree-policy changes.
- Predictor/model-format/API changes (`v0.5+` scope).
- External LightGBM benchmark suite onboarding (deferred to later `v0.4` closeout work).

## Interfaces and Types
- `crates/backend_cpu/src/lib.rs`:
  - `CpuBackend::build_histograms(...)` remains the optimization locus.
  - `best_split(...)` may receive localized scalar hot-path cleanups only if parity remains unchanged.
- `crates/backend_cpu/benches/histogram_kernels.rs`:
  - benchmark matrix expanded for clearer small-vs-medium signal quality.
- `crates/backend_cpu/src/lib.rs` tests:
  - parity assertions for histogram aggregates and split selection retained/expanded.

Backward-compatibility expectations:
- no public API or contract changes in `core`, `engine`, `predictor`, or Python bindings.
- deterministic behavior and artifact stability must remain unchanged under fixed seeds.

## Implementation Sequence
1. Expand benchmark matrix in `histogram_kernels` to capture small/medium histogram behavior with consistent same-run baseline/backend comparisons.
2. Profile scalar `build_histograms` overhead contributors from `v0.4.1` (tile-local buffer setup, conversion to `HistogramBin`, tile-loop bookkeeping).
3. Implement bounded optimization pass:
   - reduce per-call setup overhead for small workloads,
   - preserve medium-workload throughput,
   - keep behavior identical for all feature-tile layouts.
4. Add/extend parity tests for any strategy branching introduced by the optimization.
5. Run benchmark command and record relative deltas vs same-run baseline reference.
6. Run full verification gates and publish `implementation_notes.md` + `verification_report.md`.

## Test Cases and Scenarios
- Unit cases:
  - histogram aggregate invariants remain exact,
  - split-candidate behavior remains unchanged,
  - tile partition invariance remains true.
- Benchmark cases:
  - small histogram workload baseline vs backend,
  - medium histogram workload baseline vs backend,
  - split-scan workload (`best_split_medium`) stability.
- Integration cases:
  - workspace/unit/integration and Python binding suites pass unchanged.
- Failure and edge cases:
  - empty/invalid tile handling remains unchanged,
  - no new contract-violation behavior changes from optimization.
- Acceptance test mapping:
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `histogram_kernels` benchmark matrix is expanded and continues to report same-run baseline/backend deltas for histogram and split hot paths.
2. `build_histograms` receives an additive scalar optimization pass that preserves API/contract behavior.
3. Parity tests confirm no drift in histogram aggregates, split selection behavior, or tile partition invariance.
4. Benchmark evidence shows:
   - small workload regression from `v0.4.1` is materially reduced (no worse than `+10%` vs baseline reference),
   - medium workload retains meaningful improvement (at least `10%` faster than baseline reference).
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
10. Layer artifacts (`implementation_notes.md`, `verification_report.md`) are created with command outputs and delta table.

## Risks and Mitigations
- Risk: optimizing for small workloads erodes medium-workload gains.
  - Mitigation: require explicit dual-target benchmark thresholds in acceptance criteria.
- Risk: conditional scalar strategy introduces silent behavior drift.
  - Mitigation: enforce parity tests and tile-layout invariance checks as hard gates.
- Risk: benchmark variance obscures true deltas.
  - Mitigation: keep same-run baseline/backend comparison and fixed fixture generation.
- Risk: scope drift into SIMD/multithreading.
  - Mitigation: keep `v0.4.2` scalar-only and defer SIMD/threading to later slices.

## Assumptions and Defaults
- Device scope remains CPU scalar path only in this slice.
- Default benchmark command remains `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`.
- Baseline comparator remains the in-harness reference implementation for this slice.
- If threshold trade-offs are required, prioritize preserving medium-workload gain while eliminating severe small-workload regression.
