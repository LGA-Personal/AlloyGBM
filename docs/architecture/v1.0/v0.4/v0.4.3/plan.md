# AlloyGBM v0.4.3 Plan (AVX2 Runtime Dispatch + Scalar Fallback Validation)

## Summary
- Goal: execute the third `v0.4` child slice by introducing an explicit AVX2-dispatchable histogram path with deterministic scalar fallback, while preserving correctness parity and benchmark behavior established in `v0.4.2`.
- Success criteria:
  - runtime feature detection controls AVX2 path selection explicitly,
  - scalar fallback remains portable and correctness-equivalent,
  - benchmark and verification gates remain green.
- Audience: engineers implementing SIMD-ready backend dispatch and reviewers validating portability/correctness guardrails.

## Scope
### In Scope
- `crates/backend_cpu/src/lib.rs` updates for histogram kernel path selection:
  - explicit runtime AVX2 capability detection,
  - deterministic scalar fallback when AVX2 is unavailable,
  - AVX2-targeted row-first histogram builder route for eligible workloads.
- Additional backend tests to validate:
  - dispatch selection rules,
  - AVX2/scalar parity where AVX2 is available at runtime,
  - unchanged histogram and split correctness behavior.
- Benchmark evidence refresh in `crates/backend_cpu/benches/histogram_kernels.rs` command output.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.4.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.4.3/verification_report.md`

### Out of Scope
- AVX-512/NEON implementation.
- Multithreaded kernel redesign.
- Objective/tree-policy/model-format changes.
- Parent `v0.4` rollup closeout artifacts (deferred until remaining child work is complete).

## Interfaces and Types
- `crates/backend_cpu/src/lib.rs`:
  - introduce internal kernel-path selection and AVX2 runtime detection helpers,
  - preserve `BackendOps` public behavior and signatures.
- `crates/backend_cpu/benches/histogram_kernels.rs`:
  - retain benchmark matrix for tiny/small/medium histogram and split workloads.
- `crates/backend_cpu` tests:
  - expand with dispatch/fallback and parity checks.

Backward-compatibility expectations:
- no public API changes in `core`, `engine`, `predictor`, or Python bindings.
- deterministic semantics and existing validation errors remain stable.

## Implementation Sequence
1. Add internal histogram kernel path selector with deterministic thresholds and runtime AVX2 capability check.
2. Add AVX2-targeted row-first histogram route and keep scalar implementations intact.
3. Wire `build_histograms(...)` dispatch to select per-feature scalar, row-first scalar, or row-first AVX2 path.
4. Add/extend tests for dispatch decisions and AVX2-vs-scalar histogram parity.
5. Run benchmark and full verification gates; capture evidence in layer artifacts.

## Test Cases and Scenarios
- Unit cases:
  - small workloads stay on scalar per-feature path,
  - large workloads select AVX2 path only when runtime AVX2 is available,
  - large workloads fall back to scalar row-first when AVX2 is unavailable.
- Parity cases:
  - AVX2 row-first output matches scalar row-first output for equivalent fixtures (on AVX2-capable hosts).
  - existing histogram aggregate, split-candidate, and tile partition invariance tests remain green.
- Benchmark cases:
  - tiny/small/medium histogram baseline vs backend deltas,
  - split benchmark stability (`best_split_small`, `best_split_medium`).
- Acceptance test mapping:
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. Backend includes explicit runtime AVX2 capability detection and dispatch logic for histogram kernel path selection.
2. Scalar fallback behavior is implemented and validated for non-AVX2 environments.
3. Tests cover dispatch decision rules and AVX2/scalar row-first parity where runtime AVX2 is available.
4. Existing histogram/split correctness tests remain passing with no contract/API drift.
5. Benchmark evidence is captured in `verification_report.md` and does not violate `v0.4.2` performance guardrails:
   - small workload no worse than `+10%` vs baseline reference (median of repeated runs),
   - medium workload at least `10%` faster than baseline reference (median of repeated runs).
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: runtime dispatch introduces portability regressions.
  - Mitigation: architecture-gated code, explicit fallback tests, and full workspace verification.
- Risk: AVX2 route drifts from scalar behavior.
  - Mitigation: direct AVX2-vs-scalar parity test and existing histogram/split invariance suite.
- Risk: benchmark variance obscures regressions.
  - Mitigation: repeated benchmark runs and median-based threshold checks in verification artifact.

## Assumptions and Defaults
- AVX2 dispatch is implemented for `x86/x86_64` builds only; other targets always use scalar routes.
- Default benchmark command remains `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`.
- Public API remains unchanged in this layer.
