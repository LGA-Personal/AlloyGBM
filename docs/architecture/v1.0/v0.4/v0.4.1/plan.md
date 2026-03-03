# AlloyGBM v0.4.1 Plan (v0.4 Baseline Bench + First Kernel Optimization Slice)

## Summary
- Goal: execute the first child step of `v0.4` by establishing reproducible CPU-kernel benchmark baselines and delivering the first low-risk histogram-kernel optimization in `backend_cpu` with strict correctness parity.
- Success criteria:
  - backend benchmark harness is added and runnable on stable Rust toolchain,
  - first `build_histograms` optimization improves hot-path work without changing semantics,
  - full workspace and Python verification gates remain green.
- Audience: engineers implementing `v0.4` performance slices and reviewers validating the benchmark-first optimization workflow.

## Scope
### In Scope
- Add benchmark infrastructure for `alloygbm-backend-cpu` hot paths (histogram build and split scan) using deterministic fixture data.
- Optimize `CpuBackend::build_histograms(...)` to reduce per-feature allocation overhead and tighten inner-loop behavior without changing outputs.
- Add parity-focused tests ensuring optimized kernel outputs match expected histogram/split invariants.
- Preserve current engine/wrapper API contracts and runtime behavior.
- Produce layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.4.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.4.1/verification_report.md`

### Out of Scope
- AVX2 intrinsics/dispatch (reserved for later `v0.3.x` slice).
- Multithreaded training-loop redesign.
- Objective-function or tree-growth policy changes.
- Predictor/model-format changes (`v0.5` scope).

## Interfaces and Types
- `crates/backend_cpu/Cargo.toml`:
  - benchmark dependencies and bench target declarations required for stable benchmark execution.
- `crates/backend_cpu/benches/`:
  - benchmark entry points for histogram and split hot paths with deterministic fixture setup.
- `crates/backend_cpu/src/lib.rs`:
  - targeted optimization changes within `CpuBackend::build_histograms(...)` only.
- `crates/backend_cpu/src/lib.rs` tests:
  - parity assertions for histogram aggregates and split behavior.

Backward-compatibility expectations:
- no public API changes in `core`, `engine`, `predictor`, or Python wrapper modules.
- benchmark additions and backend internal optimizations must be additive and correctness-preserving.

## Deliverables
1. Benchmark package:
  - bench target(s) in `crates/backend_cpu/benches/` for histogram-build and split-scan workloads,
  - deterministic benchmark fixture generation shared by bench cases.
2. Kernel optimization package:
  - targeted `build_histograms` refactor reducing avoidable allocations and loop overhead.
3. Verification package:
  - updated/additional backend tests proving correctness parity and deterministic behavior,
  - layer artifacts (`implementation_notes.md`, `verification_report.md`) with command and benchmark evidence.
4. State package:
  - `docs/architecture/state/layer_index.yaml` update after verification.

## Implementation Sequence
1. Add deterministic benchmark fixtures and initial benchmark harness for backend CPU hot paths.
2. Run benchmark command to capture pre-optimization baseline for verification evidence.
3. Implement first histogram-kernel optimization pass in `CpuBackend::build_histograms(...)`.
4. Add/update tests to lock parity on histogram aggregates, split selection behavior, and determinism.
5. Re-run benchmark command and record relative delta vs baseline.
6. Run full verification gates and publish layer artifacts.
7. Update `layer_index.yaml` for `v0.4.1` progression.

## Test Cases and Scenarios
- Unit cases:
  - existing histogram aggregate invariants remain exact,
  - split-candidate behavior remains unchanged for deterministic fixtures,
  - deterministic-training fixture tests remain stable.
- Benchmark cases:
  - histogram-build dense fixture,
  - split-scan dense fixture.
- Integration cases:
  - full workspace tests and Python runtime tests confirm no contract regression outside backend internals.
- Failure and edge cases:
  - benchmark harness handles small and medium fixture sizes without panics,
  - optimization path does not alter validation/error semantics.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`,
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels` (baseline + post-change).

## Acceptance Criteria
1. `alloygbm-backend-cpu` exposes runnable benchmark target `histogram_kernels` covering histogram and split hot paths.
2. `CpuBackend::build_histograms(...)` optimization is implemented with no API contract changes.
3. Backend correctness tests demonstrate parity for histogram counts/grad/hess aggregates and split selection behavior.
4. Benchmark evidence in `verification_report.md` records baseline and post-change results with explicit relative deltas.
5. Existing engine/predictor/wrapper tests continue passing unchanged.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: benchmark harness introduces noise that obscures actual deltas.
  - Mitigation: fixed fixture sizes, deterministic data generation, and same-runner baseline/post-change comparison.
- Risk: allocation reduction refactor introduces subtle correctness drift.
  - Mitigation: lock parity with explicit histogram/split fixture assertions before and after optimization.
- Risk: optimization work broadens into SIMD/multithreading prematurely.
  - Mitigation: keep `v0.4.1` scalar-only and defer SIMD/threading changes to later child slices.

## Assumptions and Defaults
- Benchmark command default: `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`.
- First optimization locus is `build_histograms`; `best_split` and `apply_split` remain behavior-only touch points unless required for parity fixes.
- Baseline vs post-change deltas are reported from the same local environment in this slice.

## Exit Condition
`v0.4.1` is complete when benchmark baseline + first kernel optimization are implemented, correctness parity is verified, gate commands pass, and layer/state artifacts are updated.
