# AlloyGBM v0.5 Technical Plan

## Summary
- Goal: deliver the `0.5.0` milestone by improving CPU training-kernel performance and introducing SIMD acceleration readiness on top of verified `v0.4` evaluation/validation tooling.
- Success criteria:
  - measurable CPU hot-path improvements are delivered without changing model correctness,
  - SIMD acceleration is introduced behind explicit runtime capability checks with scalar fallback,
  - performance work is reproducible through deterministic fixtures and benchmark evidence captured in verification artifacts.
- Audience: engineers implementing `v0.5` child slices and reviewers gating progression toward `v0.6` model IO and predictor integration.

## Scope
### In Scope
- CPU hot-path optimization work in `crates/backend_cpu` and minimally required `crates/engine` call paths:
  - histogram-build and split-scan loop efficiency,
  - allocation reduction and memory-access improvements,
  - deterministic behavior parity under fixed seeds.
- SIMD readiness and rollout for CPU kernels:
  - AVX2-optimized path where available,
  - scalar fallback for unsupported CPUs and CI portability.
- Reproducible benchmark harnesses for targeted dense workloads used to track improvement across slices.
- Child-layer decomposition and state tracking under `v0.5` using `v0.4.x` slices.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/verification_report.md`

### Out of Scope
- CUDA/Metal/MLX backend implementation.
- New objectives (ranking), SHAP algorithm expansion, or categorical pipeline expansion.
- Model-format/predictor compatibility policy changes (`v0.6+` scope).
- Python API redesign beyond compatibility-preserving behavior needed for regression verification.

## Interfaces and Types
- `crates/backend_cpu/src/lib.rs`:
  - primary locus for histogram/split/reduction kernel optimization and SIMD dispatch integration.
- `crates/engine/src/lib.rs`:
  - only minimal integration changes needed to consume optimized backend behavior.
- `crates/backend_cpu/benches/`:
  - reproducible performance measurements for representative kernel workloads.
- `bindings/python/tests/`:
  - regression guard that wrapper/runtime behavior remains unchanged by kernel optimization work.

Backward-compatibility expectations:
- no breaking changes to `GBMRegressor` public API or existing metric/split helper signatures from `v0.4`.
- deterministic training semantics and artifact compatibility behavior remain stable unless explicitly changed by a child-layer plan.

## Implementation Sequence
1. Execute `docs/architecture/v1.0/v0.5/v0.4.1/plan.md` to establish benchmark baseline and first low-risk kernel optimization slice.
2. Open next child slice (`v0.4.2`) for broader histogram/split-path optimization with parity coverage and memory-access tuning.
3. Open next child slice (`v0.4.3`) for AVX2 path introduction and runtime dispatch/fallback validation.
4. Open optional child slice (`v0.4.4`, if needed) for residual performance gaps, benchmark stabilization, and closeout polish.
5. Close parent `v0.5` with rollup notes, verification report, and `docs/architecture/state/layer_index.yaml` update.

## Test Cases and Scenarios
- Unit cases:
  - deterministic histogram/split/reduction parity assertions before and after optimization changes,
  - SIMD-dispatch behavior checks (AVX2 path selected only when supported).
- Integration cases:
  - `Trainer` and `GBMRegressor` end-to-end predictions remain stable under deterministic fixtures.
- Performance evidence:
  - benchmark suite captures baseline and post-change metrics for representative dense workloads,
  - each child verification report includes command, environment summary, and relative delta table.
- Failure and edge cases:
  - unsupported CPU feature environments correctly use scalar fallback,
  - no regressions in contract-validation errors.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`,
  - targeted benchmark commands defined in each child slice.

## Acceptance Criteria
1. `v0.5` child slices collectively deliver benchmark harness coverage for backend CPU histogram/split hot paths with reproducible fixture definitions.
2. Kernel optimization changes are implemented in additive, reviewable slices and maintain deterministic prediction/correctness parity.
3. SIMD acceleration is introduced with explicit runtime feature detection and validated scalar fallback behavior.
4. Parent verification report includes benchmark-evidence rollup summarizing baseline and post-change deltas for target workloads.
5. Existing engine/predictor/wrapper regression tests remain green throughout `v0.5` execution.
6. `cargo fmt -- --check` passes at closeout.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
8. `cargo test --workspace` passes at closeout.
9. `cargo doc --workspace --no-deps` passes at closeout.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes at closeout.

## Risks and Mitigations
- Risk: performance changes silently alter prediction behavior.
  - Mitigation: treat deterministic parity tests as hard gates in every child slice.
- Risk: architecture-specific SIMD code harms portability and CI reliability.
  - Mitigation: runtime feature detection with explicit scalar fallback and feature-gated tests.
- Risk: optimization work lacks reproducible measurement discipline.
  - Mitigation: require benchmark harness + baseline/post-change evidence format in child verification reports.
- Risk: scope creep into non-`v0.5` milestones.
  - Mitigation: keep changes bounded to CPU kernel performance and benchmark infrastructure.

## Assumptions and Defaults
- Device scope remains CPU-only through `v0.5`.
- AVX2 is the first SIMD target; AVX-512 and NEON expansion are optional future follow-ons.
- `v0.5` child layers continue `v0.4.x` numbering.
- Benchmark evidence is captured with repository-local workloads first; roadmap-level LightGBM-relative performance comparisons are evaluated at parent closeout readiness.
