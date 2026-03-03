# AlloyGBM v0.8 Technical Plan

## Summary
- Goal: deliver the `0.8.0` release-candidate hardening milestone by closing residual test, documentation, benchmark reproducibility, and compatibility-verification gaps ahead of `1.0.0`.
- Success criteria:
  - release gates are explicit and reproducible across Rust and Python surfaces,
  - `v0.7` SHAP/categorical/predictor behavior remains non-regressed while hardening work lands,
  - migration notes and compatibility checks are complete enough for a `1.0.0` go/no-go review.
- Audience: engineers implementing `v0.8` child slices and reviewers deciding readiness to plan `v1.0` closeout.

## Scope
### In Scope
- Release-candidate hardening for the existing CPU baseline:
  - test expansion for edge/compatibility paths not yet covered by `v0.7` closeout artifacts,
  - documentation updates for operational usage, compatibility expectations, and release gating,
  - benchmark reproducibility artifacts and command discipline for repeated runs,
  - migration notes and compatibility checks across current model artifact behaviors.
- Child-layer decomposition under `docs/architecture/v1.0/v0.8/v0.7.x`.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.8/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/verification_report.md`

### Out of Scope
- New roadmap features (`1.0`+), including ranking objectives, probabilistic outputs, or GPU backends.
- Model-format version bump beyond current v1 contract.
- Public API redesigns that break current `GBMRegressor` call signatures.
- Net-new algorithmic feature development outside hardening and compatibility validation.

## Interfaces and Types
- `crates/core`, `crates/engine`, `crates/predictor`, `crates/shap`, `crates/categorical`:
  - may receive hardening fixes only where needed to satisfy deterministic behavior, compatibility checks, and release gates.
- `bindings/python/src/lib.rs` and `bindings/python/alloygbm/regressor.py`:
  - maintain additive/backward-compatible behavior while expanding contract coverage.
- CI and release evidence artifacts:
  - `.github/workflows/ci.yml` command alignment,
  - architecture verification artifacts under `docs/architecture/v1.0/v0.8/`.
- State tracking:
  - `docs/architecture/state/layer_index.yaml` must be updated as child slices and parent closeout complete.

Backward-compatibility expectations:
- preserve `v0.7` SHAP APIs and behavior baseline (`shap_values`, SHAP-based feature importance),
- preserve artifact compatibility handling for strict and legacy-supported payloads,
- keep numeric-only and categorical-capable workflows contract-stable in Python.

## Implementation Sequence
1. Execute `docs/architecture/v1.0/v0.8/v0.8.1/` first to lock the `v0.8` hardening matrix (release gates, artifact checklist, and baseline inventory).
2. Open `v0.8.2` for targeted test-gap closure and deterministic edge-case coverage aligned to the matrix.
3. Open `v0.8.3` for benchmark reproducibility artifacts, command normalization, and evidence packaging.
4. Open `v0.8.4` for migration-note finalization, compatibility checks, and parent artifact rollup readiness.
5. Close parent `v0.8` with implementation notes, verification report, and layer index update for next-layer planning.

## Test Cases and Scenarios
- Unit cases:
  - deterministic serialization/compatibility assertions across strict and legacy artifact paths,
  - SHAP and categorical guardrail/error-semantics stability checks.
- Integration cases:
  - Rust train -> artifact -> predictor parity checks on representative fixtures,
  - Python estimator contract + SHAP contract checks for fitted-model workflows.
- Reproducibility/benchmark cases:
  - repeated benchmark runs with documented environment and command inputs,
  - evidence tables showing run-to-run stability for selected workloads.
- Failure and edge cases:
  - malformed artifact handling and deterministic error paths,
  - feature/shape mismatch behavior at Rust and Python boundaries.
- Acceptance test mapping:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
  - child-defined reproducible benchmark commands with recorded environment context

## Acceptance Criteria
1. `v0.8` child slices produce a decision-complete hardening matrix with explicit gate-to-evidence mapping.
2. Test coverage is expanded for compatibility and deterministic edge paths without regressing `v0.7` behavior.
3. Reproducible benchmark artifacts are published with stable command definitions and environment notes.
4. Migration notes and compatibility checks are complete and traceable for `1.0.0` gate review.
5. Parent rollup artifacts summarize all child evidence and residual risks.
6. `cargo fmt -- --check` passes at closeout.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
8. `cargo test --workspace` passes at closeout.
9. `cargo doc --workspace --no-deps` passes at closeout.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes at closeout.

## Risks and Mitigations
- Risk: hardening work introduces unintended behavior drift while touching multiple crates.
  - Mitigation: treat `v0.7` verification behaviors as non-regression gates in every child slice.
- Risk: benchmark evidence is noisy or non-reproducible.
  - Mitigation: standardize environment capture and repeated-run protocol before reporting deltas.
- Risk: compatibility assumptions are documented but not executable.
  - Mitigation: require command-backed compatibility checks referenced directly in verification artifacts.
- Risk: scope creep into `1.0.0` feature changes.
  - Mitigation: enforce hardening-only boundary in each `v0.7.x` child plan.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.8` child layers use `v0.7.x` numbering.
- Immediate next child target is `docs/architecture/v1.0/v0.8/v0.8.1`.
- `v0.8` focuses on hardening and release evidence; feature expansion remains outside this layer.
