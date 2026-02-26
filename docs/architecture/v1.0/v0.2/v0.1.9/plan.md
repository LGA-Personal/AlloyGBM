# AlloyGBM v0.1.9 Plan (v0.2 Python Runtime Success-Path Parity)

## Objective
Close the remaining `v0.2` runtime-evidence gap by adding Python-runtime tests that execute predictor inference on valid artifact bytes and verify parity between direct native entrypoint execution and the public `GBMRegressor.predict_from_artifact(...)` bridge.

## Scope
- In scope:
  - Add deterministic valid-artifact fixture coverage in Python runtime integration tests.
  - Assert successful native prediction execution from installed wheel runtime.
  - Assert public regressor bridge predictions match direct native entrypoint predictions for the same artifact/rows.
  - Keep existing runtime error-path checks intact.
- Out of scope:
  - Engine training/inference semantics changes.
  - Predictor artifact format changes.
  - New public estimator/training API expansion.
  - Parent-layer rollup artifacts for `v0.2` and `v1.0`.

## Deliverables
1. Runtime parity test package:
  - updated `bindings/python/tests/test_native_runtime_integration.py` with valid-artifact success-path parity assertions.
2. Verification package:
  - `docs/architecture/v1.0/v0.2/v0.1.9/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/v0.1.9/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.9` completion and next target.

## Implementation Sequence
1. Create `v0.1.9` plan artifact.
2. Add valid-artifact fixture and success-path runtime assertions in integration tests.
3. Run verification command gates and collect criterion-mapped evidence.
4. Write implementation/verification artifacts and update state index.

## Acceptance Criteria
1. Runtime integration tests execute native predictor inference successfully on valid artifact bytes and assert deterministic expected values.
2. Runtime integration tests verify `GBMRegressor.predict_from_artifact(...)` matches direct native predictor predictions on valid artifact bytes.
3. Existing runtime integration error-path checks continue passing.
4. Existing Python regressor contract tests remain passing.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: fixture artifact drifts from deterministic assumptions across algorithm changes.
  - Mitigation: keep fixture generation tied to deterministic training config and assert only to stable precision.
- Risk: runtime parity tests might mask mismatches if both paths regress similarly.
  - Mitigation: retain explicit expected-value assertions for direct native path in addition to bridge/native parity assertion.
- Risk: test runtime cost increases due to wheel build/install step.
  - Mitigation: keep one shared class-level setup and focused assertions.

## Exit Condition
`v0.1.9` is complete when Python-runtime success-path parity is covered with deterministic evidence, all verification gates pass, and layer/state artifacts are updated.
