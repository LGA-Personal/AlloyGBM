# AlloyGBM v0.1.7 Plan (v0.1 Python Binding Predictor Bridge)

## Objective
Close a remaining `v0.1` integration gap by exposing predictor artifact inference through the Python native binding entry points and adding parity-focused evidence that Python-facing calls produce the same predictions as engine inference from the same serialized model bytes.

## Scope
- In scope:
  - Add a Python native entry point in `bindings/python` that accepts model artifact bytes plus feature rows and returns predictor batch predictions.
  - Reuse `predictor` crate artifact import/inference behavior from `v0.1.6` without introducing training-path dependencies in Python API.
  - Map predictor/input-contract failures to clear Python exceptions.
  - Add binding-layer tests that validate:
    - Python entry point prediction parity with engine predictions on deterministic fixtures.
    - Input-shape validation failures are surfaced as Python errors.
- Out of scope:
  - New sklearn-style Python training APIs.
  - Changes to engine training semantics.
  - Parent `v0.1` and `v1.0` rollup artifacts.

## Deliverables
1. Python binding bridge package:
  - `bindings/python/src/lib.rs` exports predictor-backed batch inference entry point.
  - `bindings/python/Cargo.toml` includes required crate dependencies for binding implementation/tests.
2. Verification package:
  - binding tests proving engine/predictor parity through Python entry points and validation error behavior.
  - `docs/architecture/v1.0/v0.1/v0.1.7/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/v0.1.7/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.7` completion and next target.

## Implementation Sequence
1. Add `v0.1.7` plan artifact.
2. Implement predictor-backed Python binding function and error mapping.
3. Add binding tests for parity and validation behavior.
4. Run verification command gates and capture evidence.
5. Write implementation/verification artifacts and update layer state index.

## Acceptance Criteria
1. Python binding module exports a predictor-backed batch inference function that accepts artifact bytes and feature rows.
2. Binding function predictions match engine predictions from the same serialized model bytes on deterministic fixture rows.
3. Binding function rejects invalid inputs (for example feature-count mismatch or empty rows) with clear Python errors.
4. Existing Python `GBMRegressor` contract tests remain passing.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: binding-level conversions could mask predictor contract errors.
  - Mitigation: preserve predictor error detail in exception messages and assert on failure behavior in tests.
- Risk: Python entry point behavior could drift from engine inference semantics.
  - Mitigation: parity test trains engine model, serializes bytes, and compares predictions returned through binding function.
- Risk: scope drift into full Python training integration.
  - Mitigation: keep this layer inference-bridge only and defer estimator training API upgrades.

## Exit Condition
`v0.1.7` is complete when predictor-backed Python inference entry points are implemented with parity/validation evidence, all verification commands pass, and layer/state artifacts are updated.
