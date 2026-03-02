# AlloyGBM v0.5.2 Plan (Predictor Path Canonicalization Slice)

## Summary
- Goal: execute the `v0.5.2` slice by making trained-model scoring flow use a strict canonical artifact path while preserving legacy-compatible external artifact prediction behavior.
- Success criteria:
  - `GBMRegressor.predict` uses a strict dual-section artifact gate before predictor execution,
  - externally supplied artifact prediction remains compatibility-friendly via `predict_from_artifact`,
  - strict-vs-legacy behavior is covered by Rust binding tests and Python contract tests.
- Audience: engineers implementing `v0.6` predictor-path integration and reviewers validating canonicalization boundaries before broader hardening in `v0.5.3`.

## Scope
### In Scope
- Add canonical predictor bridge in Python native module (`bindings/python/src/lib.rs`) that:
  - requires strict dual-section artifacts (`Trees` + `PredictorLayout`) for canonical scoring path,
  - delegates prediction execution to predictor after strict compatibility validation.
- Route Python estimator prediction path in `bindings/python/alloygbm/regressor.py`:
  - `GBMRegressor.predict` -> canonical strict bridge,
  - `GBMRegressor.predict_from_artifact` -> existing compatibility bridge.
- Add/adjust tests:
  - Rust binding tests for strict-path acceptance and legacy rejection behavior.
  - Python contract tests confirming `predict` and `predict_from_artifact` call distinct loaders.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.6/v0.5.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.6/v0.5.2/verification_report.md`

### Out of Scope
- Model format version bump or section schema redesign.
- New public Python API methods.
- Predictor traversal/performance optimization.
- Parent `v0.6` closeout rollup artifacts.

## Interfaces and Types
- `bindings/python/src/lib.rs`:
  - new canonical bridge function for strict artifact prediction path.
- `bindings/python/alloygbm/regressor.py`:
  - internal loader split between canonical strict path (`predict`) and compatibility path (`predict_from_artifact`).
- `crates/engine/src/lib.rs`:
  - strict compatibility mode remains the policy gate implementation.
- `crates/predictor/src/lib.rs`:
  - prediction execution remains canonical scorer backend.

Backward-compatibility expectations:
- no changes to public `GBMRegressor` method names or argument signatures,
- `predict_from_artifact` continues to support legacy trees-only payloads through compatibility path.

## Deliverables
1. Canonical bridge package:
  - strict artifact-gated predictor bridge exposed by Python native module.
2. Python routing package:
  - `GBMRegressor.predict` moved to canonical bridge loader.
3. Test package:
  - Rust + Python tests validating strict-vs-compatibility path behavior.
4. Verification package:
  - full gate command evidence and layer artifacts.
5. State package:
  - `docs/architecture/state/layer_index.yaml` update after verification.

## Implementation Sequence
1. Add canonical strict predictor bridge in `bindings/python/src/lib.rs` and register it in module exports.
2. Update `bindings/python/alloygbm/regressor.py` so `predict` uses canonical loader while `predict_from_artifact` retains compatibility loader.
3. Add/update Rust binding tests and Python contract tests for route separation and strict/legacy behavior.
4. Run targeted tests for bindings and regressor contract.
5. Run full verification gates and record evidence.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `layer_index.yaml` for `v0.5.2` completion and next target suggestion.

## Test Cases and Scenarios
- Unit cases:
  - canonical bridge accepts strict artifacts from training output.
  - canonical bridge rejects legacy trees-only artifacts with deterministic contract error.
  - compatibility bridge remains able to predict from legacy trees-only artifacts.
- Integration cases:
  - `GBMRegressor.fit` + `predict` remains successful via canonical bridge.
  - `GBMRegressor.predict_from_artifact` remains successful via compatibility bridge.
- Failure and edge cases:
  - malformed artifacts continue to surface deterministic errors.
  - loader routing mistakes are caught by Python contract tests.
- Acceptance test mapping:
  - `cargo test -p alloygbm-python`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_regressor_contract.py'`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. Canonical strict predictor bridge exists in Python native module and is exported.
2. `GBMRegressor.predict` uses canonical strict bridge path.
3. `GBMRegressor.predict_from_artifact` remains on compatibility path.
4. Rust binding tests validate strict-accept/legacy-reject behavior for canonical bridge.
5. Python contract tests validate loader routing separation for `predict` vs `predict_from_artifact`.
6. `docs/architecture/v1.0/v0.6/v0.5.2/implementation_notes.md` is created.
7. `docs/architecture/v1.0/v0.6/v0.5.2/verification_report.md` is created.
8. `cargo fmt -- --check` passes.
9. `cargo clippy --workspace --all-targets -- -D warnings` passes.
10. `cargo test --workspace` passes.
11. `cargo doc --workspace --no-deps` passes.
12. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: strict gating on estimator `predict` breaks fitted-model scoring.
  - Mitigation: explicit parity tests for training artifacts through canonical bridge.
- Risk: compatibility path accidentally removed for external artifacts.
  - Mitigation: maintain separate loader and tests for `predict_from_artifact`.
- Risk: divergence between engine strict semantics and Python bridge behavior.
  - Mitigation: implement canonical gate directly via engine strict compatibility mode.

## Assumptions and Defaults
- Training artifacts generated by current engine bridge are strict dual-section payloads.
- Canonical path enforcement is limited to estimator `predict`; artifact utility method retains compatibility behavior.
- Device scope remains CPU-only.
