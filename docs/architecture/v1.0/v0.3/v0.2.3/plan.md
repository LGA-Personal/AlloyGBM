# AlloyGBM v0.2.3 Plan (v0.3 Estimator Round-Control Slice)

## Objective
Execute the next `v0.3` child slice by introducing explicit estimator round-count control (`n_estimators`) for `GBMRegressor`, wiring it through the Python wrapper and native training bridge while preserving existing sklearn-style contracts and validation semantics.

## Scope
- In scope:
  - Add `n_estimators` to `GBMRegressor` constructor and parameter surface (`get_params`/`set_params`) with deterministic validation (`> 0`).
  - Forward `n_estimators` from `GBMRegressor.fit(...)` into native training so wrapper-configured rounds are honored.
  - Extend native binding `train_regression_artifact(...)` signature to accept explicit round count and execute training with that value.
  - Add/extend contract and runtime tests covering parameter validation, forwarding behavior, and deterministic prediction behavior with configured rounds.
  - Produce `v0.2.3` implementation and verification artifacts; update layer index.
- Out of scope:
  - Continuous-feature quantization or model-format redesign.
  - Ranking/categorical/SHAP scope expansion.
  - SIMD/performance optimization campaigns.
  - Parent rollup closure artifacts at `v0.3` or `v1.0`.

## Deliverables
1. Estimator parameter package:
  - updates to `bindings/python/alloygbm/regressor.py` for `n_estimators` constructor, representation, parameter roundtrip, and fit forwarding.
2. Native bridge package:
  - updates to `bindings/python/src/lib.rs` so `train_regression_artifact(...)` accepts and enforces explicit training rounds.
3. Test evidence package:
  - updates to `bindings/python/tests/test_regressor_contract.py` and `bindings/python/tests/test_native_runtime_integration.py` covering round-count behavior and contract preservation.
4. Layer documentation package:
  - `docs/architecture/v1.0/v0.3/v0.2.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/v0.2.3/verification_report.md`
5. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.2.3` completion and next suggested target.

## Implementation Sequence
1. Extend `GBMRegressor` parameter surface with `n_estimators` and constructor-equivalent validation in `set_params`.
2. Pass `n_estimators` through `fit(...)` into the native training bridge call.
3. Update native bridge signature and training call path to use provided rounds.
4. Add/adjust tests for parameter validation, parameter forwarding, and runtime round-control behavior.
5. Run full verification gates and publish implementation/verification artifacts.
6. Update layer index state.

## Acceptance Criteria
1. `GBMRegressor(...)` accepts `n_estimators` with default value and rejects non-positive values.
2. `GBMRegressor.get_params()` and `set_params(...)` include `n_estimators` with stable sklearn-style behavior.
3. `GBMRegressor.fit(...)` forwards configured `n_estimators` to native `train_regression_artifact(...)`.
4. Native training bridge uses caller-provided rounds and rejects invalid round counts.
5. Python contract/runtime tests provide evidence that configured rounds flow through successfully while existing predict/error contracts remain passing.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: introducing `n_estimators` may break existing parameter roundtrip behavior.
  - Mitigation: extend existing parameter contract tests and keep validation logic constructor-equivalent.
- Risk: bridge round-control could diverge between Python and Rust defaults.
  - Mitigation: use a single explicit default (`6`) at both call sites and assert forwarding in tests.
- Risk: runtime behavior checks may become flaky if prediction deltas are too strict.
  - Mitigation: assert deterministic invariants and robust, coarse-grained behavior differences for round-count-sensitive fixtures.

## Exit Condition
`v0.2.3` is complete when explicit round-count control is implemented and verified end-to-end with passing gate commands, and all required layer artifacts are present.
