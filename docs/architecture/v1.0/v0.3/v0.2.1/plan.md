# AlloyGBM v0.2.1 Plan (v0.3 Wrapper Native Fit/Predict Slice)

## Objective
Execute the first child step of `v0.3` by replacing scaffold-only constant-baseline estimator behavior with native-backed `fit`/`predict` flow in `GBMRegressor`, while preserving sklearn-style parameter and validation contracts.

## Scope
- In scope:
  - Wire `GBMRegressor.fit(...)` to native training/inference-capable path (direct bridge or intermediate artifact flow) for deterministic regression behavior.
  - Keep `GBMRegressor.predict(...)` dependent on fitted native model state rather than constant baseline fallback.
  - Preserve/extend contract tests so `get_params`/`set_params` semantics remain stable.
  - Add coverage proving native-backed predictions differ from trivial constant baseline on deterministic fixture data.
  - Update layer-state index for `v0.2.1` planning/implementation progression.
- Out of scope:
  - Full multi-library dataframe adapters (broader NumPy/pandas/Polars parity can be split into later `v0.2.x` slices).
  - New ranking/categorical/SHAP capabilities.
  - Performance optimization beyond correctness-oriented wrapper behavior.
  - Parent `v0.3` and top-level `v1.0` rollup closure artifacts.

## Deliverables
1. Wrapper behavior package:
  - updates to `bindings/python/alloygbm/regressor.py` for native-backed `fit`/`predict` execution.
  - any required native bridge additions in `bindings/python/src/lib.rs`.
2. Test evidence package:
  - updated `bindings/python/tests/test_regressor_contract.py` and/or runtime integration tests to prove native-backed fit/predict behavior.
3. Layer documentation package:
  - `docs/architecture/v1.0/v0.3/v0.2.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/v0.2.1/verification_report.md`
4. State package:
  - `docs/architecture/state/layer_index.yaml` updated after implementation/verification.

## Implementation Sequence
1. Confirm available native bridge primitives in `bindings/python/src/lib.rs` and extend minimally for training/prediction flow.
2. Replace scaffold constant-baseline internals in `GBMRegressor.fit`/`predict` with deterministic native-backed path.
3. Update/add tests for fit-before-predict, feature-shape validation, and non-trivial prediction behavior.
4. Run full verification gates and capture evidence in layer artifacts.
5. Mark `v0.2.1` state transitions in `layer_index.yaml`.

## Acceptance Criteria
1. `GBMRegressor.fit(...)` stores a native-backed fitted state and no longer relies solely on mean-target baseline behavior.
2. `GBMRegressor.predict(...)` executes against fitted native-backed state and preserves feature-count guardrails.
3. Existing parameter contract tests (`get_params`/`set_params`) remain passing.
4. Python tests include deterministic evidence that fitted predictions are produced through native-backed behavior on fixture data.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: native bridge additions can expand beyond first-slice needs.
  - Mitigation: add only the minimum entrypoints required for `fit`/`predict` contract behavior.
- Risk: deterministic behavior diverges between Python tests and Rust fixtures.
  - Mitigation: use fixed seeds and static fixture data with explicit expected assertions.
- Risk: regressor contract tests become brittle during transition from scaffold to native state.
  - Mitigation: preserve existing public-method invariants and update tests only where behavior is intentionally upgraded.

## Exit Condition
`v0.2.1` is complete when native-backed `fit`/`predict` behavior is implemented and verified, all gate commands pass, and layer/state artifacts are updated.
