# AlloyGBM v0.2.2 Plan (v0.3 Input Adapter Coverage Slice)

## Objective
Execute the next `v0.3` child slice by adding deterministic input normalization adapters so `GBMRegressor` accepts NumPy-like, pandas-like, and Polars-like tabular inputs for `fit`, `predict`, and `predict_from_artifact`, while preserving existing parameter and validation contracts.

## Scope
- In scope:
  - Extend Python-side input normalization in `bindings/python/alloygbm/regressor.py` for row inputs accepted by:
    - `fit(X, y)`,
    - `predict(X)`,
    - `predict_from_artifact(..., X)`.
  - Support duck-typed input sources commonly used by NumPy/pandas/Polars workflows (`tolist`, `to_list`, `to_numpy` paths).
  - Keep feature-count and shape validation semantics explicit and deterministic.
  - Add/extend tests that prove adapter paths and existing contracts remain intact.
  - Update layer artifacts and layer-state index for `v0.2.2`.
- Out of scope:
  - Native training bridge algorithm changes.
  - Continuous-feature quantization redesign or model-format changes.
  - Estimator round-count API expansion.
  - Parent `v0.3` rollup closeout artifacts.

## Deliverables
1. Adapter behavior package:
  - updates to `bindings/python/alloygbm/regressor.py` to normalize row/target inputs from NumPy-like, pandas-like, and Polars-like objects.
2. Test evidence package:
  - updates to `bindings/python/tests/test_regressor_contract.py` and/or runtime integration tests to validate adapter paths and invariant preservation.
3. Layer documentation package:
  - `docs/architecture/v1.0/v0.3/v0.2.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/v0.2.2/verification_report.md`
4. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.2.2` completion and next target.

## Implementation Sequence
1. Add adapter helpers in `regressor.py` for row-like and target-like normalization (`tolist`, `to_list`, `to_numpy`).
2. Route `fit`, `predict`, and `predict_from_artifact` through the new normalization helpers while preserving existing guardrails.
3. Add tests covering adapter acceptance and mismatch/error behavior.
4. Run layer verification command gates and capture criterion-mapped evidence.
5. Write layer implementation and verification artifacts; update state index.

## Acceptance Criteria
1. `GBMRegressor.fit(...)` accepts row inputs supplied as sequence rows and NumPy-like/pandas-like/Polars-like tabular objects via normalization adapters.
2. `GBMRegressor.predict(...)` and `GBMRegressor.predict_from_artifact(...)` accept the same adapter-supported row input shapes.
3. Feature-count mismatch and malformed-input error semantics remain explicit and deterministic.
4. Existing parameter contract behavior (`get_params`/`set_params`) remains passing.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: adapter logic may silently coerce malformed structures.
  - Mitigation: keep strict sequence/shape validation after adapter conversion and retain explicit error messages.
- Risk: adapter support could drift into dependency-specific code paths.
  - Mitigation: use duck-typed interfaces and keep adapters dependency-agnostic.
- Risk: new adapter paths may regress existing bridge behavior.
  - Mitigation: preserve existing tests and add targeted adapter-path coverage.

## Exit Condition
`v0.2.2` is complete when adapter-supported inputs are implemented with passing tests and all gate commands are green, with implementation and verification artifacts recorded.
