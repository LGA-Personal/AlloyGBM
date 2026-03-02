# AlloyGBM v0.6.4 Plan (Python Categorical Bridge and End-to-End Contract Slice)

## Summary
- Goal: execute the `v0.6.4` slice by exposing additive categorical-training controls through the Python bridge and validating end-to-end parity across bridge and engine paths.
- Success criteria:
  - native Python binding supports optional categorical target-encoding inputs without breaking numeric-only behavior,
  - `GBMRegressor` adds additive categorical configuration and fit-time validation,
  - bridge-level tests prove categorical artifact generation and strict predictor replay paths remain consistent.
- Audience: engineers closing `v0.7` child execution and reviewers gating readiness for parent `v0.7` rollup.

## Scope
### In Scope
- Native binding updates in `bindings/python/src/lib.rs`:
  - extend `train_regression_artifact` with optional categorical bridge arguments,
  - construct `CategoricalTargetEncodingSpec` + `TargetEncoderConfig` when categorical options are provided,
  - wire optional `time_index` into `TrainingDataset` and route categorical training through `Trainer::fit_iterations_with_single_target_encoded_feature`.
- Python estimator updates in `bindings/python/alloygbm/regressor.py`:
  - add additive categorical configuration parameters,
  - add fit-time validation for categorical feature index, categorical values, and time-index requirements,
  - pass categorical bridge arguments through to native training while preserving numeric-only defaults.
- Test coverage:
  - Rust bridge tests for categorical artifact path parity and validation errors,
  - Python contract/runtime tests for additive categorical API behavior and non-regression of numeric paths.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.7/v0.6.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.7/v0.6.4/verification_report.md`

### Out of Scope
- Multi-feature categorical preprocessing orchestration in engine.
- Predictor-side categorical transformation execution from artifact state.
- Ranking/SHAP/GPU scope or model-format version changes.
- Breaking changes to existing numeric-only `GBMRegressor` usage.

## Interfaces and Types
- `bindings/python/src/lib.rs`:
  - extended `train_regression_artifact` keyword signature with optional categorical and time-index arguments,
  - bridge helper logic to build optional categorical spec.
- `bindings/python/alloygbm/regressor.py`:
  - additive estimator parameters and fit-time optional categorical inputs.
- `crates/engine/src/lib.rs`:
  - consume existing `CategoricalTargetEncodingSpec` and training wrapper; no contract changes expected.

Backward-compatibility expectations:
- Existing numeric-only fit/predict behavior and tests remain unchanged.
- Canonical strict-artifact predictor path remains default for `GBMRegressor.predict`.
- New categorical controls are optional and additive.

## Deliverables
1. Bridge extension package:
  - native binding accepts optional categorical/time-index arguments and validates option consistency.
2. Estimator API package:
  - additive categorical constructor and fit-time controls in `GBMRegressor`.
3. Test package:
  - updated Rust/Python tests covering categorical bridge path and numeric-path non-regression.
4. Verification package:
  - criterion-to-evidence mapping in `verification_report.md` with command outcomes.
5. State package:
  - `docs/architecture/state/layer_index.yaml` updated to mark `v0.6.4` verified and set next suggested target.

## Implementation Sequence
1. Add `v0.6.4/plan.md` and lock scope to Python bridge integration and tests.
2. Implement native binding optional categorical argument parsing and trainer routing.
3. Implement additive `GBMRegressor` categorical configuration + fit-time input validation.
4. Add/update Rust and Python tests for categorical bridge behavior and numeric compatibility.
5. Run targeted and full verification gates.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `layer_index.yaml` for `v0.6.4` completion.

## Test Cases and Scenarios
- Rust bridge cases:
  - categorical bridge artifact generation attaches categorical state,
  - bridge categorical training path predictions match direct engine categorical path on the same fixture,
  - bridge rejects mismatched categorical option combinations.
- Python contract/runtime cases:
  - estimator categorical params round-trip through `get_params`/`set_params`,
  - fit rejects missing categorical values when categorical feature index is configured,
  - fit rejects missing time index in time-aware categorical mode,
  - categorical bridge path remains deterministic and usable through native runtime.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. Native `train_regression_artifact` supports optional categorical and time-index arguments without breaking existing numeric signature usage.
2. Bridge categorical path routes through engine categorical wrapper and emits artifact categorical state.
3. `GBMRegressor` adds additive categorical configuration with explicit fit-time validation for incompatible/missing inputs.
4. Numeric-only regressor and bridge behavior remain green under existing tests.
5. `docs/architecture/v1.0/v0.7/v0.6.4/implementation_notes.md` is created.
6. `docs/architecture/v1.0/v0.7/v0.6.4/verification_report.md` is created.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. `cargo doc --workspace --no-deps` passes.
11. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: optional categorical arguments create ambiguous bridge behavior.
  - Mitigation: enforce explicit option-consistency validation and clear error messages.
- Risk: categorical bridge changes regress numeric-only Python path.
  - Mitigation: keep defaults unchanged and retain existing contract/runtime tests as hard gates.
- Risk: time-aware mode misuse without time index.
  - Mitigation: validate at Python and native bridge boundaries.

## Assumptions and Defaults
- `v0.6.4` remains single-feature categorical bridge scope, matching current engine integration shape.
- Categorical predictor replay remains artifact-state aware but transform execution is deferred beyond this slice.
- CPU-only device scope remains unchanged.
