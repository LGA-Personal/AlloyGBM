# AlloyGBM v0.3.3 Technical Plan (v0.4 Leakage Guardrail Splits Slice)

## Summary
- Goal: execute the third `v0.4` child slice by adding time-aware validation split helpers with purge/embargo controls for leakage-safe evaluation workflows.
- Success criteria:
  - deterministic split helper APIs exist for time-only and panel (time+group) data,
  - purge/embargo semantics and edge-case failures are explicitly validated by tests,
  - existing metrics and regressor/runtime behavior remain stable while all verification gates stay green.
- Audience: engineers implementing `v0.3.3` and reviewers validating readiness for potential `v0.3.4` polish and parent `v0.4` closeout.

## Scope
### In Scope
- Add a new validation helper module at `bindings/python/alloygbm/validation.py` with:
  - `purged_time_series_splits(time_index, *, n_splits=5, purge_gap=0, embargo=0)`
  - `purged_panel_splits(time_index, group_index, *, n_splits=5, purge_gap=0, embargo=0)`
- Implement deterministic contiguous time-fold splitting using ordered unique timestamps/period keys.
- Apply purge and embargo windows around each test fold to remove leakage-prone training periods.
- Add explicit parameter and input validation for split helpers.
- Export split helpers from package API via `bindings/python/alloygbm/__init__.py`.
- Add deterministic split tests and runtime export checks.
- Produce layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.3.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.3.3/verification_report.md`

### Out of Scope
- Model training-time leakage enforcement changes (this slice is evaluation tooling only).
- Native Rust bridge changes.
- Ranking objective training, SHAP expansion, categorical expansion, backend/performance work.
- Tail metrics or additional finance metrics beyond completed `v0.3.2` scope.

## Interfaces and Types
- `bindings/python/alloygbm/validation.py` (new):
  - split helpers returning `list[tuple[list[int], list[int]]]` where each tuple is `(train_indices, test_indices)`.
- `bindings/python/alloygbm/__init__.py`:
  - additive exports for new split helpers.
- `bindings/python/tests/test_validation_splits.py` (new):
  - deterministic correctness, purge/embargo semantics, and failure-path tests.
- `bindings/python/tests/test_native_runtime_integration.py`:
  - runtime export/callability checks for split helpers.

Backward-compatibility expectations:
- do not modify existing `GBMRegressor` API or behavior.
- keep existing `evaluation.py` metrics semantics unchanged.
- split helpers are additive only.

## Implementation Sequence
1. Create `validation.py` with helper implementations and input/parameter validation.
2. Export split helpers in `__init__.py`.
3. Add `test_validation_splits.py` and extend runtime integration checks.
4. Run full verification gates and record evidence in layer artifacts.
5. Update `docs/architecture/state/layer_index.yaml` after verification.

## Test Cases and Scenarios
- Unit cases:
  - deterministic repeated-call outputs for identical inputs.
  - no train/test overlap for every returned fold.
  - purge and embargo windows remove expected neighboring periods from training.
  - panel helper respects time-bucketed splits for all groups in a period.
- Integration cases:
  - installed package exposes split helpers and returns split structures in runtime tests.
- Failure and edge cases:
  - invalid `n_splits` (`<2` or greater than unique periods),
  - negative `purge_gap` / `embargo`,
  - mismatched `time_index` and `group_index` lengths,
  - empty indices and non-orderable time values.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `purged_time_series_splits` and `purged_panel_splits` are implemented and exported from package API.
2. Split outputs are deterministic and satisfy purge/embargo/no-overlap invariants in tests.
3. Invalid inputs and invalid split configurations raise explicit `ValueError` semantics.
4. Existing `v0.3` regressor/runtime and `v0.3.1`/`v0.3.2` metric tests continue to pass unchanged.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: ambiguous period ordering creates nondeterministic folds.
  - Mitigation: require orderable time values and use explicit sorted unique periods.
- Risk: purge/embargo settings can produce empty training sets.
  - Mitigation: detect and raise explicit configuration errors.
- Risk: panel semantics are unclear without group constraints.
  - Mitigation: define panel behavior as time-bucketed splitting across all groups and assert this in tests.

## Assumptions and Defaults
- `n_splits` default is `5`; folds are contiguous over ordered unique time periods.
- `purge_gap` and `embargo` are measured in unique time periods (not raw row count).
- Panel helper uses time as split axis and includes all groups present in each test period.
- Outputs are index-based folds over the original row order.
