# AlloyGBM v0.2.3 Implementation Notes

## Summary of What Was Built
- Added explicit estimator round-count control in [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - introduced `n_estimators` in `GBMRegressor.__init__` with `> 0` validation,
  - included `n_estimators` in `__repr__`, `get_params`, and `set_params`,
  - forwarded `n_estimators` to native training via `rounds=...` in `fit(...)`.
- Extended native bridge in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - `train_regression_artifact(...)` now accepts `rounds` (default `6`),
  - added explicit bridge-level rejection for `rounds == 0`,
  - training call now uses caller-provided `rounds` instead of a fixed internal value.
- Expanded Python evidence in:
  - [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py): validation and roundtrip coverage for `n_estimators`, plus forwarding assertion in native bridge mock calls.
  - [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py): runtime rejection for zero rounds and end-to-end proof that different `n_estimators` values alter trained artifacts/predictions.

## Non-Intuitive Decisions
- Decision: keep default `n_estimators` at `6`.
- Reason: existing bridge behavior already used six training rounds (`DEFAULT_TRAIN_ROUNDS`), so this preserves prior runtime behavior while exposing explicit control.
- Impact: no regression for users relying on default behavior; configurability is now explicit and test-covered.

- Decision: validate round count at both Python estimator and Rust bridge boundaries.
- Reason: wrapper-level validation preserves sklearn-style contract ergonomics, and bridge-level validation protects direct `_alloygbm` callers.
- Impact: invalid non-positive round configuration fails early and consistently.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none required.

## Boundary/Interface Changes vs Plan
- Public Python estimator interface now includes `n_estimators`.
- Native Python extension function `train_regression_artifact(...)` now accepts `rounds`.
- No expansion into ranking/SHAP/categorical/quantization scope.

## Known Gaps Deferred to Next Layer
- Continuous-feature quantization and mapping for non-prebinned training inputs remains deferred.
- Parent rollup artifacts are still open:
  - `docs/architecture/v1.0/v0.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/verification_report.md`
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Follow-Up Actions
- Close `v0.3` parent rollup artifacts now that child layers `v0.2.1` to `v0.2.3` are verified.
- Re-run/update `docs/architecture/state/layer_index.yaml` after parent rollup completion to advance active target.
