# AlloyGBM v0.6.4 Implementation Notes

## Summary of What Was Built
- Executed `v0.6.4` by wiring categorical training options through the Python bridge while preserving numeric-only defaults.
- Updated Rust Python binding in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - added optional categorical + time-index keyword arguments to `train_regression_artifact`,
  - added `resolve_categorical_spec(...)` consistency validation for bridge options,
  - routed categorical-enabled training through `Trainer::fit_iterations_with_single_target_encoded_feature`,
  - forwarded optional `time_index` into `TrainingDataset`.
- Added categorical dependency in [bindings/python/Cargo.toml](/Users/lashby/Projects/AlloyGBM/bindings/python/Cargo.toml) to construct `TargetEncoderConfig` in binding layer.
- Expanded Python estimator in [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - additive constructor params (`categorical_feature_index`, `categorical_smoothing`, `categorical_min_samples_leaf`, `categorical_time_aware`),
  - additive `fit(..., categorical_feature_values=..., time_index=...)` path with explicit validation,
  - pass-through of categorical bridge args to native training function.
- Added/updated tests:
  - Rust binding tests in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
    - `train_bridge_categorical_path_matches_engine_predictions`
    - `train_bridge_rejects_partial_categorical_arguments`
  - Python contract tests in [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py):
    - constructor/params coverage for new categorical options,
    - fit-time validation checks,
    - bridge argument pass-through assertions.
  - Python runtime integration in [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py):
    - `test_native_and_regressor_categorical_bridge_paths_match`.

## Non-Intuitive Decisions
- Decision: enforce categorical bridge argument consistency at both Python and Rust boundaries (for example, index requires values, and time-aware mode requires `time_index`).
- Reason: fail-fast behavior avoids ambiguous bridge behavior and prevents silent numeric fallbacks when categorical mode was intended.
- Impact: callers receive deterministic validation errors before training starts.

- Decision: keep categorical integration single-feature in bridge path.
- Reason: current engine categorical integration contract for this milestone is single-feature target encoding (`CategoricalTargetEncodingSpec`), and expanding to multi-feature orchestration would exceed `v0.6.4` scope.
- Impact: bridge achieves planned parity with current engine capabilities while deferring broader categorical orchestration.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- `train_regression_artifact` now accepts additive categorical/time-index keywords; numeric-only invocation remains valid.
- `GBMRegressor.fit` now optionally accepts `categorical_feature_values` and `time_index` while preserving existing `fit(X, y)` behavior.
- Predictor strict artifact behavior remains unchanged for `GBMRegressor.predict`.

## Known Gaps Deferred Beyond This Layer
- Multi-feature categorical preprocessing orchestration remains deferred.
- Predictor/runtime execution of categorical transforms from artifact state remains deferred (artifact currently records categorical state metadata, not full transform replay state).
- Parent-layer `docs/architecture/v1.0/v0.7/implementation_notes.md` and `verification_report.md` remain pending parent closeout.

## Follow-Up Actions
- Close parent `v0.7` rollup artifacts now that child slices `v0.6.1` through `v0.6.4` are complete.
- Decide next active target in `layer_index.yaml` after parent closeout (likely transition to `v0.8` planning).
