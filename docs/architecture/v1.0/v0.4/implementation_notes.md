# AlloyGBM v0.4 Implementation Notes (Parent Rollup)

## Summary of What Was Built
- Completed `v0.4` through three verified child slices:
  - `v0.3.1`: baseline evaluation metrics (`rmse`, `mae`, `r2_score`, `pearson_correlation`)
  - `v0.3.2`: finance metrics (`rank_ic`, `hit_rate`, `icir`)
  - `v0.3.3`: leakage guardrail split helpers (`purged_time_series_splits`, `purged_panel_splits`)
- Delivered additive Python API surface under `bindings/python/alloygbm/`:
  - metric helpers in [evaluation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/evaluation.py),
  - validation helpers in [validation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/validation.py),
  - package exports via [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py).
- Consolidated deterministic coverage in Python tests:
  - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py)
  - [test_validation_splits.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_validation_splits.py)
  - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
- Preserved existing `GBMRegressor` and native bridge behavior while extending evaluation tooling.

## Non-Intuitive Decisions
- Decision: keep `v0.4` implementation primarily in Python helper modules rather than changing native Rust bridge interfaces.
- Reason: parent `v0.4` scope is evaluation/validation tooling and additive API growth, not training/inference bridge redesign.
- Impact: reduced regression risk to `v0.3` wrapper contract while still delivering required finance and leakage-safe evaluation features.

- Decision: define panel split behavior as time-bucketed splitting across groups.
- Reason: leakage controls (purge/embargo) are most naturally enforced on the time axis for finance panel workflows.
- Impact: deterministic time-aware splits are available now; group balancing/stratified panel split policies remain future enhancement scope.

## Plan Contradictions and Why
- Original Plan Statement: optional `v0.3.4` polish slice may be opened if residual acceptance gaps exist.
- Implemented Decision: no `v0.3.4` slice was opened.
- Reason: `v0.3.1` through `v0.3.3` satisfied parent `v0.4` in-scope deliverables and verification gates.
- Impact: parent closeout proceeded directly after `v0.3.3`.
- Rollback or Migration Consideration: if future gaps are discovered, a follow-up child layer can still be opened without breaking existing APIs.

## Boundary/Interface Changes vs Plan
- Added/expanded public evaluation and validation helpers exactly within `v0.4` scope.
- No ranking objective training, SHAP expansion, categorical expansion, CUDA/Metal work, or performance campaign changes were introduced.
- Existing `GBMRegressor` constructor/fit/predict contract remains unchanged.

## Known Gaps Deferred to Next Layer
- `v0.4` child-scope items are closed for current parent plan.
- Deferred beyond `v0.4`:
  - ranking objective training (`1.1.0` scope),
  - advanced panel split balancing/stratification policies,
  - optional tail metrics and broader evaluation tooling expansion.

## Follow-Up Actions
- Move to next parent-layer planning/implementation target under `v1.0` (for example `v0.5` CPU optimization scope) via new child planning.
- Keep `docs/architecture/state/layer_index.yaml` aligned as next layer planning begins.
