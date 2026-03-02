# AlloyGBM v0.7.1 Implementation Notes

## Summary of What Was Built
- Executed `v0.7.1` by replacing `crates/shap` placeholder behavior with artifact-backed SHAP contract APIs and fixture-driven additivity validation.
- Updated [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs):
  - introduced `ShapExplanationBatch` (`expected_value` + per-row contribution matrix),
  - added `explain_rows_from_artifact_bytes(...)`,
  - added `global_importance_from_shap_values(...)` and `global_importance_from_artifact_bytes(...)`,
  - added deterministic validation for rows (empty/mismatched/non-finite) and artifact compatibility,
  - replaced `NotImplemented` placeholder errors with explicit `ContractViolation` behavior.
- Added SHAP runtime dependency in [crates/shap/Cargo.toml](/Users/lashby/Projects/AlloyGBM/crates/shap/Cargo.toml) on `alloygbm-engine` to consume existing artifact/prediction contract.
- Added focused tests in `crates/shap/src/lib.rs` covering:
  - artifact compatibility failures,
  - row validation failures,
  - deterministic explanation shape,
  - per-row additivity identity against model predictions,
  - global-importance aggregation ordering.

## Non-Intuitive Decisions
- Decision: keep legacy `shap_values_stub`/`global_importance_stub` functions as deterministic compatibility shims returning zeroed outputs.
- Reason: avoid abrupt API break for any callers bound to the earlier placeholder names while introducing artifact-backed APIs as the primary path.
- Impact: existing placeholder function names remain callable but are no longer `NotImplemented`; real artifact-backed behavior is available through new entrypoints.

- Decision: implement additivity by accumulating active stump leaf contributions on split features in this slice.
- Reason: `v0.7.1` scope is contract and additivity harness, not exact TreeSHAP path weighting (`v0.7.2` scope).
- Impact: deterministic explanation shape/additivity is now locked by tests, enabling exact weighting replacement in the next layer without changing public contract.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Boundaries remained in-scope for `v0.7.1`:
  - changed only `crates/shap` runtime and dependency wiring,
  - did not modify engine/predictor/core prediction behavior,
  - did not add Python SHAP bridge surface yet.
- New SHAP API contracts are additive and artifact-backed as planned.

## Known Gaps Deferred to Next Layer
- Exact TreeSHAP path-weight/probability algorithm is deferred to `v0.7.2`.
- Python bridge/regressor SHAP APIs are deferred to `v0.7.4`.
- Interaction SHAP and approximate SHAP modes remain out-of-scope.

## Follow-Up Actions
- Plan and implement `docs/architecture/v1.0/v0.8/v0.7.2/plan.md` for exact TreeSHAP traversal/math while preserving the `v0.7.1` API contract and tests.
- Keep additivity fixtures as non-regression tests during algorithm replacement.
