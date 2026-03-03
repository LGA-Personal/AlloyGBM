# AlloyGBM v0.5.2 Implementation Notes

## Summary of What Was Built
- Added canonical strict predictor bridge in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - `predictor_predict_batch_canonical_impl` validates strict dual-section artifacts via engine compatibility mode,
  - `predictor_predict_batch_canonical` is exported through the `_alloygbm` module.
- Updated Python estimator routing in [regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - `GBMRegressor.predict` now uses `_load_native_predictor_predict_batch_canonical`,
  - `GBMRegressor.predict_from_artifact` remains on compatibility loader `_load_native_predictor_predict_batch`.
- Expanded contract coverage:
  - Rust binding tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs) for canonical strict acceptance and legacy rejection.
  - Python routing tests in [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py) for canonical-vs-compatibility loader separation.

## Non-Intuitive Decisions
- Decision: enforce strict artifact compatibility only on estimator `predict`, not on `predict_from_artifact`.
- Reason: canonical path should apply to internally trained artifacts, while utility artifact prediction must remain backward-compatible with legacy external payloads.
- Impact: canonicalization is applied without breaking legacy artifact workflows.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No public API signature changes for `GBMRegressor`.
- Native module added an internal callable (`predictor_predict_batch_canonical`) used by Python wrapper routing.
- Compatibility behavior for `predict_from_artifact` was preserved.

## Known Gaps Deferred to Next Layer
- Parent `v0.5` serialization hardening and migration/documentation tightening remain for `v0.5.3`.
- Parent closeout artifacts remain pending:
  - `docs/architecture/v1.0/v0.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/verification_report.md`

## Follow-Up Actions
- Create `docs/architecture/v1.0/v0.5/v0.5.3/plan.md` for serialization-contract hardening and failure-mode consistency.
- Keep strict/legacy semantics synchronized between engine compatibility reporting and Python canonical bridge behavior.
