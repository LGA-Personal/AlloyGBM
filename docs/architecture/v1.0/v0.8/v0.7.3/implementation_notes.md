# AlloyGBM v0.7.3 Implementation Notes

## Summary of What Was Built
- Executed `v0.7.3` as a hardening slice focused on compatibility coverage and predictor-parity validation for artifact-backed SHAP behavior.
- Updated [crates/shap/Cargo.toml](/Users/lashby/Projects/AlloyGBM/crates/shap/Cargo.toml):
  - added test-only dependency on `alloygbm-predictor`.
- Updated [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs):
  - added fixture helper `fixture_trees_payload()` for constructing compatibility-edge artifacts,
  - added compatibility tests for:
    - legacy trees-only artifact acceptance,
    - duplicate `Trees` required-section rejection,
    - metadata/payload feature-count mismatch rejection,
  - added predictor-parity test to verify SHAP additivity reconstruction against `Predictor::predict_row(...)`,
  - added deterministic tie-break ordering test for global importance sorting.

## Non-Intuitive Decisions
- Decision: implement `v0.7.3` as test-focused hardening rather than changing SHAP runtime algorithms.
- Reason: `v0.7.1` and `v0.7.2` already delivered core runtime behavior; parent `v0.8` sequence calls for compatibility/parity hardening before Python bridge work.
- Impact: API/runtime stability is preserved while expanding confidence in edge-case compatibility and prediction parity.

- Decision: verify parity against `alloygbm-predictor` in `crates/shap` tests via dev-dependency.
- Reason: parent plan requires parity checks against predictor outputs, and predictor is the artifact-source-of-truth inference path.
- Impact: SHAP additivity guarantees are now explicitly tied to predictor behavior rather than only `TrainedModel` internals.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted in `v0.7.3/plan.md`.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Public SHAP APIs remained unchanged.
- No changes were made to engine/predictor production behavior.
- Changes were limited to SHAP crate test coverage and test dependency wiring, as planned.

## Known Gaps Deferred to Next Layer
- Python SHAP bridge/regressor surface (`bindings/python` + `GBMRegressor.shap_values`) remains deferred to `v0.7.4`.
- Parent-level `v0.8` closeout artifacts (`implementation_notes.md` and `verification_report.md` at `docs/architecture/v1.0/v0.8/`) remain pending final child-slice completion.

## Follow-Up Actions
- Plan and execute `docs/architecture/v1.0/v0.8/v0.7.4/plan.md` for Python SHAP bridge API and corresponding Python-side additivity/error tests.
- Preserve new predictor-parity and compatibility tests as non-regression gates for subsequent SHAP bridge integration.
