# v0.1.8 Implementation Notes

## Summary of What Was Built
- Added Python runtime integration coverage in `bindings/python/tests/test_native_runtime_integration.py`:
  - builds a wheel from `bindings/python/Cargo.toml` via `python3 -m maturin build`,
  - installs the built wheel into an isolated temporary target directory,
  - imports `alloygbm` from the installed runtime package,
  - executes native APIs from Python runtime (`native_runtime_info`, `_alloygbm.predictor_predict_batch`, and `GBMRegressor.predict_from_artifact`).
- Added runtime assertions confirming native error propagation for invalid artifact payloads through both direct native entrypoint and public regressor bridge.

## Non-Intuitive Decisions
- Decision: build/install a wheel inside tests rather than importing local source package.
- Reason: the unresolved `v0.1.7` gap was specifically Python-runtime native extension execution evidence, which source-only imports do not prove.
- Impact: tests validate real extension import/runtime behavior while remaining self-contained.

- Decision: validate runtime execution using invalid artifact bytes and expected runtime errors.
- Reason: this closes runtime invocation/evidence scope without introducing new artifact-fixture generation machinery in `v0.1.8`.
- Impact: proves Python runtime reaches native predictor execution path and preserves error mapping; successful-prediction runtime parity remains a possible future enhancement.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.1.8/plan.md`.

## Boundary/Interface Changes vs Plan
- No Rust API or training-path behavior changes.
- No public Python surface expansion beyond existing APIs.
- Test harness scope only, as planned.

## Known Gaps Deferred to Next Layer
- Python-runtime success-path parity for valid artifact bytes is not yet explicitly asserted in Python tests.
- Parent rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/verification_report.md`
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Follow-Up Actions
- Define `v0.1.9` to add Python-runtime success-path parity assertions for valid artifact bytes if stricter end-to-end evidence is required before parent rollups.
