# v0.5.2 Contract Drift Report

## Scope
- Layer: `docs/architecture/v1.0/v0.6/v0.5.2`
- Plan reference: `docs/architecture/v1.0/v0.6/v0.5.2/plan.md`
- Parent reference: `docs/architecture/v1.0/v0.6/plan.md`
- Check date (UTC): 2026-03-02

## Planned Contract Baseline (v0.5.2)
- Add canonical strict predictor bridge in `bindings/python/src/lib.rs`.
- Route `GBMRegressor.predict` through canonical strict bridge.
- Keep `GBMRegressor.predict_from_artifact` on compatibility bridge.
- Keep public Python estimator API signatures unchanged.
- Validate strict-vs-legacy routing behavior via Rust binding tests and Python contract tests.

## Observed Contract Snapshot
- Canonical bridge exists and is exported:
  - `predictor_predict_batch_canonical_impl(...)`
  - `predictor_predict_batch_canonical(...)`
  - module export via `m.add_function(...)`
  - Source: `bindings/python/src/lib.rs`.
- `GBMRegressor.predict` uses canonical loader:
  - `_load_native_predictor_predict_batch_canonical()`
  - Source: `bindings/python/alloygbm/regressor.py`.
- `GBMRegressor.predict_from_artifact` remains compatibility path:
  - `_load_native_predictor_predict_batch()`
  - Source: `bindings/python/alloygbm/regressor.py`.
- Test evidence exists for strict/legacy separation:
  - Rust: canonical accepts strict and rejects legacy trees-only artifacts.
  - Python: route-separation tests for `predict` vs `predict_from_artifact`.
  - Sources: `bindings/python/src/lib.rs`, `bindings/python/tests/test_regressor_contract.py`.

## Drift Findings
- No contract mismatches detected for this layer.

## Category Check Summary

### dependency-direction drift
- Result: No drift detected.
- Evidence:
  - `crates/engine/Cargo.toml` depends on `alloygbm-core` only.
  - `crates/predictor/Cargo.toml` depends on `alloygbm-core` (dev-only engine/backend for tests).
  - `bindings/python/Cargo.toml` depends on `alloygbm-backend-cpu`, `alloygbm-core`, `alloygbm-engine`, and `alloygbm-predictor`; no new reverse dependency direction was introduced in this slice.

### public API drift
- Result: No drift detected.
- Evidence:
  - `GBMRegressor` public methods/signatures are unchanged (`fit`, `predict`, `predict_from_artifact`, `get_params`, `set_params`).
  - Canonicalization changed internal loader routing only.
  - Source: `bindings/python/alloygbm/regressor.py`.

### artifact contract drift
- Result: No drift detected.
- Evidence:
  - Canonical path enforces strict artifact mode using `TrainedModel::from_artifact_bytes_with_mode(..., Strict)` before predictor execution.
  - Compatibility path remains available for artifact utility prediction.
  - Sources: `bindings/python/src/lib.rs`, `crates/engine/src/lib.rs`, `crates/predictor/src/lib.rs`.

### naming/path drift
- Result: No drift detected.
- Evidence:
  - Planned files and implemented paths match (`bindings/python/src/lib.rs`, `bindings/python/alloygbm/regressor.py`, test paths in `bindings/python/tests/`).

## Resolution
- `accepted-with-rationale`: No drift found; implementation is aligned with declared `v0.5.2` contracts and boundaries.
