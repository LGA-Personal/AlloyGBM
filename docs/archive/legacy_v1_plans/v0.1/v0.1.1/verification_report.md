# v0.1.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.1.1`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion: `TrainParams` includes `row_subsample`, `col_subsample`, `early_stopping_rounds`, and `min_validation_improvement`, with invalid-value rejection.
- Evidence: `crates/core/src/lib.rs` updates plus passing tests:
  - `rejects_invalid_row_subsample`
  - `rejects_invalid_col_subsample`
  - `rejects_invalid_early_stopping_rounds`
  - `rejects_negative_min_validation_improvement`
- Status: PASS

- Criterion: `IterationControls` represents subsampling and validation early-stopping policy with validation checks.
- Evidence: `crates/engine/src/lib.rs` updates and passing test `iteration_controls_reject_invalid_values` (including invalid subsample/early-stopping builder paths).
- Status: PASS

- Criterion: engine validation-aware iterative training path reports validation loss trace and can stop on plateau.
- Evidence: `crates/engine/src/lib.rs` includes `fit_iterations_with_validation_summary(...)`; passing tests:
  - `validation_early_stopping_requires_validation_dataset`
  - `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason`
- Status: PASS

- Criterion: Python `GBMRegressor` exposes/validates new params in constructor/get/set flows.
- Evidence: `bindings/python/alloygbm/regressor.py` and passing `bindings/python/tests/test_regressor_contract.py` (`Ran 7 tests`, `OK`).
- Status: PASS

- Criterion: verification command gates pass.
- Evidence:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
  all succeeded in this pass.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.1` scope.

## Residual Risks
- Subsampling behavior is deterministic baseline (prefix-based) and intentionally not final stochastic sampling.
- Parent `v0.1` rollup verification artifacts are still pending until more child-layer execution is completed.

## Final Readiness
- Ready: Yes (for `v0.1.1` scoped contract-lock and verification goals).
