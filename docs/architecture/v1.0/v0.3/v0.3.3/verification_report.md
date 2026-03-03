# AlloyGBM v0.3.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.3/v0.3.3`
- Date: 2026-02-27

## Acceptance Criteria Matrix
- Criterion: (1) `purged_time_series_splits` and `purged_panel_splits` are implemented and exported from package API.
- Evidence:
  - [validation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/validation.py) defines both helpers.
  - [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py) exports both helpers.
  - Runtime export checks in `test_runtime_import_exposes_metric_helpers` inside [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py).
- Status: PASS

- Criterion: (2) Split outputs are deterministic and satisfy purge/embargo/no-overlap invariants.
- Evidence:
  - [test_validation_splits.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_validation_splits.py):
    - `test_time_series_splits_are_deterministic`
    - `test_time_series_splits_enforce_no_overlap_and_windows`
    - `test_panel_splits_use_time_buckets_across_groups`
  - Targeted run: `python3 -m unittest discover -s bindings/python/tests -p 'test_validation_splits.py'` -> PASS (`Ran 7 tests`, `OK`).
- Status: PASS

- Criterion: (3) Invalid inputs/configurations raise explicit `ValueError` semantics.
- Evidence:
  - [test_validation_splits.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_validation_splits.py):
    - `test_invalid_split_parameters_raise_value_error`
    - `test_invalid_data_shapes_raise_value_error`
    - `test_extreme_windows_that_remove_training_rows_raise_value_error`
- Status: PASS

- Criterion: (4) Existing regressor/runtime and metric tests continue to pass unchanged.
- Evidence:
  - Full suite run (`python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`) includes:
    - [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
    - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py)
    - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
  - Result: PASS (`Ran 52 tests`, `OK`).
- Status: PASS

- Criterion: (5) `cargo fmt -- --check` passes.
- Evidence:
  - Command run in this verification pass: `cargo fmt -- --check` -> PASS.
- Status: PASS

- Criterion: (6) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence:
  - Command run in this verification pass: `cargo clippy --workspace --all-targets -- -D warnings` -> PASS.
- Status: PASS

- Criterion: (7) `cargo test --workspace` passes.
- Evidence:
  - Command run in this verification pass: `cargo test --workspace` -> PASS.
- Status: PASS

- Criterion: (8) `cargo doc --workspace --no-deps` passes.
- Evidence:
  - Command run in this verification pass: `cargo doc --workspace --no-deps` -> PASS.
- Status: PASS

- Criterion: (9) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence:
  - Command run in this verification pass: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 52 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- Added: [test_validation_splits.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_validation_splits.py)
- Updated: [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)

## Commands Executed
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_validation_splits.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 52 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Split helpers are currently time-axis based and do not perform group balancing/stratification.
- Parent `v0.3` closeout is still pending parent-layer rollup artifacts.

## Final Readiness
- Ready: Yes (`v0.3.3` acceptance criteria satisfied).
- Required follow-up before merge/release: decide whether to open `v0.3.4` for final polish or close parent `v0.3` with rollup implementation/verification artifacts.
