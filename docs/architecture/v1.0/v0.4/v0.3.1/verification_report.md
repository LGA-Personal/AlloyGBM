# AlloyGBM v0.3.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.4/v0.3.1`
- Date: 2026-02-27

## Acceptance Criteria Matrix
- Criterion: (1) Baseline metric helpers exist and are exported from Python package API.
- Evidence:
  - [evaluation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/evaluation.py) provides `rmse`, `mae`, `r2_score`, and `pearson_correlation`.
  - [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py) exports all four helpers.
  - Runtime export assertion: `test_runtime_import_exposes_metric_helpers` in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py).
- Status: PASS

- Criterion: (2) Metric helper semantics are deterministic and validated with fixture-based tests.
- Evidence:
  - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
    - `test_perfect_predictions_metrics`
    - `test_inverse_order_metrics`
    - `test_non_trivial_fixture_metrics`
    - `test_metrics_are_deterministic_for_repeated_calls`
    - `test_r2_constant_target_fallback_behavior`
    - `test_pearson_returns_zero_for_zero_variance_series`
  - Explicit targeted run: `python3 -m unittest discover -s bindings/python/tests -p 'test_evaluation_metrics.py'` -> PASS (`Ran 10 tests`, `OK`).
- Status: PASS

- Criterion: (3) Invalid inputs are rejected with explicit `ValueError` semantics.
- Evidence:
  - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
    - `test_mismatched_lengths_raise_value_error`
    - `test_empty_inputs_raise_value_error`
    - `test_non_finite_values_raise_value_error`
- Status: PASS

- Criterion: (4) Existing `v0.3` regressor contract/runtime tests remain passing unchanged.
- Evidence:
  - Full suite run (`python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`) includes:
    - [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
    - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
  - Result: PASS (`Ran 36 tests`, `OK`).
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
  - Command run in this verification pass: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 36 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- None in this verification pass (existing `v0.3.1` tests already covered all criteria).

## Commands Executed
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_evaluation_metrics.py'`
- Result: PASS (`Ran 10 tests`, `OK`)

- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 36 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Sample-weighted metrics are intentionally out of scope for `v0.3.1`.
- Finance-specific ranking metrics and leakage split helpers are deferred to later `v0.4` child slices.

## Final Readiness
- Ready: Yes (`v0.3.1` acceptance criteria satisfied).
- Required follow-up before merge/release: update `docs/architecture/state/layer_index.yaml` to mark `docs/architecture/v1.0/v0.4/v0.3.1` as `verified`.
