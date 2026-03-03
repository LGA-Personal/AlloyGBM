# AlloyGBM v0.3.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.3/v0.3.2`
- Date: 2026-02-27

## Acceptance Criteria Matrix
- Criterion: (1) `rank_ic`, `hit_rate`, and `icir` are implemented and exported from package API.
- Evidence:
  - [evaluation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/evaluation.py) defines all three helpers.
  - [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py) exports `rank_ic`, `hit_rate`, and `icir`.
  - Runtime export checks in `test_runtime_import_exposes_metric_helpers` inside [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py).
- Status: PASS

- Criterion: (2) Finance metric semantics are deterministic and covered by fixture-based tests.
- Evidence:
  - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
    - `test_rank_ic_perfect_and_inverse_order`
    - `test_rank_ic_uses_average_rank_for_ties`
    - `test_hit_rate_default_threshold`
    - `test_hit_rate_with_non_zero_threshold`
    - `test_icir_computes_mean_over_population_std`
    - `test_icir_zero_variance_fallback`
  - Targeted run: `python3 -m unittest discover -s bindings/python/tests -p 'test_evaluation_metrics.py'` -> PASS (`Ran 19 tests`, `OK`).
- Status: PASS

- Criterion: (3) Invalid inputs are rejected with explicit `ValueError` semantics.
- Evidence:
  - [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
    - `test_finance_pair_metrics_reject_mismatched_lengths`
    - `test_hit_rate_rejects_non_finite_threshold`
    - `test_icir_rejects_empty_or_non_finite_values`
- Status: PASS

- Criterion: (4) Existing `v0.2` regressor/runtime tests continue to pass unchanged.
- Evidence:
  - Full suite run (`python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`) includes existing:
    - [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
    - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
  - Result: PASS (`Ran 45 tests`, `OK`).
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
  - Command run in this verification pass: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 45 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- Updated: [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py)
- Updated: [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)

## Commands Executed
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_evaluation_metrics.py'`
- Result: PASS (`Ran 19 tests`, `OK`)

- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 45 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Leakage guardrail helpers (purged/embargo/time-aware splits) are still pending in `v0.3.3`.
- Optional tail metrics remain out of scope for this slice.

## Final Readiness
- Ready: Yes (`v0.3.2` acceptance criteria satisfied).
- Required follow-up before merge/release: add and execute `v0.3.3` plan for leakage split tooling.
