# AlloyGBM v0.3.1 Technical Plan (v0.4 Baseline Metrics Slice)

## Summary
- Goal: execute the first `v0.4` child slice by adding a decision-complete baseline metric API (`RMSE`, `MAE`, `R2`, Pearson correlation) for wrapper workflows.
- Success criteria:
  - deterministic metric helpers are available from the Python package,
  - metric semantics and edge-case behavior are explicitly defined and test-covered,
  - all existing `v0.3` wrapper/runtime gates remain green.
- Audience: engineers implementing `v0.3.1` and reviewers validating readiness for later `v0.4` slices (finance metrics and leakage splits).

## Scope
### In Scope
- Add baseline metric helpers to Python package surface:
  - `rmse(y_true, y_pred) -> float`
  - `mae(y_true, y_pred) -> float`
  - `r2_score(y_true, y_pred) -> float`
  - `pearson_correlation(y_true, y_pred) -> float`
- Add shared numeric validation for metric helpers:
  - equal non-zero lengths,
  - finite numeric values only,
  - explicit and stable error messages for invalid input.
- Add deterministic tests for correctness fixtures and failure semantics.
- Produce layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.3.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.3.1/verification_report.md`

### Out of Scope
- Purged K-fold / embargo split helpers (later `v0.3.x` slices).
- Finance-ranking metrics (`rank-IC`, `ICIR`, `hit-rate`) beyond this baseline set.
- Changes to native Rust training/predictor APIs.
- Ranking objective training, SHAP expansion, categorical expansion, backend/performance work.

## Interfaces and Types
- `bindings/python/alloygbm/evaluation.py` (new):
  - metric functions listed above.
  - internal helper(s) for input normalization/validation dedicated to evaluation paths.
- `bindings/python/alloygbm/__init__.py`:
  - additive exports of baseline metric helpers.
- `bindings/python/tests/test_evaluation_metrics.py` (new):
  - deterministic fixture tests and error-path tests.
- Existing files that must remain behaviorally stable:
  - `bindings/python/alloygbm/regressor.py`
  - `bindings/python/tests/test_regressor_contract.py`
  - `bindings/python/tests/test_native_runtime_integration.py`

Backward-compatibility expectations:
- no signature or behavior changes to existing `GBMRegressor` methods in this slice.
- metric helpers are additive and independent; existing imports continue to work unchanged.

## Implementation Sequence
1. Create `evaluation.py` with baseline metric functions and shared validation helper(s).
2. Export new helpers through package `__init__.py` without removing existing exports.
3. Add `test_evaluation_metrics.py` with deterministic fixtures and explicit failure assertions.
4. Run full verification gates and record results in `implementation_notes.md` and `verification_report.md`.
5. Refresh `docs/architecture/state/layer_index.yaml` after verification completes.

## Test Cases and Scenarios
- Unit cases:
  - exact-value fixtures:
    - perfect agreement (`y_true == y_pred`) for zero error and correlation `1.0`,
    - inverse-order fixture for negative correlation and non-zero errors,
    - mixed-error fixture for non-trivial `RMSE`/`MAE`/`R2`.
  - deterministic repeated-call assertions for identical inputs.
- Integration cases:
  - train with `GBMRegressor`, compute predictions, then evaluate with new metric helpers in Python tests.
- Failure and edge cases:
  - mismatched lengths,
  - empty input vectors,
  - non-finite numeric values (`nan`, `inf`, `-inf`),
  - constant series correlation case.
- Acceptance test mapping:
  - baseline metric tests via `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`,
  - workspace quality gates:
    - `cargo fmt -- --check`,
    - `cargo clippy --workspace --all-targets -- -D warnings`,
    - `cargo test --workspace`,
    - `cargo doc --workspace --no-deps`.

## Acceptance Criteria
1. Baseline metric helpers exist and are exported from the Python package API.
2. Metric helper semantics are deterministic and validated with fixture-based tests.
3. Invalid inputs are rejected with explicit `ValueError` semantics documented in tests.
4. Existing `v0.3` regressor contract/runtime tests continue to pass unchanged.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: metric definitions diverge from expected quant/sklearn conventions.
  - Mitigation: define formulas and edge-case defaults explicitly in code comments/tests and verify against deterministic fixtures.
- Risk: adding shared validation utilities introduces unintended behavior changes in regressor paths.
  - Mitigation: keep evaluation validation local to `evaluation.py` for this slice; avoid refactoring regressor validators in `v0.3.1`.
- Risk: ambiguous constant-series handling causes unstable downstream behavior.
  - Mitigation: lock defaults in this plan and test them directly.

## Assumptions and Defaults
- Metric formulas:
  - `RMSE = sqrt(mean((y_true - y_pred)^2))`
  - `MAE = mean(abs(y_true - y_pred))`
  - `R2 = 1 - ss_res / ss_tot`, with constant-target fallback: return `1.0` when `ss_res == 0`, else `0.0`.
  - `pearson_correlation` uses standard Pearson correlation and returns `0.0` when either series has zero variance.
- Input defaults:
  - no sample-weight support in `v0.3.1`,
  - no NaN/Inf coercion; non-finite values are rejected,
  - inputs accepted as sequence-like values supported by existing adapter expectations.
