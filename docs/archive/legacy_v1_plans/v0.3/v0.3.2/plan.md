# AlloyGBM v0.3.2 Technical Plan (v0.3 Finance Metrics Slice)

## Summary
- Goal: execute the second `v0.3` child slice by adding finance-oriented evaluation metrics (`rank-IC`, `hit-rate`, `ICIR`) on top of verified `v0.3.1` baseline metrics.
- Success criteria:
  - finance metric helpers are available from the Python package API,
  - metric semantics and edge-case defaults are deterministic and test-covered,
  - existing `v0.2` wrapper/runtime behavior remains unchanged and all verification gates stay green.
- Audience: engineers implementing `v0.3.2` and reviewers validating readiness for the leakage-guardrail slice (`v0.3.3`).

## Scope
### In Scope
- Add finance-oriented metric helpers to `bindings/python/alloygbm/evaluation.py`:
  - `rank_ic(y_true, y_pred) -> float`
  - `hit_rate(y_true, y_pred, *, threshold: float = 0.0) -> float`
  - `icir(ic_values) -> float`
- Extend package exports in `bindings/python/alloygbm/__init__.py`.
- Extend evaluation test coverage with deterministic correctness fixtures and invalid-input semantics for the new helpers.
- Ensure runtime package integration test confirms new helpers are exported and callable from installed wheel.
- Produce layer artifacts:
  - `docs/architecture/v1.0/v0.3/v0.3.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/v0.3.2/verification_report.md`

### Out of Scope
- Purged K-fold / embargo / time-aware split tooling (`v0.3.3` scope).
- Changes to `GBMRegressor` fit/predict behavior or native Rust bridge signatures.
- Ranking objective training, SHAP expansion, categorical expansion, backend/performance work.

## Interfaces and Types
- `bindings/python/alloygbm/evaluation.py`:
  - add `rank_ic`, `hit_rate`, `icir`,
  - keep baseline `rmse`, `mae`, `r2_score`, `pearson_correlation` semantics unchanged.
- `bindings/python/alloygbm/__init__.py`:
  - additive exports only for finance metric helpers.
- `bindings/python/tests/test_evaluation_metrics.py`:
  - add deterministic correctness and failure tests for `rank_ic`, `hit_rate`, `icir`.
- `bindings/python/tests/test_native_runtime_integration.py`:
  - extend export checks for finance metric helper availability.

Backward-compatibility expectations:
- no breaking changes to existing public names or signatures in `GBMRegressor` and `v0.3.1` metrics.
- finance helpers are additive and independently callable.

## Implementation Sequence
1. Add finance metric helper implementations in `evaluation.py` using existing validation utilities.
2. Export new helpers from `__init__.py`.
3. Add/extend tests for finance metric formulas, deterministic behavior, and invalid input handling.
4. Run full verification gates and record evidence in layer artifacts.
5. Update `docs/architecture/state/layer_index.yaml` after verification.

## Test Cases and Scenarios
- Unit cases:
  - `rank_ic` returns `1.0` for perfectly aligned ranking and `-1.0` for perfectly inverse ranking.
  - `rank_ic` handles ties deterministically (average-rank tie policy).
  - `hit_rate` matches expected directional accuracy with default and non-zero threshold.
  - `icir` matches expected `mean(ic) / std(ic)` fixture values and zero-variance fallback behavior.
- Integration cases:
  - installed package runtime exposes and executes finance metric helpers.
- Failure and edge cases:
  - mismatched lengths for pair metrics,
  - empty `ic_values`,
  - non-finite numeric inputs (`nan`, `inf`, `-inf`),
  - non-finite `threshold`.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `rank_ic`, `hit_rate`, and `icir` are implemented and exported from package API.
2. Finance metric semantics are deterministic and covered by fixture-based tests.
3. Invalid inputs are rejected with explicit `ValueError` semantics.
4. Existing `v0.2` regressor/runtime tests continue to pass unchanged.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: finance metric definitions diverge from common quant conventions.
  - Mitigation: encode formula defaults explicitly in tests and this plan.
- Risk: tie handling in rank-based metrics is underspecified.
  - Mitigation: lock tie behavior to average-rank assignment and test it directly.
- Risk: additive metrics accidentally alter existing baseline metric behavior.
  - Mitigation: keep previous `v0.3.1` tests and runtime checks mandatory in full suite runs.

## Assumptions and Defaults
- `rank_ic` uses Spearman-style rank correlation by applying average-rank tie handling, then Pearson correlation on ranks.
- `hit_rate` computes directional agreement using three-way sign around `threshold` (`+1`, `0`, `-1`), with exact-threshold values treated as neutral (`0`).
- `icir` computes `mean(ic_values) / population_std(ic_values)` and returns `0.0` when variance is zero.
- No sample-weight support in this slice.
