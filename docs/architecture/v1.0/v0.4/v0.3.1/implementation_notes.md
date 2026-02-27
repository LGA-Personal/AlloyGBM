# AlloyGBM v0.3.1 Implementation Notes

## Summary of What Was Built
- Implemented baseline evaluation metrics for `v0.3.1` in [evaluation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/evaluation.py):
  - `rmse(y_true, y_pred)`
  - `mae(y_true, y_pred)`
  - `r2_score(y_true, y_pred)`
  - `pearson_correlation(y_true, y_pred)`
- Added deterministic input validation shared by metric helpers:
  - equal non-zero lengths,
  - finite numeric values only,
  - explicit conversion and sequence-like adapter handling (`to_numpy`/`to_list`/`tolist`).
- Exported metric helpers through package API in [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py).
- Added dedicated test coverage in [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
  - deterministic correctness fixtures,
  - repeated-call determinism checks,
  - adapter input handling,
  - invalid/malformed input error semantics,
  - constant-target and constant-variance edge behavior.
- Extended runtime integration evidence in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py) to assert new metric helpers are exported on the installed package and callable.
- Executed layer gate commands successfully:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` (`Ran 36 tests`, `OK`)

## Non-Intuitive Decisions
- Decision: keep evaluation input validation local to `evaluation.py` instead of refactoring `GBMRegressor` validators.
- Reason: this isolates `v0.3.1` changes to additive evaluation scope and avoids risk of regressions in already-verified `v0.3` regressor behavior.
- Impact: there is some duplicated adapter/validation logic between regressor and evaluation paths, but scope and risk stayed controlled for this slice.

- Decision: define constant-series Pearson behavior as `0.0` and constant-target `R2` fallback as `1.0` when residuals are zero else `0.0`.
- Reason: these defaults were explicitly chosen in the `v0.3.1` plan to prevent ambiguous runtime behavior.
- Impact: deterministic edge semantics are now locked by tests and can be documented consistently in later layers.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Added new public Python API exports:
  - `rmse`, `mae`, `r2_score`, `pearson_correlation`.
- Existing `GBMRegressor` API and native Rust bridge interfaces were not modified.
- No ranking, leakage-split, SHAP, categorical, backend, or performance scope was introduced.

## Known Gaps Deferred to Next Layer
- `v0.3.2+` scope remains open:
  - finance-ranking metrics (`rank-IC`, `hit-rate`, `ICIR`),
  - purge/embargo/time-aware split helpers.
- Sample-weight support is intentionally not included in `v0.3.1`.

## Follow-Up Actions
- Update `docs/architecture/state/layer_index.yaml` to mark `docs/architecture/v1.0/v0.4/v0.3.1` as `verified`.
- Open and execute `docs/architecture/v1.0/v0.4/v0.3.2/plan.md` for finance-oriented metrics.
