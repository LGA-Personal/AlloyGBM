# AlloyGBM v0.3 Implementation Notes (Parent Rollup)

## Summary of What Was Built
- Completed `v0.3` through child slices `v0.2.1` to `v0.2.3` with full plan/implementation/verification artifacts.
- Delivered sklearn-style wrapper contract hardening in [regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - native-backed `fit`/`predict` flow,
  - stable parameter roundtrip (`get_params`/`set_params`),
  - deterministic input normalization for NumPy-like/pandas-like/Polars-like containers.
- Extended native bridge entrypoints in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - artifact-based training/prediction path,
  - explicit round-count control (`rounds`) for training.
- Consolidated verification evidence via contract tests and runtime wheel integration tests in:
  - [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
  - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)

## Non-Intuitive Decisions
- Decision: decompose `v0.3` into three focused child slices instead of one broad implementation pass.
- Reason: isolate risk across bridge wiring (`v0.2.1`), adapter normalization (`v0.2.2`), and estimator round-control (`v0.2.3`) with verification at each boundary.
- Impact: stronger traceability and easier rollback/debugging for regressions in wrapper behavior.

- Decision: keep adapter handling dependency-agnostic via duck typing (`to_numpy`/`to_list`/`tolist`).
- Reason: support common dataframe ecosystems without introducing hard runtime dependencies into the package.
- Impact: broader compatibility while preserving deterministic validation paths.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none required.

## Boundary/Interface Changes vs Plan
- `GBMRegressor` now consistently executes native-backed training and prediction rather than scaffold fallback behavior.
- `GBMRegressor` parameter surface now includes explicit estimator round control (`n_estimators`) with bridge forwarding.
- Native extension training entrypoint accepts explicit `rounds` and validates non-zero values.
- Scope remained inside `v0.3` parent boundaries; no ranking, SHAP expansion, categorical expansion, or backend/perf campaigns were introduced.

## Known Gaps Deferred to Next Layer
- `v0.4` finance-evaluation scope (metrics and leakage guardrails) remains unimplemented in this layer.
- Bridge training currently expects pre-binned integer-valued non-negative features; full continuous-feature quantization remains deferred.

## Follow-Up Actions
- Open and execute `docs/architecture/v1.0/v0.4/v0.3.1` as the first child slice of `0.4.0`.
- Keep `docs/architecture/state/layer_index.yaml` aligned as `v0.4` child layers progress.
