# AlloyGBM v0.3.2 Implementation Notes

## Summary of What Was Built
- Implemented finance-oriented evaluation metrics in [evaluation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/evaluation.py):
  - `rank_ic(y_true, y_pred)` using Spearman-style average-rank tie handling followed by Pearson correlation on ranks.
  - `hit_rate(y_true, y_pred, threshold=0.0)` using three-way directional agreement around threshold.
  - `icir(ic_values)` using `mean(ic_values) / population_std(ic_values)` with zero-variance fallback.
- Extended package exports in [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py):
  - added `rank_ic`, `hit_rate`, and `icir`.
- Expanded deterministic test coverage in [test_evaluation_metrics.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_evaluation_metrics.py):
  - rank-IC perfect/inverse and tie-handling fixtures,
  - hit-rate default and non-zero-threshold fixtures,
  - ICIR formula and zero-variance fallback fixtures,
  - finance-metric error-path assertions.
- Extended runtime wheel integration checks in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py):
  - asserts finance metric helpers are exported and callable from installed package.
- Added layer planning + verification artifacts:
  - `docs/architecture/v1.0/v0.3/v0.3.2/plan.md`
  - `docs/architecture/v1.0/v0.3/v0.3.2/verification_report.md`

## Non-Intuitive Decisions
- Decision: define `rank_ic` by ranking each series with average-rank tie handling rather than implementing a separate complex tie-correction branch.
- Reason: this keeps semantics explicit and deterministic while aligning with expected Spearman-style behavior.
- Impact: tie behavior is now stable and directly asserted in tests.

- Decision: use population standard deviation (`N`) for `icir` and apply a small absolute tolerance for zero-variance fallback.
- Reason: avoids divide-by-near-zero blowups caused by floating-point noise on near-constant IC sequences.
- Impact: constant/near-constant series return deterministic `0.0` ICIR instead of extreme values.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Added three new public evaluation APIs (`rank_ic`, `hit_rate`, `icir`) exactly within planned scope.
- Kept `GBMRegressor` and native Rust bridge interfaces unchanged.
- Did not introduce leakage split helpers, ranking training logic, or backend/performance changes.

## Known Gaps Deferred to Next Layer
- `v0.3.3` remains open for purge/embargo/time-aware split tooling.
- Tail metrics remain optional/deferred beyond this slice.
- Sample-weight support remains out of scope for `v0.3.2`.

## Follow-Up Actions
- Update `docs/architecture/state/layer_index.yaml` to include `docs/architecture/v1.0/v0.3/v0.3.2` as `verified`.
- Open and execute `docs/architecture/v1.0/v0.3/v0.3.3/plan.md` for leakage guardrail helpers.
