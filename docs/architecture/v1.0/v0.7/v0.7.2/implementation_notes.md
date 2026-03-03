# AlloyGBM v0.7.2 Implementation Notes

## Summary of What Was Built
- Executed `v0.7.2` by replacing `v0.7.1` baseline contribution assignment with exact Shapley contribution math over the current tree payload semantics.
- Updated [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs):
  - added model-structure construction for tree roots, node lookup, and split-feature indexing,
  - implemented conditional expectation evaluator `expected_prediction_for_subset(...)` using unknown-feature branch weighting from node cover counts,
  - implemented exact Shapley summation over subset expectations for split features,
  - kept artifact-backed API shape stable (`ShapExplanationBatch`, `explain_rows_from_artifact_bytes`, `global_importance_*`),
  - retained deterministic validation and additivity checks.
- Expanded SHAP tests to lock exact-math outcomes:
  - expected-value check (`2.25`) for fixture model,
  - exact per-row contribution matrix,
  - zero contribution for unused features,
  - compatibility and validation regressions preserved.

## Non-Intuitive Decisions
- Decision: compute exact Shapley over distinct split features only, then project back into full feature vector.
- Reason: model output depends only on split features; non-split features must receive zero contribution while keeping exactness and reducing subset count.
- Impact: exactness is preserved for model semantics and computation stays tractable for typical small split-feature sets.

- Decision: add explicit split-feature guardrail (`MAX_EXACT_SPLIT_FEATURES = 20`).
- Reason: exact subset enumeration is exponential and can become impractical for very wide split-feature sets.
- Impact: deterministic contract violation is returned for unsupported wide cases instead of silent degradation.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted in `v0.7.2/plan.md`.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Public SHAP API signatures remained unchanged.
- `engine`/`predictor` behavior was not modified.
- Exact-math internals were introduced entirely within `crates/shap` as scoped.

## Known Gaps Deferred to Next Layer
- Python SHAP bridge/regressor methods remain deferred to the later bridge slice.
- Interaction/approximate SHAP modes remain out of scope.
- Performance optimization beyond exactness-first implementation remains deferred.

## Follow-Up Actions
- Plan and execute `docs/architecture/v1.0/v0.7/v0.7.3/plan.md` as the next layer target.
- Preserve new exact-value fixtures as regression anchors for future SHAP integration work.
