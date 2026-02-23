# v0.0.3 Implementation Notes

## Summary of What Was Built
- Added concrete deterministic CPU primitive behavior in `crates/backend_cpu/src/lib.rs`:
  - histogram aggregation over binned features
  - best split search using gain-based scoring
  - row partitioning from split candidate
  - gradient/hessian reductions for arbitrary row sets
- Added executable one-round training flow in `crates/engine/src/lib.rs`:
  - `SquaredErrorObjective` implementation
  - `Trainer::fit_one_round` contract-driven single-round path
  - `FitContractEvaluation` and `TrainRoundSummary` outputs
- Upgraded Python estimator baseline behavior in `bindings/python/alloygbm/regressor.py`:
  - implemented `fit` as constant-mean baseline
  - implemented `predict` with fitted-state and feature-shape checks
- Added focused tests:
  - backend aggregation/split/partition/reduction tests
  - engine one-round and contract-mismatch tests
  - Python unit tests for fit/predict baseline behavior and validation

## Non-Intuitive Decisions
- Kept the engine scope at single-round orchestration (`fit_one_round`) rather than introducing multi-tree loops. This preserves `v0.1` contracts-first intent and avoids crossing into full `0.2.0` algorithm scope.
- Implemented a minimal gain formula with epsilon stabilization to avoid division-by-zero behavior in tiny hessian cases while keeping logic deterministic.
- Python estimator intentionally returns constant predictions from `fit` target mean, serving as an executable API slice without claiming final model behavior.

## Plan Contradictions and Why
- No contradictions to `v0.0.3/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate boundary changes were made.
- Internal interface evolution aligns with planned scope:
  - `backend_cpu` moved from placeholders to baseline executable behavior.
  - `engine` gained one-round training path and baseline objective.
  - Python regressor moved from stub exceptions to minimal executable baseline.

## Known Gaps Deferred to v0.0.4+
- Multi-level tree growth and boosting-loop iteration are not implemented.
- No early stopping, subsampling, or regularization controls are wired into round execution.
- Python path is still baseline-only and not yet connected to native Rust model objects.

## Follow-Up Actions
- Plan `v0.0.4` for iterative tree growth loop and first end-to-end model artifact emission.
- Add integration tests that couple backend primitives and trainer flow under more diverse binned fixtures.
- Begin bridging Python estimator fit/predict to native runtime outputs once core training loop expands.
