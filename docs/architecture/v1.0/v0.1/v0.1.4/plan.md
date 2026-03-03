# AlloyGBM v0.1.4 Plan (v0.1 Validation Best-Checkpoint Semantics)

## Objective
Close a remaining `v0.1` training-loop semantics gap by ensuring validation early-stopping returns the best-validation checkpoint model rather than retaining the last plateau-triggering round.

## Scope
- In scope:
  - Update engine validation early-stopping flow to rollback model state to `best_validation_round` when stopping for validation plateau.
  - Keep loss/sample traces and final-loss fields aligned with the rolled-back checkpoint.
  - Add/update engine unit tests proving plateau stop reason is preserved while returned model/summary reflect best-validation checkpoint semantics.
  - Produce `v0.1.4` implementation and verification artifacts.
- Out of scope:
  - Full-depth CART growth redesign beyond current stump-level iterative behavior.
  - New Python API parameters or sklearn-surface changes.
  - SIMD/performance work and parent `v0.1` rollup artifacts.

## Deliverables
1. Engine semantics package:
  - `crates/engine/src/lib.rs` applies best-checkpoint rollback on validation plateau stop.
2. Verification package:
  - engine tests proving rollback behavior is active and deterministic.
  - `docs/architecture/v1.0/v0.1/v0.1.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/v0.1.4/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.4` completion and next target.

## Implementation Sequence
1. Create `v0.1.4` plan artifact.
2. Implement validation-plateau rollback to best checkpoint in engine summary flow.
3. Update/add unit tests for returned-round/model semantics under plateau stopping.
4. Run verification command gates and capture results.
5. Write layer implementation/verification artifacts and update state index.

## Acceptance Criteria
1. When stop reason is `ValidationLossPlateau`, returned model state is rolled back to `best_validation_round` (including zero-round case).
2. Summary fields (`rounds_completed`, loss traces, sampled-count traces, `final_loss`, `final_validation_loss`) align with the rolled-back checkpoint.
3. Existing subsampling and validation-stop contract tests remain passing.
4. `cargo fmt -- --check` passes.
5. `cargo clippy --workspace --all-targets -- -D warnings` passes.
6. `cargo test --workspace` passes.
7. `cargo doc --workspace --no-deps` passes.
8. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: rollback can desynchronize summary arrays from returned model.
  - Mitigation: truncate/realign all per-round vectors and recompute final-loss fields from retained rounds.
- Risk: behavior change may invalidate previous assumptions in tests.
  - Mitigation: update targeted validation plateau tests to assert new best-checkpoint semantics explicitly.
- Risk: scope drift into broader objective/tuning logic.
  - Mitigation: limit changes to validation-plateau finalization path and corresponding tests only.

## Exit Condition
`v0.1.4` is complete when validation plateau stopping returns best-checkpoint model/summary semantics with passing engine tests and full verification command evidence.
