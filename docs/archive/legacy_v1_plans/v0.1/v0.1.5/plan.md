# AlloyGBM v0.1.5 Plan (v0.1 Depth-Limited Multi-Node Round Growth)

## Objective
Close a remaining `v0.1` algorithm-depth gap by replacing single-stump-per-round behavior with depth-limited multi-node split growth per boosting round, while preserving deterministic subsampling and validation checkpoint semantics established in `v0.1.2` and `v0.1.4`.

## Scope
- In scope:
  - Update engine iterative training so `controls.rounds` is the round cap, while `TrainParams.max_depth` constrains per-round node expansion depth.
  - Grow multiple node-conditioned splits within a round (breadth-first) using backend histogram/split primitives on node slices.
  - Ensure non-root split contributions apply only when ancestor path conditions are satisfied during both:
    - model inference (`predict_row` / `predict_batch`),
    - validation-loss simulation during training.
  - Preserve model artifact compatibility by reusing existing stump payload fields and encoding tree-local node identity through `split.node_id`.
  - Keep validation plateau rollback semantics correct when rounds can now emit multiple stumps.
  - Add/adjust engine tests to cover depth growth semantics and path-conditioned inference behavior.
- Out of scope:
  - New public Python API surface changes.
  - SIMD/performance optimization work.
  - Parent `v0.1` rollup artifacts.
  - Full predictor crate migration beyond current engine model behavior.

## Deliverables
1. Engine behavior package:
  - `crates/engine/src/lib.rs` supports depth-limited per-round multi-node split growth and path-aware stump application.
2. Verification package:
  - updated engine unit tests proving round-cap semantics and node-path gating.
  - `docs/architecture/v1.0/v0.1/v0.1.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/v0.1.5/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.5` completion and next target.

## Implementation Sequence
1. Add `v0.1.5` plan artifact.
2. Refactor iterative training loop to build node-conditioned splits up to per-round depth budget.
3. Introduce path-aware prediction application helpers for inference and validation simulation.
4. Preserve and adapt validation plateau rollback by truncating stumps with per-round stump counts.
5. Update tests for new depth semantics and add direct path-gating coverage.
6. Run verification command gates and record criterion-mapped evidence.

## Acceptance Criteria
1. Iterative training no longer caps rounds by `max_depth`; `effective_round_cap == controls.rounds`.
2. Within a round, engine can commit more than one split when `max_depth > 1` and valid child splits exist.
3. Non-root split contributions are only applied when ancestor path conditions match the row (inference + validation path).
4. Validation plateau rollback truncates model stumps and summary traces consistently to `best_validation_round` when rounds contain multiple stumps.
5. Existing gain/min-row/min-leaf/loss-improvement control contracts remain passing.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: path-conditioned split application can desynchronize training and inference behavior.
  - Mitigation: share explicit path-check helpers and add direct path-gating unit coverage.
- Risk: multi-stump-per-round behavior can break validation rollback semantics.
  - Mitigation: track per-round committed stump counts and truncate by round boundary.
- Risk: introducing new tree identity fields could require artifact format changes.
  - Mitigation: keep payload schema stable and encode tree-local identity through existing `split.node_id`.

## Exit Condition
`v0.1.5` is complete when depth-limited multi-node round growth is active, path-conditioned inference is verified, rollback semantics remain correct, and full verification command gates pass.
