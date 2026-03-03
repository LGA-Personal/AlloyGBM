# AlloyGBM v0.1.10 Plan (v0.1 Parent Closeout Rollup)

## Objective
Close out `v0.1` by producing parent-layer implementation and verification rollups that consolidate evidence from verified child slices `v0.1.1` through `v0.1.9`, confirm gate-command health, and mark the parent `v0.1` layer as verified.

## Scope
- In scope:
  - Create parent rollup artifacts:
    - `docs/architecture/v1.0/v0.1/implementation_notes.md`
    - `docs/architecture/v1.0/v0.1/verification_report.md`
  - Create `v0.1.10` layer artifacts documenting the closeout operation.
  - Map parent `v0.1` plan objectives to concrete evidence from completed child layers.
  - Re-run verification command gates and record outputs.
  - Update `docs/architecture/state/layer_index.yaml` to mark:
    - `v0.1.10` as verified,
    - parent `v0.1` as verified,
    - next target progression.
- Out of scope:
  - New algorithmic or API behavior changes.
  - Benchmark harness redesign or full performance-regression campaign.
  - Parent `v1.0` rollup closeout.

## Deliverables
1. Parent closeout package:
  - `docs/architecture/v1.0/v0.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/verification_report.md`
2. Child closeout package:
  - `docs/architecture/v1.0/v0.1/v0.1.10/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/v0.1.10/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.10` and parent `v0.1` completion.

## Implementation Sequence
1. Create `v0.1.10` plan artifact.
2. Synthesize `v0.1` parent implementation rollup from verified child implementation notes.
3. Build parent verification matrix mapping `v0.1` plan goals to child-layer evidence.
4. Re-run verification command gates and capture pass/fail evidence.
5. Author `v0.1.10` implementation/verification artifacts.
6. Update layer state index to record closeout and next target.

## Acceptance Criteria
1. Parent `v0.1` implementation rollup exists and summarizes all `v0.1.1`–`v0.1.9` contributions.
2. Parent `v0.1` verification report exists with criterion-mapped evidence for `v0.1` goals:
  - depth-limited histogram CART growth behavior,
  - shrinkage + row/column subsampling + validation early stopping controls,
  - row/batch inference from serialized artifacts.
3. Parent report includes explicit readiness statement and any residual risks.
4. `docs/architecture/state/layer_index.yaml` marks parent `v0.1` as verified and records `v0.1.10` as verified.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: parent closeout may over-claim evidence not tied to explicit child verification artifacts.
  - Mitigation: cite only documented child-layer verification evidence and current command outputs.
- Risk: remaining benchmark/performance unknowns could be mistaken as solved by closeout.
  - Mitigation: state residual benchmark risk explicitly in parent verification report.
- Risk: state index drift after closeout.
  - Mitigation: update state index in same pass as artifacts and verify consistency.

## Exit Condition
`v0.1.10` is complete when parent `v0.1` rollup artifacts are written, command gates pass, `v0.1` is marked verified in layer state, and next target is identified.
