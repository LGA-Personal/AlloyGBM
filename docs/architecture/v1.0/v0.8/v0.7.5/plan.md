# AlloyGBM v0.7.5 Plan (v0.8 Parent Closeout and Evidence Rollup Slice)

## Summary
- Goal: execute `v0.7.5` as the parent-closeout slice for `v0.8` by publishing milestone-level implementation/verification artifacts that consolidate `v0.7.1` through `v0.7.4` evidence.
- Success criteria:
  - `docs/architecture/v1.0/v0.8/implementation_notes.md` and `docs/architecture/v1.0/v0.8/verification_report.md` are created and evidence-complete,
  - `v0.8` acceptance criteria are explicitly mapped to child-layer artifacts and command gates,
  - state index marks `v0.7.5` and parent `v0.8` as verified and advances to next top-level target.
- Audience: engineers and reviewers performing `0.8.0` milestone sign-off before moving to `v0.9`.

## Scope
### In Scope
- Parent closeout artifacts:
  - create `docs/architecture/v1.0/v0.8/implementation_notes.md`,
  - create `docs/architecture/v1.0/v0.8/verification_report.md`.
- Child closeout artifacts:
  - create `docs/architecture/v1.0/v0.8/v0.7.5/implementation_notes.md`,
  - create `docs/architecture/v1.0/v0.8/v0.7.5/verification_report.md`.
- Evidence consolidation:
  - map `v0.8` parent acceptance criteria to child slices `v0.7.1`..`v0.7.4`,
  - include current verification command evidence from this closeout run.
- State and continuity updates:
  - update `docs/architecture/state/layer_index.yaml` to:
    - mark `v0.7.5` verified,
    - mark parent `v0.8` verified,
    - set next target to `docs/architecture/v1.0/v0.9`,
  - synchronize context docs (`session_brief.md`, `handoff.md`) to the new active target.

### Out of Scope
- New SHAP runtime algorithm work in `crates/shap`.
- New Python SHAP API surface beyond `v0.7.4`.
- Model format changes or compatibility policy changes.
- Non-`v0.8` roadmap work (ranking, probabilistic outputs, constraints, GPU backends).

## Interfaces and Types
- No production code interface changes are expected in this slice.
- Documentation/state interfaces updated:
  - `docs/architecture/v1.0/v0.8/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/verification_report.md`
  - `docs/architecture/state/layer_index.yaml`
  - context continuity docs.

Backward-compatibility expectations:
- `GBMRegressor.fit/predict/shap_values/feature_importances` behavior remains unchanged from `v0.7.4`.
- Rust SHAP artifact-backed APIs remain unchanged.

## Deliverables
1. Parent closeout package:
  - `v0.8` implementation notes and verification report with consolidated evidence.
2. `v0.7.5` traceability package:
  - plan, implementation notes, and verification report.
3. State-transition package:
  - layer index marks `v0.7.5` + `v0.8` verified and points to `v0.9`.
4. Continuity package:
  - updated context brief and handoff aligned to post-`v0.8` target.

## Implementation Sequence
1. Author `v0.7.5` plan and lock closeout-only scope.
2. Draft parent `v0.8` implementation notes from `v0.7.1`..`v0.7.4` artifacts.
3. Draft parent `v0.8` verification report mapping all parent criteria to child evidence.
4. Run full verification gate commands and capture outputs for closeout evidence.
5. Write `v0.7.5` implementation notes and verification report.
6. Update layer index to mark `v0.7.5` + `v0.8` verified and advance next target to `v0.9`.
7. Update context docs to reflect new active target and immediate next actions.

## Test Cases and Scenarios
- Documentation and traceability checks:
  - parent `v0.8` reports include evidence links for all child slices and command gates.
- Verification commands:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- State transition checks:
  - `layer_index.yaml` reflects `v0.7.5` + `v0.8` verified and next target `v0.9`.

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.7.5/plan.md` exists and is decision-complete.
2. `docs/architecture/v1.0/v0.8/implementation_notes.md` is created and summarizes delivered `v0.8` scope across `v0.7.1`..`v0.7.4`.
3. `docs/architecture/v1.0/v0.8/verification_report.md` is created and maps parent criteria to child evidence.
4. `docs/architecture/v1.0/v0.8/v0.7.5/implementation_notes.md` is created.
5. `docs/architecture/v1.0/v0.8/v0.7.5/verification_report.md` is created.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. Python unittest suite passes.
11. `docs/architecture/state/layer_index.yaml` marks `v0.7.5` and parent `v0.8` as verified and advances to `docs/architecture/v1.0/v0.9`.

## Risks and Mitigations
- Risk: parent rollup can drift from child evidence.
  - Mitigation: anchor each parent criterion with direct links to child reports and current gate outputs.
- Risk: documentation-only slice could miss latent regressions.
  - Mitigation: rerun full verification gate during closeout and include outputs in reports.
- Risk: next target inference ambiguity after parent closeout.
  - Mitigation: explicitly set `active_target`/`suggested_next_layer` to `docs/architecture/v1.0/v0.9` in layer index.

## Assumptions and Defaults
- `v0.8` functional implementation is complete as of verified slices `v0.7.1`..`v0.7.4`.
- `v0.9` is the next top-level target following `v0.8` closeout.
- This slice does not require production code changes unless a verification regression appears.
