# AlloyGBM v0.0.12 Plan (v0.1 Milestone Closeout)

## Objective
Complete `v0.1` milestone closeout by consolidating child-layer evidence into parent-layer artifacts, resolving the remaining historical process gap, and advancing architecture state to the next planning target.

## Scope
- In scope:
  - Add final `v0.0.12` closeout artifacts for this layer.
  - Add parent-layer closeout artifacts at `docs/architecture/v1.0/v0.1/`.
  - Backfill `docs/architecture/v1.0/v0.1/v0.0.1/verification_report.md` to close the historical missing-verification artifact.
  - Re-run `v0.1` verification command set and capture fresh evidence.
  - Update `docs/architecture/state/layer_index.yaml` to reflect `v0.1` completion and next suggested target.
- Out of scope:
  - New trainer/runtime functionality.
  - Model format/schema changes.
  - New CI workflow behavior beyond existing `v0.0.11` closeout checks.
  - Any `v0.2+` implementation work.

## Deliverables
1. `v0.0.12` layer artifact package:
  - `docs/architecture/v1.0/v0.1/v0.0.12/plan.md`
  - `docs/architecture/v1.0/v0.1/v0.0.12/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/v0.0.12/verification_report.md`
2. Parent `v0.1` closeout package:
  - `docs/architecture/v1.0/v0.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/verification_report.md`
3. Historical gap closure:
  - `docs/architecture/v1.0/v0.1/v0.0.1/verification_report.md`
4. State progression:
  - updated `docs/architecture/state/layer_index.yaml` with `v0.0.12` and `v0.1` statuses and next suggested target.

## Implementation Plan
1. Create `v0.0.12` planning artifact and freeze closeout scope.
2. Consolidate `v0.0.2` through `v0.0.11` evidence into parent `v0.1` implementation notes.
3. Backfill `v0.0.1` verification report from current acceptance criteria and rerun command evidence.
4. Execute verification command set and Python contract checks.
5. Record `v0.0.12` and parent `v0.1` verification artifacts with criterion-to-evidence mapping.
6. Update layer state index to advance target beyond `v0.1` closeout.

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.1/v0.0.12/implementation_notes.md` exists and documents closeout decisions.
2. `docs/architecture/v1.0/v0.1/v0.0.12/verification_report.md` exists and maps every criterion in this plan to evidence.
3. `docs/architecture/v1.0/v0.1/implementation_notes.md` and `docs/architecture/v1.0/v0.1/verification_report.md` exist and summarize `v0.1` completion evidence.
4. `docs/architecture/v1.0/v0.1/v0.0.1/verification_report.md` exists, closing the historical missing artifact.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
10. Installed-wheel smoke confirms `GBMRegressor` construction and invalid-parameter rejection.
11. `docs/architecture/state/layer_index.yaml` no longer points to `v0.0.12` as active target and reflects completed artifact statuses.

## Risks and Mitigations
- Risk: closeout docs drift from actual command evidence.
  - Mitigation: run verification commands in this layer and record direct outputs.
- Risk: historical `v0.0.1` backfill could overstate original-era evidence.
  - Mitigation: clearly label evidence as current rerun validation against original criteria.
- Risk: dirty working tree can contaminate closeout edits.
  - Mitigation: keep edits restricted to explicit architecture artifact paths for this layer.

## Exit Condition
`v0.0.12` is complete when `v0.1` parent closeout artifacts exist, the `v0.0.1` verification gap is closed, verification commands are green, and state index points to the next planning target.
