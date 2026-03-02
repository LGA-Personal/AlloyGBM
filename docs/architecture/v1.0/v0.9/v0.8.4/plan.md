# AlloyGBM v0.8.4 Plan (v0.9 Migration and Compatibility Narrative Slice)

## Summary
- Goal: execute `v0.8.4` by finalizing the migration/compatibility narrative and producing command-backed compatibility evidence required to close `v0.9` parent rollup.
- Success criteria:
  - migration and compatibility guidance is explicit, actionable, and traceable to executable checks,
  - strict and legacy artifact behavior expectations are validated with focused commands,
  - layer state advances from child `v0.8.4` to parent `docs/architecture/v1.0/v0.9` closeout target.
- Audience: engineers and reviewers preparing `0.9.0` release-candidate hardening sign-off and `1.0.0` go/no-go review.

## Scope
### In Scope
- Create `v0.8.4` layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.8.4/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.8.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.8.4/verification_report.md`
- Author migration/compatibility narrative artifact:
  - `docs/architecture/v1.0/v0.9/v0.8.4/migration_compatibility_narrative.md`
- Update hardening baseline traceability:
  - refresh `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md` with `v0.8.4` bucket closure status and artifact links.
- Execute and record compatibility-focused command evidence:
  - strict/legacy artifact compatibility checks in `core`, `predictor`, and Python bridge/package surfaces.
- Update `docs/architecture/state/layer_index.yaml`:
  - mark `docs/architecture/v1.0/v0.9/v0.8.4` as `verified`,
  - advance `active_target` and `suggested_next_layer` to `docs/architecture/v1.0/v0.9` for parent rollup closeout.

### Out of Scope
- Authoring parent `v0.9` rollup artifacts (`docs/architecture/v1.0/v0.9/implementation_notes.md`, `verification_report.md`).
- New feature delivery (`1.0+` roadmap items, new objectives, backend changes).
- Model format version bump or API-breaking changes.
- Benchmark baseline thresholds in CI.

## Interfaces and Types
- Documentation/state interfaces in scope:
  - `docs/architecture/v1.0/v0.9/v0.8.4/migration_compatibility_narrative.md`
  - `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md`
  - `docs/architecture/state/layer_index.yaml`
- Executable compatibility boundaries validated:
  - core artifact compatibility report semantics (`strict` vs `legacy trees-only`),
  - predictor artifact loading for strict and legacy-supported layouts,
  - Python bridge canonical predictor behavior and bytes-like artifact contract handling.

Backward-compatibility expectations:
- no production API signature changes,
- no model format version changes,
- strict and legacy-compatible behavior remains as established in `v0.7.x` and carried through `v0.8.x`.

## Implementation Sequence
1. Author `v0.8.4` plan and lock narrative/compatibility-only scope.
2. Produce migration/compatibility narrative with explicit operator checklist and release-gate traceability.
3. Update `v0.8.1` hardening matrix to mark migration bucket closure artifacts.
4. Run compatibility-focused commands plus full gate commands.
5. Write `implementation_notes.md` and `verification_report.md` with criterion-to-evidence mapping.
6. Update `layer_index.yaml` to set next target to parent `docs/architecture/v1.0/v0.9`.

## Test Cases and Scenarios
- Documentation and traceability checks:
  - migration narrative exists with compatibility policy, migration impact summary, and operator checklist,
  - hardening matrix reflects `v0.8.4` closure for migration/compatibility bucket.
- Compatibility command checks:
  - `cargo test -p alloygbm-core required_section_compatibility_report_classifies_strict_and_legacy_layouts`
  - `cargo test -p alloygbm-core strict_compatibility_allows_optional_categorical_state_section`
  - `cargo test -p alloygbm-predictor predictor_accepts_legacy_trees_only_artifact`
  - `cargo test -p alloygbm-python canonical_bridge_rejects_legacy_trees_only_artifacts`
  - `python3 -m unittest bindings/python/tests/test_regressor_contract.py`
- Full non-regression gate checks:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.8.4/plan.md` is present and decision-complete.
2. `docs/architecture/v1.0/v0.9/v0.8.4/migration_compatibility_narrative.md` exists and includes migration impact, compatibility policy, and operator checklist.
3. `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md` is updated to record migration/compatibility bucket closure in `v0.8.4`.
4. `docs/architecture/v1.0/v0.9/v0.8.4/implementation_notes.md` is present.
5. `docs/architecture/v1.0/v0.9/v0.8.4/verification_report.md` is present with criterion-to-evidence mapping.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
11. Compatibility-focused command checks in core/predictor/python surfaces pass.
12. `docs/architecture/state/layer_index.yaml` marks `v0.8.4` verified and advances `active_target`/`suggested_next_layer` to `docs/architecture/v1.0/v0.9`.

## Risks and Mitigations
- Risk: migration guidance can become descriptive but not executable.
  - Mitigation: require command-backed compatibility evidence and map each checklist item to concrete tests.
- Risk: compatibility assumptions drift from actual behavior.
  - Mitigation: rerun strict/legacy compatibility tests in this slice and preserve full gate reruns.
- Risk: state progression advances before evidence is complete.
  - Mitigation: update `layer_index.yaml` only after all artifacts and verification evidence are present.

## Assumptions and Defaults
- `v0.8.4` remains a hardening/documentation/state slice unless verification reveals regressions.
- Parent `v0.9` rollup is the immediate next target after this slice.
- Existing benchmark reproducibility artifacts from `v0.8.3` remain baseline inputs, not reimplemented here.
