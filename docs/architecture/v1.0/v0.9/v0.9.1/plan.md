# AlloyGBM v0.9.1 Plan (Bug Triage and Deterministic Reproduction Slice)

## Summary
- Goal: execute the first `v0.9` child slice by triaging current benchmark/runtime anomalies, reproducing them deterministically, and landing low-risk fixes that improve evidence quality for later `v0.9.2+` work.
- Success criteria:
  - a prioritized bug triage artifact exists with reproducible commands and test mapping,
  - at least one validated triage defect is fixed in-repo with verification evidence,
  - layer artifacts and state progression are complete for handoff to `v0.9.2`.
- Audience: engineers preparing benchmark-expansion and optimization work in `v0.9.2` and `v0.9.3`.

## Scope
### In Scope
- Create `v0.9.1` layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.9.1/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.9.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.9.1/verification_report.md`
- Add triage artifact:
  - `docs/architecture/v1.0/v0.9/v0.9.1/bug_triage.md` with prioritized issues, deterministic reproduction commands, and fix-to-test mapping.
- Implement one low-risk triage fix for benchmark evidence correctness in:
  - `scripts/benchmark_avx2_compare.sh`.
- Verify the fix and non-regression commands.
- Update `docs/architecture/state/layer_index.yaml`:
  - mark `docs/architecture/v1.0/v0.9/v0.9.1` as `verified`,
  - set `active_target` and `suggested_next_layer` to `docs/architecture/v1.0/v0.9/v0.9.2`.

### Out of Scope
- Full shallow/deep benchmark expansion (`v0.9.2` scope).
- Broad performance tuning of training/inference kernels (`v0.9.3` scope).
- End-user documentation/tutorial authoring pass (`v0.9.6` scope).
- New roadmap features or API/model-format breaking changes.

## Interfaces and Types
- `scripts/benchmark_avx2_compare.sh` output contract:
  - preserve existing summary fields,
  - add explicit non-AVX2 handling so AVX2 delta is not presented as meaningful when AVX2 is unavailable for both runs.
- Documentation artifacts:
  - bug triage entries must include issue ID, severity, reproduction command, expected behavior, observed behavior, and verification linkage.
- Backward-compatibility expectations:
  - no Rust/Python API changes,
  - no model artifact contract changes.

## Implementation Sequence
1. Collect anomaly evidence from existing `v0.8.3` benchmark artifacts and rerun targeted commands for current machine behavior.
2. Author `bug_triage.md` with prioritized issue list and explicit reproduction commands.
3. Patch `scripts/benchmark_avx2_compare.sh` to avoid misleading AVX2 delta interpretation on non-AVX2 hardware.
4. Execute verification commands and capture outcomes.
5. Publish `implementation_notes.md`, `verification_report.md`, and update `layer_index.yaml` to advance to `v0.9.2`.

## Test Cases and Scenarios
- Unit/script behavior checks:
  - `bash scripts/benchmark_avx2_compare.sh --runs 1` executes successfully.
  - On hosts where both benchmark modes report `runtime_avx2_enabled: false`, summary output marks AVX2 delta as not applicable rather than a percentage delta.
- Triage reproducibility checks:
  - each bug-triage item includes an executable command and a pass/fail expectation.
- Non-regression command gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.1/plan.md` is present and decision-complete.
2. `docs/architecture/v1.0/v0.9/v0.9.1/bug_triage.md` exists with prioritized issues, deterministic reproductions, and fix-to-test mapping.
3. `scripts/benchmark_avx2_compare.sh` handles non-AVX2 environments without reporting misleading AVX2 delta percentages.
4. `docs/architecture/v1.0/v0.9/v0.9.1/implementation_notes.md` is present.
5. `docs/architecture/v1.0/v0.9/v0.9.1/verification_report.md` is present with criterion-to-evidence mapping.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
11. `docs/architecture/state/layer_index.yaml` marks `v0.9.1` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.2`.

## Risks and Mitigations
- Risk: triage findings are too shallow to guide `v0.9.2+` work.
  - Mitigation: require deterministic reproduction commands and explicit fix ownership/status in the triage artifact.
- Risk: benchmark script changes alter downstream parsing assumptions.
  - Mitigation: preserve existing fields and add clear compatibility notes in implementation/verification artifacts.
- Risk: full gate reruns consume time without providing new signal for docs/script-heavy changes.
  - Mitigation: still run full gates once for closeout consistency and capture results succinctly.

## Assumptions and Defaults
- `v0.9.1` is allowed to ship low-risk script/documentation fixes and defer larger optimizations to `v0.9.2+`.
- Current baseline evidence is `v0.8.3` benchmark artifacts and latest local reruns.
- Immediate next target after closeout is `docs/architecture/v1.0/v0.9/v0.9.2`.
