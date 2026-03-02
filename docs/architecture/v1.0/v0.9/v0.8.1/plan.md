# AlloyGBM v0.8.1 Plan (v0.9 Hardening Matrix Baseline Slice)

## Summary
- Goal: execute `v0.8.1` as the first `v0.9` child slice by locking a decision-complete release hardening matrix, baseline evidence inventory, and verification protocol for subsequent `v0.8.x` slices.
- Success criteria:
  - a concrete hardening matrix artifact exists and maps release gates to evidence sources,
  - baseline compatibility and behavior inventory from `v0.8` closeout is captured with traceable links,
  - `v0.8.1` artifacts are complete and the next child target advances to `v0.8.2`.
- Audience: engineers implementing `v0.9` hardening slices and reviewers gating `1.0.0` readiness evidence quality.

## Scope
### In Scope
- Author `v0.8.1` layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.8.1/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.8.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.8.1/verification_report.md`
- Create hardening baseline artifact:
  - `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md`
- Hardening matrix content requirements:
  - release gate inventory (Rust, Python, docs/compat checks),
  - baseline evidence sources from `v0.8` child/parent verification artifacts,
  - open hardening work buckets assigned to future slices (`v0.8.2+`).
- State progression update:
  - mark `v0.8.1` verified in `docs/architecture/state/layer_index.yaml`,
  - create/advance next target to `docs/architecture/v1.0/v0.9/v0.8.2`.

### Out of Scope
- New algorithmic features or roadmap expansion.
- Model-format version changes.
- Python/Rust public API redesigns.
- Parent `v0.9` rollup artifact completion (`implementation_notes.md`, `verification_report.md` at parent level).

## Interfaces and Types
- Documentation/state interfaces only for this slice:
  - `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md`
  - `docs/architecture/state/layer_index.yaml`
- Existing code interfaces remain unchanged:
  - SHAP APIs and behavior baseline from `v0.8`,
  - predictor artifact compatibility policy and Python estimator contract.

Backward-compatibility expectations:
- no production source changes are required for `v0.8.1`,
- all behavior commitments from `v0.8` remain the reference baseline for later hardening slices.

## Implementation Sequence
1. Build a `v0.9` hardening baseline from ancestor and `v0.8` closeout artifacts.
2. Author `release_hardening_matrix.md` with gate inventory, evidence links, and unresolved hardening buckets for `v0.8.2+`.
3. Record implementation decisions and boundary confirmations in `implementation_notes.md`.
4. Run verification gates and capture results for this slice.
5. Author `verification_report.md` and update `layer_index.yaml` to mark `v0.8.1` verified and advance to `v0.8.2`.

## Test Cases and Scenarios
- Documentation/traceability checks:
  - hardening matrix exists and includes release gates + baseline evidence links,
  - matrix lists unresolved hardening buckets tied to future child slices.
- State transition checks:
  - `layer_index.yaml` reflects `v0.8.1` as verified and sets next target to `v0.8.2`.
- Verification command gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.8.1/plan.md` is present and decision-complete for the hardening-matrix slice.
2. `docs/architecture/v1.0/v0.9/v0.8.1/release_hardening_matrix.md` exists and maps release gates to concrete evidence sources.
3. Matrix includes explicit baseline non-regression commitments carried from `v0.8`.
4. `docs/architecture/v1.0/v0.9/v0.8.1/implementation_notes.md` is present and scoped to this slice.
5. `docs/architecture/v1.0/v0.9/v0.8.1/verification_report.md` is present with criterion-to-evidence status.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
11. `docs/architecture/state/layer_index.yaml` marks `v0.8.1` verified and advances next target to `docs/architecture/v1.0/v0.9/v0.8.2`.

## Risks and Mitigations
- Risk: matrix is descriptive but not operationally useful for later slices.
  - Mitigation: include explicit gate commands, baseline sources, and unresolved bucket ownership for `v0.8.2+`.
- Risk: documentation-only slice misses latent regressions.
  - Mitigation: rerun full verification command gates and record outcomes.
- Risk: state index drifts from produced artifacts.
  - Mitigation: update status fields only after all required artifacts and verification evidence are complete.

## Assumptions and Defaults
- `v0.8.1` is a hardening-baseline slice and should avoid production code edits unless verification exposes regressions.
- Next child target after this slice is `docs/architecture/v1.0/v0.9/v0.8.2`.
- Full verification command gates remain the default evidence standard for `v0.9` children.
