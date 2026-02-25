# Scope Guard Decision Log

## Context
- Date: 2026-02-25
- Target layer: `docs/architecture/v1.0/v0.1/v0.0.10`
- Parent plan: `docs/architecture/v1.0/v0.1/plan.md`
- Target plan status: missing (`docs/architecture/v1.0/v0.1/v0.0.10/plan.md` does not exist).
- Decision basis:
  - parent constraints from `docs/architecture/v1.0/v0.1/plan.md`
  - active target from `docs/architecture/state/layer_index.yaml`
  - current working diff (`git status --short --branch`)

## Extracted Scope
- In scope:
  - create `docs/architecture/v1.0/v0.1/v0.0.10/plan.md` before any implementation changes.
  - update this scope guard record.
- Out of scope:
  - any `v0.0.10` code implementation before a target plan exists.
  - edits to prior-layer reports or unrelated docs not required for target planning.
  - cross-cutting changes outside parent `v0.1` boundaries (full trainer depth, SIMD/perf, ranking, SHAP/categorical full implementations).

## Action Classification

### allowed
- `docs/architecture/v1.0/v0.1/v0.0.10/plan.md`
  - Rationale: required first artifact; target plan is currently missing.
- `docs/architecture/context/scope_guard.md`
  - Rationale: required artifact for this scope-guard decision pass.

### requires-explicit-approval
- `crates/engine/src/lib.rs`
  - Rationale: current diff appears to be `v0.0.9` implementation carryover; modifying or committing it while target is `v0.0.10` requires explicit intent to finish/prioritize prior-layer closure first.
- `docs/architecture/state/layer_index.yaml`
  - Rationale: orchestration metadata update is valid for layer transitions, but not required for pre-plan scope gating of `v0.0.10`.
- `docs/architecture/v1.0/v0.1/v0.0.9/*`
  - Rationale: these are prior-layer artifacts; carrying them forward should be an explicit decision to close `v0.0.9` bookkeeping before starting `v0.0.10`.

### blocked-out-of-scope
- `README.md`
  - Rationale: not required by target planning for `v0.0.10`.
- `docs/architecture/v1.0/v0.1/v0.0.2/verification_report.md`
  - Rationale: prior-layer report edit unrelated to target planning.
- `docs/architecture/v1.0/v0.1/v0.0.5/verification_report.md`
  - Rationale: prior-layer report edit unrelated to target planning.
- `docs/architecture/v1.0/v0.1/v0.0.4/contract_drift_report.md`
  - Rationale: different layer artifact; not required for `v0.0.10` plan creation.
- `docs/architecture/context/handoff.md`
  - Rationale: useful continuity artifact, but not required for `v0.0.10` scope definition in this pass.
- `docs/architecture/context/session_brief.md`
  - Rationale: useful context artifact, but not required to satisfy target-plan prerequisite.

## Enforcement Summary
- Do not start `v0.0.10` implementation until `docs/architecture/v1.0/v0.1/v0.0.10/plan.md` exists.
- Keep this pass limited to scope gating + target plan creation.
- Treat current non-target dirty files as deferred unless explicitly approved for prior-layer closure.
