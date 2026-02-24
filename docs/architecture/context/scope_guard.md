# Scope Guard Decision Log

## Context
- Date: 2026-02-24
- Target layer: `docs/architecture/v1.0/v0.1/v0.0.7`
- Parent plan: `docs/architecture/v1.0/v0.1/plan.md`
- Decision basis:
  - `docs/architecture/v1.0/v0.1/v0.0.7/plan.md` in-scope/out-of-scope and acceptance criteria
  - current working diff (`git status --short --branch`)

## Extracted Scope
- In scope:
  - `engine` iterative run summary + stop reason surface
  - explicit artifact compatibility mode behavior in `engine`
  - focused tests for those behaviors
  - `v0.0.7` layer artifacts (`plan.md`, `implementation_notes.md`, `verification_report.md`)
- Out of scope:
  - multi-node depth expansion
  - predictor integration changes
  - non-CPU backend changes
  - unrelated docs/report edits outside `v0.0.7` unless explicitly requested

## Action Classification

### allowed
- `crates/engine/src/lib.rs`
  - Rationale: primary implementation surface for all `v0.0.7` deliverables and acceptance criteria.
- `docs/architecture/v1.0/v0.1/v0.0.7/plan.md`
  - Rationale: required planning artifact for this layer.
- `docs/architecture/v1.0/v0.1/v0.0.7/implementation_notes.md`
  - Rationale: required implementation artifact for this layer.
- `docs/architecture/v1.0/v0.1/v0.0.7/verification_report.md`
  - Rationale: required verification artifact for this layer.
- `docs/architecture/context/scope_guard.md`
  - Rationale: required artifact for this scope-guard decision pass.

### requires-explicit-approval
- `docs/architecture/state/layer_index.yaml`
  - Rationale: useful orchestration metadata update, but not a direct `v0.0.7` deliverable or acceptance criterion. Include only if user wants state index refreshed as part of this layer checkpoint.

### blocked-out-of-scope
- `README.md`
  - Rationale: not required by `v0.0.7` plan deliverables or acceptance criteria.
- `docs/architecture/v1.0/v0.1/v0.0.2/verification_report.md`
  - Rationale: prior-layer report edit unrelated to `v0.0.7` scope.
- `docs/architecture/v1.0/v0.1/v0.0.5/verification_report.md`
  - Rationale: prior-layer report edit unrelated to `v0.0.7` scope.
- `docs/architecture/v1.0/v0.1/v0.0.4/contract_drift_report.md`
  - Rationale: different layer artifact; not required for `v0.0.7` acceptance.
- Other untracked files under `docs/architecture/context/` (except this file)
  - Rationale: not needed to satisfy `v0.0.7` plan acceptance criteria.

## Enforcement Summary
- `v0.0.7` implementation/review should stay limited to:
  - `crates/engine/src/lib.rs`
  - `docs/architecture/v1.0/v0.1/v0.0.7/*`
  - optionally `docs/architecture/state/layer_index.yaml` only with explicit user approval for state refresh in this pass.
