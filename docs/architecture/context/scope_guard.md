# Scope Guard Decision Log

## Context
- Date: 2026-02-26
- Target layer: `docs/architecture/v1.0/v0.2/v0.1.8`
- Parent plan: `docs/architecture/v1.0/v0.2/plan.md`
- Target plan status: missing (`docs/architecture/v1.0/v0.2/v0.1.8/plan.md` does not exist).
- Decision basis:
  - active target from `docs/architecture/state/layer_index.yaml`
  - parent constraints from `docs/architecture/v1.0/v0.2/plan.md`
  - prior-layer residual risk from `docs/architecture/v1.0/v0.2/v0.1.7/verification_report.md`
  - current working diff (`git status --short --branch`)

## Extracted Scope
- In scope (for pre-plan guard pass):
  - create `docs/architecture/v1.0/v0.2/v0.1.8/plan.md` before implementation work.
  - keep `v0.1.8` intent aligned to remaining `v0.2` gap: Python-runtime execution coverage for the extension inference entry point.
  - update this scope guard record.
- Out of scope (until explicitly planned/approved):
  - `v0.1.8` code changes unrelated to closing the Python-runtime test-evidence gap.
  - algorithm/engine behavior changes not required for Python-runtime test execution.
  - parent rollup closure artifacts (`v0.2`/`v1.0`) before finishing remaining child-layer scope.
  - roadmap-deferred domains: ranking, SHAP implementation, categorical execution pipeline, SIMD/perf work, CUDA/Metal/MLX.

## Acceptance Criteria Baseline
- Since `v0.1.8/plan.md` is missing, acceptance criteria are provisional and must be finalized in that plan.
- Provisional criteria boundary:
  - add explicit acceptance checks for Python-runtime import/execution of extension-backed inference path.
  - preserve passing workspace verification gates already required by `v0.2`.
  - avoid introducing new training or sklearn-surface features (belongs to `0.3.0+`).

## Action Classification

### allowed
- `docs/architecture/v1.0/v0.2/v0.1.8/plan.md`
  - Rationale: required first artifact for target layer execution.
- `docs/architecture/context/scope_guard.md`
  - Rationale: required artifact for this scope-guard decision pass.

### requires-explicit-approval
- `bindings/python/Cargo.toml`
  - Rationale: currently part of pending `v0.1.7` closure diff; carrying forward into `v0.1.8` should be an explicit decision.
- `bindings/python/src/lib.rs`
  - Rationale: currently contains uncommitted `v0.1.7` parity-evidence work; any further edits should follow explicit decision to land/isolate prior-layer changes first.
- `docs/architecture/v1.0/v0.2/v0.1.7/verification_report.md`
  - Rationale: prior-layer artifact; update/commit is valid for closure but not required to define `v0.1.8` scope.
- `docs/architecture/state/layer_index.yaml`
  - Rationale: should be updated after `v0.1.8` verification, not during pre-plan gating.
- `docs/architecture/context/handoff.md`
  - Rationale: continuity doc is useful, but not required to establish `v0.1.8` boundaries.
- `docs/architecture/context/session_brief.md`
  - Rationale: continuity doc is useful, but not required to establish `v0.1.8` boundaries.

### blocked-out-of-scope
- `crates/engine/src/lib.rs`
  - Rationale: engine algorithm changes are not required to close `v0.1.8` Python-runtime extension execution gap.
- `docs/architecture/v1.0/v0.2/v0.1.2/verification_report.md`
  - Rationale: prior-layer report edit unrelated to `v0.1.8` target scope.
- `docs/architecture/v1.0/v0.2/v0.1.5/verification_report.md`
  - Rationale: prior-layer report edit unrelated to `v0.1.8` target scope.
- Any new public Python training API surface in `bindings/python/alloygbm/*`
  - Rationale: parent plan flags sklearn-surface expansion as out-of-scope for `v0.2`.

## Enforcement Summary
- Do not start `v0.1.8` implementation until `docs/architecture/v1.0/v0.2/v0.1.8/plan.md` is created.
- Keep `v0.1.8` scoped to Python-runtime extension execution evidence and required verification.
- Treat non-target dirty files as deferred unless explicitly approved for prior-layer closure.
