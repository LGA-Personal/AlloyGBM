# v0.1.3 Implementation Notes

## Summary of What Was Built
- Implemented `v0.1.3` as a CPU-backend integration evidence slice for `v0.2`.
- Added end-to-end training tests in `crates/backend_cpu/src/lib.rs` that run `Trainer + CpuBackend + SquaredErrorObjective` on a deterministic fixture:
  - `cpu_backend_training_beats_naive_baseline_mse`
  - `cpu_backend_deterministic_training_has_stable_artifact_bytes`
- Added test-only fixture utilities to support quality and reproducibility assertions:
  - deterministic dataset/binned-matrix builders,
  - fixture row conversion helper,
  - mean-squared-error helper,
  - deterministic training-parameter helper.

## Non-Intuitive Decisions
- Decision: place end-to-end quality/reproducibility tests in `backend_cpu` crate tests rather than engine tests.
- Reason: engine tests are mock-backend focused; real backend execution evidence belongs where concrete backend behavior is exercised.
- Impact: the layer now has explicit coverage of real histogram/split/reduction behavior in conjunction with trainer logic.

- Decision: use explicit artifact-byte equality for deterministic reproducibility.
- Reason: this is a strict and auditable determinism check aligned with phase-level reproducibility goals.
- Impact: any non-deterministic regression in training/model serialization will fail fast.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.2/v0.1.3/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No public API or crate-boundary contract changes.
- Scope stayed in backend test coverage and layer artifacts.

## Known Gaps Deferred to Next Layer
- Training remains stump-level iterative boosting; full depth-limited CART tree behavior is still pending for later `v0.2` child slices.
- Parent rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/verification_report.md`

## Follow-Up Actions
- Define and implement `v0.1.4` focused on remaining `v0.2` algorithm depth/behavior gaps while preserving deterministic and validation-stop contracts.
