# Title
`AlloyGBM v0.1 Technical Plan`

## Summary
- Goal: Deliver the `0.1.0` milestone as a minimal histogram GBDT CPU regression implementation that transitions the project from contracts-first scaffolding into usable training/inference behavior.
- Success criteria:
  - depth-limited histogram CART growth for regression,
  - training loop with shrinkage, row/column subsampling, and early stopping on validation,
  - row and batch prediction path from trained model artifacts,
  - verification evidence aligned to `v1.0` phase gates.
- Audience: engineers implementing `v0.1` child layers (starting with `v0.1.1`) and reviewers gating progression to `v0.2`.

## Scope
### In Scope
- CPU-only minimal GBDT for regression objective (L2/squared error).
- Histogram-based split search and depth-limited tree growth with minimum leaf-size controls.
- Boosting loop controls:
  - learning-rate shrinkage,
  - row subsampling,
  - column subsampling,
  - validation-set early stopping policy.
- Trained-model prediction behavior (single-row and batch) from serialized artifacts.
- Model artifact contract evolution needed for tree payload completeness while preserving explicit versioned schema discipline.
- Verification artifacts and tests that prove correctness-oriented `0.1.0` behavior.

### Out of Scope
- Ranking objectives/metrics.
- SHAP algorithm implementation.
- Full categorical transform execution pipeline.
- SIMD/performance optimization work targeted for later milestones (`0.5+`).
- CUDA/Metal/MLX backends.
- Full sklearn-surface expansion planned for `0.2.0`.

## Interfaces and Types
- `core`:
  - extend/confirm training configuration contracts for `0.1.0` controls (depth/leaf, subsampling, early stopping inputs),
  - keep versioned model artifact + metadata contracts explicit and testable.
- `engine`:
  - trainer interfaces evolve from scaffolding to executable minimal histogram-GBDT loop,
  - expose deterministic row/batch prediction from trained model structure.
- `backend_cpu`:
  - implement/complete CPU primitives required by tree construction and iterative training loop.
- `predictor`:
  - align predictor-facing contracts with trained tree payload needed for row/batch inference.
- `bindings/python`:
  - keep Python surface stable and compatible with existing baseline contract while core Rust behavior matures.

Backward-compatibility expectations:
- preserve existing contract semantics unless explicitly versioned/migrated in artifacts,
- document any compatibility policy changes in layer artifacts before behavior flips.

## Implementation Sequence
1. Establish `v0.1` decomposition baseline and create first child layer plan at `docs/architecture/v1.0/v0.1/v0.1.1/plan.md` (future step; not part of this planning pass).
2. Lock `core` and `engine` contracts for minimal histogram training controls and validation-set early stopping interfaces.
3. Implement depth-limited histogram tree-growth loop in `engine` + `backend_cpu` with deterministic behavior under fixed seed.
4. Wire boosting controls (shrinkage, row/column subsampling) and stop-policy reporting.
5. Finalize artifact/predictor path for row+batch inference parity from trained model bytes.
6. Run verification and document evidence in layer artifacts (`implementation_notes.md`, `verification_report.md`, drift checks as needed).

## Test Cases and Scenarios
- Unit cases:
  - config validation for depth/leaf/subsampling/early-stopping bounds,
  - histogram and split correctness on deterministic toy fixtures,
  - deterministic repeatability under fixed seed.
- Integration cases:
  - end-to-end train -> serialize -> deserialize -> predict parity,
  - validation-set early stopping triggers and stop-reason traceability,
  - row and batch prediction agreement.
- Failure and edge cases:
  - empty/invalid datasets and shape mismatches,
  - degenerate split candidates and no-split rounds,
  - malformed/legacy artifact contract handling per explicit compatibility mode.
- Acceptance test mapping:
  - roadmap `0.1.0` deliverables map to engine/backend tests plus workspace command gates (`cargo fmt`, `cargo clippy`, `cargo test`, `cargo doc`),
  - baseline regression quality check must beat naive constant predictor on selected fixture(s),
  - verification evidence captured in layer verification reports.

## Risks and Mitigations
- Risk: scope creep into `0.2.0` Python UX work.
  - Mitigation: keep Python changes contract-stability only for `v0.1`.
- Risk: unstable interfaces during first real trainer implementation.
  - Mitigation: land work through small child layers (`v0.1.1+`) with contract drift checks each slice.
- Risk: artifact compatibility regressions while expanding tree payload.
  - Mitigation: maintain explicit versioned contract tests and dual-path compatibility checks.
- Risk: performance pressure distorts correctness-first goals.
  - Mitigation: keep `0.1` acceptance correctness-focused; defer optimization objectives to `0.4`.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Objective scope for `v0.1` is regression only.
- `v0.1` execution will continue the nested workflow using child layers starting at `v0.1.1` (as requested), but that child plan is intentionally deferred.
- Existing CI quality gates remain required for every child layer.
- If compatibility behavior must change, default is "document first, then switch" rather than silent behavior flips.
