# AlloyGBM v0.7 Implementation Notes

## Summary of What Was Built
- Closed `v0.7` as a four-slice delivery (`v0.6.1` through `v0.6.4`) for categorical support v1.
- Delivered contract, runtime, integration, and Python bridge scope in order:
  - `v0.6.1`: categorical artifact contract + schema invariant enforcement in `core`.
  - `v0.6.2`: deterministic target/frequency encoders with leakage-safe time-aware behavior in `categorical`.
  - `v0.6.3`: engine/predictor integration for optional categorical state artifact flow.
  - `v0.6.4`: additive Python/native bridge exposure for categorical fit controls and end-to-end parity tests.
- Preserved parent constraints:
  - CPU-only scope,
  - no model-format version bump,
  - additive (non-breaking) API evolution for numeric-only call sites.

## Parent-Level Outcome by Contract Area
- Categorical encoding functionality:
  - Production-ready target/frequency/count encoders with deterministic behavior and explicit unknown-category fallbacks are in place.
- Artifact contract continuity:
  - `ModelSectionKind::CategoricalState` remains optional/additive and validated against feature bounds.
  - Strict/legacy required-section behavior remains unchanged.
- Engine/predictor integration:
  - Engine can train with single-feature categorical target encoding wrapper and emit categorical state.
  - Predictor accepts artifacts carrying optional categorical state and preserves prediction parity.
- Python surface:
  - `GBMRegressor` and native binding now expose additive categorical/time-index controls.
  - Numeric-only behavior remains unchanged and validated.

## Non-Intuitive Decisions
- Decision: maintain single-feature categorical orchestration for this milestone.
- Reason: aligns with scoped `v0.7` integration target and avoids unplanned multi-feature orchestration expansion.
- Impact: v1 categorical support is available end-to-end for current wrapper path, while full multi-feature orchestration remains a follow-on item.

- Decision: enforce strict categorical argument consistency in both Python and Rust bridge boundaries.
- Reason: prevent silent fallback behavior and keep failure semantics deterministic.
- Impact: callers get explicit input-contract errors when categorical mode is partially configured.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Planned boundaries were preserved:
  - `core` contract additions landed in `v0.6.1`.
  - `categorical` runtime landed in `v0.6.2`.
  - `engine` + `predictor` artifact path integration landed in `v0.6.3`.
  - Python/native bridge additions landed in `v0.6.4`.
- No ranking/SHAP/GPU changes were introduced in `v0.7`.

## Residual Risks
- Multi-feature categorical orchestration is not implemented yet.
- Predictor/runtime categorical transform replay beyond optional state metadata remains deferred.

## Follow-Up Actions
- Mark `docs/architecture/v1.0/v0.7` verified in `docs/architecture/state/layer_index.yaml` and advance next active target to `v0.8` planning.
- Open `docs/architecture/v1.0/v0.8/plan.md` for TreeSHAP CPU scope.
