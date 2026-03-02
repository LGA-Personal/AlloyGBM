# AlloyGBM v0.7.1 Plan (v0.8 SHAP Contract and Additivity Harness Slice)

## Summary
- Goal: execute the first `v0.8` child slice by establishing artifact-backed SHAP interfaces, deterministic validation behavior, and additivity test harnesses before exact TreeSHAP path weighting work.
- Success criteria:
  - SHAP APIs consume model artifacts directly and validate compatibility/input shape deterministically,
  - SHAP outputs have stable `rows x feature_count` shape with explicit expected-value semantics,
  - additivity checks are locked with fixture tests so `expected_value + sum(phi_i)` matches model predictions within tolerance.
- Audience: engineers implementing `v0.8` TreeSHAP work and reviewers gating readiness for `v0.7.2` exact traversal math.

## Scope
### In Scope
- `crates/shap` contract baseline:
  - replace placeholder-only behavior with artifact-backed explanation entrypoints,
  - add deterministic error handling for invalid rows, feature mismatches, and incompatible artifacts,
  - define batch explanation payload including expected value and per-feature contributions.
- Additivity harness:
  - enforce additivity checks in tests using deterministic fixture models and rows,
  - add global-importance aggregation from SHAP contributions (mean absolute contribution) with deterministic ordering.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.8/v0.7.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/v0.7.1/verification_report.md`

### Out of Scope
- Exact TreeSHAP path-probability weighting algorithm (`v0.7.2` scope).
- Python SHAP bridge and `GBMRegressor` SHAP methods (`v0.7.4` scope).
- New artifact sections or model format version changes.
- GPU/Metal SHAP execution.
- Interaction SHAP values and approximate SHAP modes.

## Interfaces and Types
- `crates/shap/src/lib.rs`:
  - introduce artifact-backed SHAP batch explanation API,
  - introduce global-importance API over SHAP outputs,
  - add deterministic `ShapError` contract violations for artifact and input errors.
- `crates/shap/Cargo.toml`:
  - include any required dependencies to decode/evaluate current model artifacts without changing upstream contracts.
- `crates/core` and `crates/engine` are consumed as contract anchors; this slice must not modify their public behavior.

Backward-compatibility expectations:
- No changes to training/prediction behavior in `engine` or `predictor`.
- SHAP APIs remain additive and non-breaking to existing non-SHAP paths.
- Artifact strict/legacy compatibility semantics remain unchanged.

## Deliverables
1. API package:
  - artifact-backed SHAP batch explanation API with expected value + contribution matrix.
2. Validation package:
  - deterministic row and artifact compatibility validation with explicit errors.
3. Aggregation package:
  - global feature importance computed as mean absolute SHAP contribution.
4. Test package:
  - unit/integration tests for shape, additivity, and validation behavior.
5. State package:
  - `layer_index.yaml` updated for `v0.7.1` verified completion and `v0.7.2` next-target suggestion.

## Implementation Sequence
1. Add `v0.7.1` plan and lock scope boundaries against parent `v0.8` plan.
2. Implement artifact-backed SHAP interfaces and deterministic validation in `crates/shap`.
3. Implement global-importance aggregation over SHAP contributions.
4. Add fixture-driven tests covering:
  - malformed/unsupported artifact and input-shape failures,
  - contribution output shape,
  - additivity identity against model predictions,
  - global-importance deterministic ordering.
5. Run verification commands and resolve failures.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `docs/architecture/state/layer_index.yaml` to mark `v0.7.1` verified and set `v0.7.2` next.

## Test Cases and Scenarios
- Unit cases:
  - rows-empty and feature-count mismatch validation,
  - deterministic contribution shape for multi-row input,
  - global-importance mean-absolute aggregation and stable ordering.
- Integration cases:
  - artifact -> SHAP explanation flow for deterministic fixture models,
  - additivity check against model prediction for each row.
- Failure and edge cases:
  - incompatible artifact required-section layouts,
  - malformed row values (non-finite) and inconsistent row widths.
- Acceptance test mapping:
  - `cargo test -p alloygbm-shap`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.7.1/plan.md` is present and decision-complete for this slice.
2. `crates/shap` provides artifact-backed SHAP batch explanation output with expected value + contribution matrix.
3. SHAP API validates rows and artifact compatibility deterministically with explicit errors.
4. SHAP output row width equals model feature count for all rows.
5. Fixture tests prove additivity (`expected_value + sum(phi_i)` matches model prediction within tolerance).
6. Global importance is computed from mean absolute SHAP contribution and returned in deterministic ordering.
7. `docs/architecture/v1.0/v0.8/v0.7.1/implementation_notes.md` is created.
8. `docs/architecture/v1.0/v0.8/v0.7.1/verification_report.md` is created.
9. `cargo fmt -- --check` passes.
10. `cargo clippy --workspace --all-targets -- -D warnings` passes.
11. `cargo test --workspace` passes.
12. Python unittest suite passes.

## Risks and Mitigations
- Risk: contract API introduced in `v0.7.1` conflicts with planned exact TreeSHAP internals.
  - Mitigation: keep API focused on artifact I/O, validation, expected value, and contribution matrix shape.
- Risk: additivity tests could encode brittle assumptions.
  - Mitigation: assert numerical tolerance and model-prediction equality instead of hardcoding internal traversal counts.
- Risk: scope creep into Python bridge before Rust contract stabilizes.
  - Mitigation: defer Python SHAP surface to `v0.7.4` explicitly.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Additivity tolerance default: `1e-5` absolute error.
- Contribution semantics in this slice prioritize deterministic contract and additivity harness; exact TreeSHAP path weighting is deferred to `v0.7.2`.
