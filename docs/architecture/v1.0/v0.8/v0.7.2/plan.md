# AlloyGBM v0.7.2 Plan (v0.8 Exact TreeSHAP Traversal Math Slice)

## Summary
- Goal: execute `v0.7.2` by replacing the `v0.7.1` additivity harness contribution assignment with exact Shapley contribution math over the current tree payload semantics.
- Success criteria:
  - per-row SHAP values are computed via exact Shapley formulation over conditional expectations,
  - expected value is computed from model expectation under branch probabilities derived from node cover counts,
  - additivity holds against model predictions for deterministic fixtures.
- Audience: engineers advancing `v0.8` SHAP internals and reviewers gating readiness for next integration/hardening slice.

## Scope
### In Scope
- `crates/shap` exact-math upgrade:
  - compute `v(S)` expectations by traversing trees with known-feature constraints and unknown-feature branch weighting,
  - compute per-feature SHAP values from exact Shapley summation over feature subsets,
  - keep existing artifact-backed SHAP public API contract from `v0.7.1`.
- Deterministic traversal assumptions:
  - use current predictor split rule (`<= threshold_bin` goes left),
  - use `left_stats.row_count` / `right_stats.row_count` for unknown-feature branch probabilities.
- Test upgrades in `crates/shap`:
  - lock expected-value correctness and contribution sign/magnitude for known fixtures,
  - lock behavior for unused features and compatibility/error paths.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.8/v0.7.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md`

### Out of Scope
- Python bridge additions (`v0.7.4` scope).
- New artifact sections or model format changes.
- GPU/Metal SHAP.
- Interaction SHAP and approximation modes.
- Performance optimization beyond correctness-oriented exact computation.

## Interfaces and Types
- `crates/shap/src/lib.rs`:
  - preserve `explain_rows_from_artifact_bytes(...)` and `global_importance_*` public entrypoints,
  - change internal contribution math from path-assignment baseline to exact Shapley using conditional expectations,
  - preserve deterministic `ShapError::{InvalidInput, ContractViolation}` semantics.
- `crates/shap/Cargo.toml`:
  - keep dependency scope minimal and aligned with existing artifact/model contracts.

Backward-compatibility expectations:
- API signatures remain unchanged.
- Existing non-SHAP engine/predictor behavior remains unchanged.
- Artifact strict/legacy compatibility checks remain unchanged.

## Deliverables
1. Exact expectation package:
  - `v(S)` evaluator based on tree traversal with known/unknown feature handling.
2. Exact Shapley package:
  - subset-weighted SHAP contribution computation over split features.
3. Determinism/safety package:
  - bounded exact subset computation guardrail and explicit error semantics.
4. Test package:
  - updated fixture expectations for expected value and per-feature contributions.
5. State package:
  - layer index update marking `v0.7.2` verified and advancing next target.

## Implementation Sequence
1. Add `v0.7.2` plan and lock exact-math scope.
2. Implement exact `v(S)` traversal and Shapley summation in `crates/shap` internals.
3. Keep global-importance and API contracts unchanged while switching to exact contribution values.
4. Update/add tests for:
  - exact expected value,
  - exact per-row contribution values on deterministic fixtures,
  - additivity and unused-feature zero contribution,
  - compatibility/error regressions.
5. Run required verification commands and resolve issues.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `docs/architecture/state/layer_index.yaml` to mark `v0.7.2` verified and set `v0.7.3` next.

## Test Cases and Scenarios
- Unit cases:
  - expected-value calculation under unknown-feature branch weighting,
  - exact contribution checks for fixture rows,
  - unused feature contributes zero.
- Integration cases:
  - artifact-backed explanation path remains functional and deterministic,
  - additivity holds (`expected_value + sum(phi_i)` equals prediction within tolerance).
- Failure and edge cases:
  - malformed required-section artifacts,
  - non-finite row inputs,
  - split-feature cardinality guardrail violations.
- Acceptance test mapping:
  - `cargo test -p alloygbm-shap`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.7.2/plan.md` exists and is decision-complete.
2. `crates/shap` computes expected value via traversal expectation, not direct baseline passthrough.
3. `crates/shap` computes per-feature SHAP using exact Shapley subset weighting over split features.
4. Fixture tests verify exact expected value and contribution values for representative rows.
5. Additivity check remains enforced and passing for fixture rows.
6. Global importance remains deterministic and derived from mean absolute SHAP contributions.
7. `docs/architecture/v1.0/v0.8/v0.7.2/implementation_notes.md` is created.
8. `docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md` is created.
9. `cargo fmt -- --check` passes.
10. `cargo clippy --workspace --all-targets -- -D warnings` passes.
11. `cargo test --workspace` passes.
12. Python unittest suite passes.

## Risks and Mitigations
- Risk: exact subset math can be expensive with many split features.
  - Mitigation: enforce an explicit split-feature guardrail and deterministic error.
- Risk: branch probability semantics diverge from future TreeSHAP choices.
  - Mitigation: lock current semantics to node cover counts and document for future revisions.
- Risk: regression in `v0.7.1` compatibility/error behavior.
  - Mitigation: retain and run prior validation/compatibility tests unchanged.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Unknown-feature branch probability defaults to `0.5` when node cover counts sum to zero.
- Exact subset computation default guardrail: maximum 20 distinct split features per model.
