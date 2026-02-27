# AlloyGBM v0.3.1 Plan (v0.4 Evaluation Metrics Slice)

## Objective
Execute the first `v0.4` child slice by introducing baseline evaluation metrics for regression and finance workflows in the Python wrapper path, with deterministic tests and explicit error semantics.

## Scope
- In scope:
  - Add first-class evaluation metric helpers for wrapper prediction outputs (baseline set: RMSE, MAE, R2, correlation).
  - Provide deterministic behavior for finite numeric inputs with explicit shape/length validation.
  - Add focused tests covering metric correctness and failure semantics.
  - Produce `v0.3.1` implementation and verification artifacts.
- Out of scope:
  - Purged K-fold and embargo split tooling (next child slices).
  - Ranking objectives/metrics integration.
  - SHAP/categorical/backend/performance expansion.

## Deliverables
1. Evaluation API package:
  - Python-visible metric helpers wired for wrapper workflows.
2. Test evidence package:
  - contract tests validating metric outputs and invalid-input errors.
3. Layer documentation package:
  - `docs/architecture/v1.0/v0.4/v0.3.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.3.1/verification_report.md`
4. State package:
  - `docs/architecture/state/layer_index.yaml` updated after verification.

## Implementation Sequence
1. Add metric helper implementations and validation guards.
2. Add deterministic fixture tests for each metric and error path.
3. Run verification command gates.
4. Publish implementation/verification artifacts and refresh layer index.

## Acceptance Criteria
1. Metric helpers return deterministic outputs for baseline fixtures (RMSE, MAE, R2, correlation).
2. Metric helpers reject malformed/mismatched input lengths with explicit errors.
3. Existing `v0.3` wrapper contracts remain passing.
4. `cargo fmt -- --check` passes.
5. `cargo clippy --workspace --all-targets -- -D warnings` passes.
6. `cargo test --workspace` passes.
7. `cargo doc --workspace --no-deps` passes.
8. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: metric definitions become ambiguous across components.
  - Mitigation: encode fixture assertions with clear expected values and tolerances.
- Risk: evaluation helpers leak into broader strategy/ranking scope.
  - Mitigation: keep this slice to baseline regression/finance metric calculations only.

## Exit Condition
`v0.3.1` is complete when baseline metrics and tests are implemented with all gates passing and required artifacts recorded.
