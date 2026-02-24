# AlloyGBM v0.0.8 Plan (v0.1 Week 8 Depth Budget Policy + Artifact Compatibility Report)

## Objective
Progress iterative tree-policy behavior by honoring an explicit depth budget in the stump-training loop, and harden artifact compatibility policy with a reportable shape-analysis API plus deterministic auto-mode import.

## Scope
- In scope:
  - Use `TrainParams.max_depth` as an effective round/depth budget in iterative training APIs.
  - Extend run summary observability to expose the effective round cap and stop reason when depth budget is reached.
  - Add artifact compatibility reporting in `engine` that classifies strict-compatible vs legacy-compatible section layouts.
  - Add an auto-mode artifact import entrypoint that chooses strict or legacy mode from the compatibility report.
  - Add focused tests for depth-budget stopping and compatibility report/auto-mode behavior.
- Out of scope:
  - Multi-node tree growth beyond current stump/root-partition approach.
  - Predictor crate integration changes.
  - Non-CPU backend behavior changes.
  - SHAP/categorical payload implementation.

## Deliverables
1. Iteration depth-budget package:
  - iterative training loop respects effective cap: `min(controls.rounds, TrainParams.max_depth)`.
  - new stop reason for depth-budget termination.
  - run summary exposes effective round cap for traceability.
2. Artifact compatibility policy package:
  - compatibility report struct capturing required-section counts and strict/legacy compatibility booleans.
  - `TrainedModel::artifact_compatibility_report(...)` helper.
  - `TrainedModel::from_artifact_bytes_auto(...)` that chooses mode from report and returns chosen mode with model.
3. Verification package:
  - focused tests for depth-budget behavior and compatibility report/auto import.

## Implementation Plan
1. Add `v0.0.8` plan artifact.
2. Extend iteration stop/summarization types and loop behavior in `crates/engine/src/lib.rs` to enforce depth budget.
3. Add compatibility report types/helpers and auto import API in `crates/engine/src/lib.rs`.
4. Add/adjust engine tests covering:
  - depth-budget stop reason and effective-round-cap reporting
  - strict and legacy report classification
  - auto import mode selection for dual-section and legacy artifacts
  - rejection for malformed required-section shapes.
5. Run verification commands and capture evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify iteration summary reports depth-budget stop when requested rounds exceed `TrainParams.max_depth`.
5. Engine tests verify run summary reports `effective_round_cap` consistently for both capped and uncapped runs.
6. Engine tests verify compatibility report classification:
  - strict-compatible dual-section payload (`Trees` + `PredictorLayout`)
  - legacy-compatible trees-only payload
  - malformed required-section duplicates are non-compatible.
7. Engine tests verify `from_artifact_bytes_auto(...)`:
  - selects strict mode for dual-section artifacts
  - selects legacy-compatible mode for trees-only legacy artifacts
  - rejects malformed required-section layouts.
8. Existing prediction-consistency artifact roundtrip test remains passing.

## Risks and Mitigations
- Risk: introducing depth budget behavior changes existing iterative expectations.
  - Mitigation: keep defaults stable (`max_depth=6`) and add targeted tests for capped vs uncapped behavior.
- Risk: compatibility report logic diverges from actual importer rules.
  - Mitigation: derive auto-mode behavior directly from report output and assert with focused tests.
- Risk: auto-mode API could hide import-policy intent.
  - Mitigation: return selected compatibility mode alongside the model to preserve explicit observability.

## Exit Condition
`v0.0.8` is complete when depth-budget and compatibility-report behaviors are test-backed, verification commands pass, and layer implementation/verification artifacts are recorded.
