# AlloyGBM v0.0.5 Plan (v0.0 Week 5 Training Controls + Artifact Coverage + Naming Drift Fix)

## Objective
Strengthen the `v0.0.4` iterative slice with explicit loop-stop controls and broader artifact section coverage, while landing a separate documentation hygiene fix for the previously reported Python naming/path drift.

## Scope
- In scope:
  - Add explicit iterative training controls in `engine` to keep stump-level training bounded and deterministic:
    - minimum split gain threshold
    - minimum rows-per-leaf threshold
  - Extend engine model artifact export/import to include and validate a predictor-layout section in addition to trees payload.
  - Add stricter artifact import guards in `engine` for required section presence and duplicate section rejection.
  - Add focused tests covering new controls and section-validation behavior.
  - Apply the contract drift naming/path fix as a separate part:
    - update stale `v0.0.3` Python docstrings to layer-neutral or current wording.
- Out of scope:
  - Full depth-controlled multi-node tree growth policy.
  - Early stopping by validation metric, regularization tuning, or production pruning policy.
  - Predictor crate production integration.
  - SHAP/categorical section payload implementation beyond section-kind compatibility.

## Deliverables
1. Engine training-controls package:
   - `Trainer` iterative flow supports configurable gain/leaf-size stop guards for stump rounds.
   - Default behavior remains deterministic and backward compatible with existing usage.
2. Engine artifact-coverage package:
   - `TrainedModel` artifact export emits both:
     - `ModelSectionKind::Trees`
     - `ModelSectionKind::PredictorLayout`
   - Artifact import validates required sections and rejects duplicate section kinds for required payloads.
3. Tests and doc-hygiene package:
   - Engine tests prove control guards can stop zero-gain/under-sized splits.
   - Engine tests prove missing/duplicate required sections are rejected.
   - Python module/test docstrings no longer claim `v0.0.3`.

## Implementation Plan
1. Add `v0.0.5` plan artifact before code changes.
2. Introduce iterative-control configuration in `crates/engine/src/lib.rs` and wire it into `fit_iterations`.
3. Add predictor-layout payload encode/decode and required-section validation in engine artifact export/import.
4. Add/update engine tests for iterative-control and artifact-section guard behavior.
5. Apply separate naming/path drift fix in Python docstrings.
6. Run verification suite and record evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify iterative controls can prevent stump addition when gain threshold is not met and when leaves would be under minimum row count.
5. Engine tests verify artifact import rejects missing required `PredictorLayout` or `Trees` sections and rejects duplicate required section kinds.
6. Engine artifact roundtrip still preserves prediction consistency with dual-section export/import.
7. Python docstrings in drift-reported files no longer reference `v0.0.3`.

## Risks and Mitigations
- Risk: control knobs introduce behavior divergence from prior test fixtures.
  - Mitigation: keep defaults equivalent to current permissive behavior and add targeted regression tests.
- Risk: stricter artifact parsing rejects older single-section bytes.
  - Mitigation: keep this layer scoped to internal v0.0.5 contract update and document incompatibility explicitly in notes.
- Risk: naming-fix edits accidentally alter runtime behavior.
  - Mitigation: restrict Python changes to docstring text only and rely on existing unit tests.

## Exit Condition
`v0.0.5` is complete when loop-control and dual-section artifact behaviors are test-backed, verification commands pass, naming drift fix is landed, and implementation/verification artifacts are recorded.
