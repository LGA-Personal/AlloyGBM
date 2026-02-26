# AlloyGBM v0.1.6 Plan (v0.2 Predictor Artifact Inference Parity)

## Objective
Close a remaining `v0.2` completion gap by implementing the `predictor` crate artifact import and row/batch inference path so predictor outputs match engine outputs from the same serialized model bytes, including depth-grown/path-conditioned stump behavior introduced in `v0.1.5`.

## Scope
- In scope:
  - Implement predictor model import from artifact bytes (`Trees` + `PredictorLayout`) with legacy trees-only compatibility.
  - Decode trained-model payload in predictor crate and validate feature-count consistency against metadata/layout.
  - Implement predictor `predict_row` and `predict_batch` using path-aware non-root stump gating semantics consistent with engine behavior.
  - Replace placeholder stub behavior by wiring stub methods as aliases to real prediction methods for compatibility.
  - Add predictor tests proving:
    - engine/predictor inference parity from strict artifacts,
    - legacy trees-only artifact compatibility,
    - input validation errors for feature-count and empty-batch misuse.
- Out of scope:
  - Changes to engine training semantics.
  - New Python API surface.
  - Parent-layer (`v0.2` / `v1.0`) rollup artifacts.

## Deliverables
1. Predictor implementation package:
  - `crates/predictor/src/lib.rs` real artifact load + inference path.
  - `crates/predictor/Cargo.toml` test-only dependencies for cross-crate parity tests.
2. Verification package:
  - predictor tests for parity, legacy compatibility, and validation.
  - `docs/architecture/v1.0/v0.2/v0.1.6/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/v0.1.6/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.6` completion and next target.

## Implementation Sequence
1. Add `v0.1.6` plan artifact.
2. Implement predictor artifact section parsing and payload decoding.
3. Implement path-aware row/batch inference in predictor.
4. Add predictor parity + compatibility + validation tests.
5. Run verification command gates and capture evidence.
6. Write implementation/verification artifacts and update state index.

## Acceptance Criteria
1. Predictor crate can load strict dual-section artifacts and produce row/batch predictions.
2. Predictor crate accepts legacy trees-only artifacts using metadata feature-count fallback.
3. Predictor inference for strict artifacts matches engine inference from the same serialized model bytes on deterministic fixtures (row/batch parity).
4. Predictor rejects invalid input shapes (feature-count mismatch, empty batch).
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: predictor path semantics diverge from engine path-conditioned non-root behavior.
  - Mitigation: parity tests use engine-trained depth-grown artifacts and assert exact prediction equality.
- Risk: artifact compatibility drift between engine and predictor parsers.
  - Mitigation: mirror strict/legacy section rules and validate feature-count consistency across metadata/layout/payload.
- Risk: test-only cross-crate dependencies accidentally leak into runtime predictor dependencies.
  - Mitigation: keep `alloygbm-engine` and `alloygbm-backend-cpu` under `[dev-dependencies]` only.

## Exit Condition
`v0.1.6` is complete when predictor artifact inference is implemented with engine-parity evidence, legacy compatibility is verified, and full verification command gates pass.
