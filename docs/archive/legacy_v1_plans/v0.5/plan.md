# AlloyGBM v0.5 Technical Plan

## Summary
- Goal: deliver the `0.5.0` milestone by making model artifact IO and predictor integration the canonical inference path ahead of `1.0.0`.
- Success criteria:
  - training output and inference input converge on a single validated artifact contract,
  - predictor-backed inference is the default runtime path for model scoring,
  - model-format v1 compatibility policy is explicit, tested, and documented for release gating.
- Audience: engineers implementing `v0.5` child slices and reviewers gating readiness for `v0.6+` feature work.

## Scope
### In Scope
- Model artifact IO hardening across `core`, `engine`, `predictor`, and Python binding surfaces:
  - deterministic serialization/deserialization roundtrip behavior,
  - explicit required-section validation (`Trees` + `PredictorLayout`) and legacy trees-only compatibility handling,
  - clear compatibility-mode behavior and failure semantics.
- Predictor integration as canonical inference route:
  - training flow produces artifacts consumed by predictor path without adapter-only logic,
  - Python prediction flows remain contract-stable while relying on predictor artifact execution.
- `v0.5` child-layer decomposition using `v0.4.x` slices.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.5/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/verification_report.md`

### Out of Scope
- New objectives (ranking), SHAP expansion, categorical expansion, or GPU backends.
- Performance-only kernel optimization campaigns (`v0.4` scope, already closed).
- Public Python API redesign that changes `GBMRegressor` constructor or existing method signatures.
- Model format version bump beyond v1.

## Interfaces and Types
- `crates/core/src/lib.rs`:
  - artifact headers/section descriptors and metadata JSON contract (`MODEL_FORMAT_V1`, `ModelSectionKind`, serialize/deserialize helpers).
- `crates/engine/src/lib.rs`:
  - `TrainedModel::{to_artifact_bytes, from_artifact_bytes, from_artifact_bytes_auto, artifact_compatibility_report}` as training/artifact boundary.
- `crates/predictor/src/lib.rs`:
  - `Predictor::from_artifact_bytes`, `predict_row`, `predict_batch` as canonical inference boundary.
- `bindings/python/alloygbm/regressor.py` and `bindings/python/src/lib.rs`:
  - native train/predict bridge using artifact bytes; no breaking parameter-surface changes.
- Verification artifacts and state tracking:
  - child reports under `docs/architecture/v1.0/v0.5/v0.4.x/`,
  - parent closeout plus `docs/architecture/state/layer_index.yaml`.

Backward-compatibility expectations:
- keep existing model-format v1 artifact readability (strict dual-section and documented legacy trees-only mode),
- keep Python estimator behavior and existing tests stable while switching internals to canonical predictor artifact usage.

## Implementation Sequence
1. Execute `docs/architecture/v1.0/v0.5/v0.5.1/plan.md` for artifact compatibility policy lock-in and baseline IO contract tests.
2. Open `v0.5.2` for predictor-path canonicalization across engine/predictor/Python scoring flow.
3. Open `v0.5.3` for serialization contract hardening, migration notes, and failure-mode consistency.
4. Open optional `v0.5.4` for residual acceptance gaps, docs polish, and release-evidence packaging.
5. Close parent `v0.5` with rollup notes, verification report, and `layer_index.yaml` update.

## Test Cases and Scenarios
- Unit cases:
  - artifact header/section validation and deterministic metadata roundtrip,
  - compatibility-mode classification and expected mode selection behavior.
- Integration cases:
  - train -> artifact -> predictor prediction parity versus engine prediction paths,
  - Python `GBMRegressor.fit` + `predict` and `predict_from_artifact` parity on deterministic fixtures.
- Failure and edge cases:
  - malformed artifacts (missing/duplicate required sections, invalid lengths/versions),
  - feature-count mismatch and unsupported compatibility-mode paths.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `v0.5` child slices establish a decision-complete artifact compatibility policy for model-format v1 (strict and legacy behavior explicitly documented and tested).
2. Predictor ingestion from training artifacts is validated as the canonical inference path with parity evidence against engine predictions.
3. Python artifact-backed inference workflows remain green without public API breakage.
4. Artifact validation behavior for malformed/unsupported payloads is deterministic and covered by tests.
5. Parent rollup artifacts summarize compatibility decisions, residual caveats, and evidence links for all child slices.
6. `cargo fmt -- --check` passes at closeout.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
8. `cargo test --workspace` passes at closeout.
9. `cargo doc --workspace --no-deps` passes at closeout.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes at closeout.

## Risks and Mitigations
- Risk: compatibility behavior drifts between `engine` and `predictor`.
  - Mitigation: enforce shared artifact-fixture parity tests and compatibility-report assertions.
- Risk: legacy trees-only behavior becomes ambiguous during hardening.
  - Mitigation: document compatibility modes and keep explicit tests for strict vs legacy paths.
- Risk: internal canonicalization accidentally changes Python-visible behavior.
  - Mitigation: treat Python contract tests as hard gates for each child slice.
- Risk: scope creep into `v0.6+` feature milestones.
  - Mitigation: constrain all child plans to model IO and predictor integration only.

## Assumptions and Defaults
- Device scope remains CPU-only in `v0.5`.
- `v0.5` child layers use `v0.4.x` numbering.
- Default execution order starts at `v0.5.1` before opening later child slices.
- Model format remains v1 in this layer; compatibility policy is frozen/documented rather than version-bumped.
