# AlloyGBM v0.7 Technical Plan

## Summary
- Goal: deliver the `0.7.0` milestone by implementing categorical support v1 with leakage-safe target encoding, frequency/count encoding, and artifact-integrated categorical state.
- Success criteria:
  - categorical encoding is deterministic and validated for both time-aware and non-time-aware flows,
  - training and prediction paths use a single artifact contract that carries categorical state when categorical features are enabled,
  - Python surface remains backward-compatible for existing numeric-only usage while adding categorical-capable options.
- Audience: engineers implementing `v0.7` child slices and reviewers gating readiness for `v0.8` TreeSHAP scope.

## Scope
### In Scope
- Categorical encoding functionality in `crates/categorical`:
  - leakage-safe target encoding with smoothing and minimum-sample controls,
  - frequency/count encoding for sparse identifiers,
  - deterministic fit/transform behavior under fixed inputs and config.
- Categorical metadata and artifact state integration:
  - explicit categorical-state payload carried through model artifacts using `ModelSectionKind::CategoricalState`,
  - schema-level tracking of categorical feature indices and encoder configuration needed for replay at inference.
- Engine and Python integration for categorical-aware training and prediction:
  - additive training options for categorical feature indices and optional time index where required for leakage-safe mode,
  - parity between in-memory prediction and artifact-backed predictor path when categorical state is present.
- Child-layer decomposition and execution under `v0.7` using `v0.6.x` slices.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.7/implementation_notes.md`
  - `docs/architecture/v1.0/v0.7/verification_report.md`

### Out of Scope
- Learned embeddings for categorical features.
- Ranking objectives and ranking-specific categorical strategies.
- GPU backend work (CUDA/Metal/MLX) and TreeSHAP implementation details (`v0.8` scope).
- Public API redesigns that break existing `GBMRegressor` constructor or current numeric-only workflows.
- Model-format version bump beyond v1.

## Interfaces and Types
- `crates/categorical/src/lib.rs`:
  - replace placeholder APIs with concrete encoder/state types and fit/transform entrypoints.
- `crates/core/src/lib.rs`:
  - maintain canonical categorical section kind (`ModelSectionKind::CategoricalState`) and metadata/schema validation for categorical configuration.
- `crates/engine/src/lib.rs`:
  - integrate categorical transform path into training and artifact emission without changing numeric-only behavior.
- `crates/predictor/src/lib.rs`:
  - support artifacts carrying categorical state while preserving strict required-section policy (`Trees` + `PredictorLayout`) and deterministic errors.
- `bindings/python/src/lib.rs` and `bindings/python/alloygbm/regressor.py`:
  - add categorical-capable, backward-compatible options for fit/predict flow.

Backward-compatibility expectations:
- numeric-only training and prediction results remain unchanged under existing tests.
- existing strict/compatibility artifact behavior from `v0.6` remains intact.
- new categorical options are additive; no required signature changes for current call sites.

## Implementation Sequence
1. Execute `docs/architecture/v1.0/v0.7/v0.6.1/` as the first child slice to lock categorical state contract, schema validation rules, and artifact serialization expectations.
2. Open `v0.6.2` for deterministic target/frequency encoder implementation in `crates/categorical` with unit coverage for leakage-safe defaults.
3. Open `v0.6.3` for engine integration: categorical preprocessing in training path, artifact persistence, and predictor replay checks.
4. Open `v0.6.4` for Python bridge integration and end-to-end contract tests across numeric + categorical fixtures.
5. Close parent `v0.7` with rollup notes, verification report, and `docs/architecture/state/layer_index.yaml` update.

## Test Cases and Scenarios
- Unit cases:
  - target-encoding smoothing behavior and minimum-leaf handling,
  - frequency/count encoding determinism and unknown-category handling,
  - categorical schema/config validation errors.
- Integration cases:
  - train -> artifact -> predictor parity for models with categorical features enabled,
  - numeric-only regression fixtures remain parity-stable versus pre-`v0.7` behavior.
- Failure and edge cases:
  - missing or invalid time index in leakage-safe mode,
  - category cardinality edge cases and empty-category partitions,
  - malformed categorical artifact section payloads.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `v0.7` child slices deliver production-ready target and frequency/count encoders with deterministic fit/transform semantics.
2. Leakage-safe categorical mode is explicitly validated, including time-index requirements and deterministic failure semantics.
3. Categorical state is serialized/deserialized through model artifacts without changing v1 format version.
4. Predictor path remains artifact-canonical and functionally consistent when categorical state is present.
5. Python categorical-capable flow is additive and keeps existing numeric-only contracts green.
6. Parent rollup artifacts summarize decisions, tradeoffs, and child-layer evidence links.
7. `cargo fmt -- --check` passes at closeout.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
9. `cargo test --workspace` passes at closeout.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes at closeout.

## Risks and Mitigations
- Risk: categorical transforms introduce silent leakage in temporal data.
  - Mitigation: leakage-safe mode defaults, explicit time-index validation, and focused temporal fixture tests.
- Risk: artifact compatibility drift between engine and predictor.
  - Mitigation: shared serialization fixtures and predictor parity tests for strict artifacts with categorical state.
- Risk: categorical integration destabilizes numeric-only behavior.
  - Mitigation: preserve numeric-only fast path and treat existing regression tests as hard gates.
- Risk: scope creep into SHAP/ranking/GPU milestones.
  - Mitigation: enforce `v0.7` boundary to categorical support v1 only.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.7` child layers use `v0.6.x` numbering.
- Immediate next child target is `docs/architecture/v1.0/v0.7/v0.6.1`.
- No `v0.5.x` child planning is created under `v0.7` in this step.
