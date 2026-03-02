# AlloyGBM v0.8 Technical Plan

## Summary
- Goal: deliver the `0.8.0` milestone by replacing SHAP placeholders with exact TreeSHAP for CPU regression artifacts and exposing global/per-row explanation APIs in Rust and Python.
- Success criteria:
  - SHAP values are deterministic and satisfy additivity (`expected_value + sum(phi_i) == prediction` within tolerance),
  - global feature importance is available as mean absolute SHAP contribution,
  - Python API gains additive SHAP entrypoints without breaking existing training/prediction behavior.
- Audience: engineers implementing `v0.8` child slices and reviewers gating readiness for `v0.9` hardening.

## Scope
### In Scope
- Rust SHAP runtime in `crates/shap`:
  - replace placeholder SHAP functions with exact TreeSHAP for current CPU regression tree payloads,
  - support per-row SHAP values for batch inputs,
  - support global importance aggregation from SHAP values.
- Artifact-backed SHAP contract:
  - decode model artifacts via existing `core` contract utilities,
  - require `Trees` section and validate feature dimensions against metadata/layout,
  - preserve existing model format version while consuming current payload structure.
- Python SHAP API surface:
  - add native bridge functions in `bindings/python/src/lib.rs` for per-row and global SHAP,
  - add `GBMRegressor.shap_values(X)` and SHAP-based `feature_importances` path backed by fitted artifact state.
- Child-layer planning and execution decomposition under `docs/architecture/v1.0/v0.8/v0.7.x`.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.8/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/verification_report.md`

### Out of Scope
- GPU/Metal/MLX SHAP acceleration.
- Interaction SHAP values or approximate SHAP modes.
- Ranking/classification SHAP semantics.
- Required model-format changes (including making `ShapAux` mandatory) or compatibility policy changes.
- Non-SHAP roadmap work (ranking, probabilistic outputs, constraints, distributed execution).

## Interfaces and Types
- `crates/shap/src/lib.rs`:
  - replace `shap_values_stub` / `global_importance_stub` with artifact-backed exact implementations,
  - standardize error semantics using `ShapError::{InvalidInput, ContractViolation}` with deterministic messages,
  - enforce row/feature shape validation before computation.
- `crates/core/src/lib.rs` (consume-only in this layer):
  - continue using `deserialize_model_artifact_v1`, `required_section_compatibility_report`, and section-kind contracts.
- `crates/predictor/src/lib.rs` and `crates/engine/src/lib.rs`:
  - remain source-of-truth for prediction parity and artifact shape; SHAP integration must not alter prediction behavior.
- `bindings/python/src/lib.rs`:
  - add pyfunctions for SHAP from artifact bytes and route Rust SHAP errors to Python exceptions.
- `bindings/python/alloygbm/regressor.py`:
  - add additive SHAP methods using the fitted `_artifact_bytes` and existing row validation paths.

Backward-compatibility expectations:
- `GBMRegressor.fit/predict` behavior and signatures remain unchanged.
- Existing artifact strict/legacy compatibility behavior remains unchanged.
- No required new artifact sections; SHAP must work with existing strict dual-section artifacts.

## Implementation Sequence
1. `v0.7.1`: lock SHAP contract and tests for artifact/row validation, output shape, and additivity expectations on deterministic fixtures.
2. `v0.7.2`: implement exact TreeSHAP core traversal for current regression tree payload, including expected-value handling and per-row SHAP outputs.
3. `v0.7.3`: implement global importance aggregation (mean absolute SHAP), artifact compatibility checks, and parity tests against predictor outputs.
4. `v0.7.4`: expose Python bridge APIs and regressor methods; add Python tests for SHAP shape, errors, and prediction additivity consistency.
5. Parent closeout: run full verification gate, publish `implementation_notes.md` + `verification_report.md`, and update `docs/architecture/state/layer_index.yaml`.

## Test Cases and Scenarios
- Unit cases:
  - deterministic SHAP for single-stump and multi-stump toy payloads,
  - feature-count mismatch and empty-row input validation,
  - global importance aggregation correctness and stable ordering.
- Integration cases:
  - artifact -> SHAP pipeline for trained fixture models,
  - additivity check per row against predictor predictions,
  - parity between Rust and Python SHAP outputs for identical artifact/rows.
- Failure and edge cases:
  - malformed artifacts (missing/duplicate required sections, invalid payload lengths),
  - unsupported compatibility layouts,
  - rows containing non-finite values or inconsistent widths.
- Acceptance test mapping:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `crates/shap` no longer returns placeholder `NotImplemented` for supported CPU regression artifacts.
2. Per-row SHAP output dimensionality is `rows x feature_count` and validates input shapes deterministically.
3. For verified fixtures, `expected_value + sum(phi_i)` matches predictor output within a documented floating-point tolerance.
4. Global importance is computed from mean absolute SHAP contribution and exposed in deterministic ordering.
5. SHAP artifact loading honors existing strict/legacy required-section compatibility checks without changing model format version.
6. Python exposes additive SHAP APIs without regressing existing fit/predict contract tests.
7. `v0.8` child slices and parent rollup artifacts are present and linked in verification.
8. `cargo fmt -- --check` passes at closeout.
9. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
10. `cargo test --workspace` and Python unittest suite pass at closeout.

## Risks and Mitigations
- Risk: exact TreeSHAP path logic is error-prone for encoded tree-node traversal.
  - Mitigation: fixture-driven additivity tests and cross-check against predictor decisions.
- Risk: numerical drift from floating-point accumulation yields unstable explanations.
  - Mitigation: deterministic accumulation order and explicit tolerance assertions.
- Risk: SHAP integration inadvertently changes predictor/engine contracts.
  - Mitigation: treat prediction parity tests as hard gate and keep model-format behavior unchanged.
- Risk: scope creep into interaction SHAP or GPU SHAP.
  - Mitigation: enforce explicit out-of-scope boundary in child slices.

## Assumptions and Defaults
- Device scope remains CPU-only for `v0.8`.
- Tree traversal semantics for SHAP follow the current predictor branch rule (`<= threshold_bin` goes left).
- Global importance default is mean absolute SHAP across provided rows.
- Child-layer numbering under `v0.8` uses `v0.7.x`; immediate next child target is `docs/architecture/v1.0/v0.8/v0.7.1`.
