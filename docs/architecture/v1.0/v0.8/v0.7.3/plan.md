# AlloyGBM v0.7.3 Plan (v0.8 SHAP Compatibility and Predictor-Parity Hardening Slice)

## Summary
- Goal: execute `v0.7.3` by hardening SHAP compatibility behavior and locking prediction-parity checks against the predictor artifact path.
- Success criteria:
  - SHAP additivity is validated against predictor outputs (not only `TrainedModel` predictions),
  - compatibility behavior is covered for strict, legacy, and malformed required-section layouts,
  - global-importance ordering remains deterministic under tie conditions.
- Audience: engineers closing `v0.8` Rust SHAP hardening before Python bridge work in `v0.7.4`.

## Scope
### In Scope
- `crates/shap` hardening updates:
  - add predictor-parity integration coverage for artifact-backed SHAP explanations,
  - expand compatibility tests for legacy trees-only artifacts and malformed section layouts,
  - add deterministic ordering coverage for global-importance tie cases.
- `crates/shap/Cargo.toml`:
  - add test-only dependency wiring needed for predictor-parity checks.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.8/v0.7.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/v0.7.3/verification_report.md`

### Out of Scope
- Python SHAP bindings and `GBMRegressor.shap_values` (`v0.7.4` scope).
- New model artifact sections or model format version changes.
- SHAP interaction values, approximate modes, or GPU/Metal SHAP execution.
- Performance optimization of exact subset enumeration.

## Interfaces and Types
- `crates/shap/src/lib.rs`:
  - preserve public SHAP API signatures,
  - preserve existing deterministic `ShapError::{InvalidInput, ContractViolation}` model,
  - add/expand tests to lock compatibility + predictor parity expectations.
- `crates/shap/Cargo.toml`:
  - add test-scoped dependency on `alloygbm-predictor`.

Backward-compatibility expectations:
- SHAP public APIs remain unchanged.
- Predictor/engine behavior remains unchanged (test-only cross-checks).
- Artifact strict/legacy compatibility policy remains unchanged.

## Deliverables
1. Predictor-parity test package:
  - artifact-backed SHAP reconstruction parity checks against predictor outputs.
2. Compatibility-hardening test package:
  - legacy trees-only acceptance and malformed required-section rejection coverage.
3. Deterministic-ordering package:
  - tie-break coverage for global-importance sorting.
4. Layer evidence package:
  - `implementation_notes.md`, `verification_report.md`, and layer index update.

## Implementation Sequence
1. Author `v0.7.3` plan and lock scope to hardening/parity work.
2. Add test dependency wiring in `crates/shap/Cargo.toml`.
3. Add/expand `crates/shap` tests for predictor parity, compatibility edges, and global-importance tie ordering.
4. Run verification gates and resolve failures.
5. Write `implementation_notes.md` and `verification_report.md`.
6. Update `docs/architecture/state/layer_index.yaml` to mark `v0.7.3` verified and set `v0.7.4` as the next target.

## Test Cases and Scenarios
- Unit cases:
  - global-importance deterministic tie ordering,
  - metadata/payload feature-count contract mismatch rejection.
- Integration cases:
  - additivity reconstruction from SHAP equals predictor outputs for fixture rows,
  - legacy trees-only artifact remains explainable.
- Failure and edge cases:
  - duplicate required sections rejected with contract violation,
  - strict/legacy compatibility gate remains deterministic.
- Acceptance test mapping:
  - `cargo test -p alloygbm-shap`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.7.3/plan.md` exists and is decision-complete.
2. SHAP additivity parity is validated against `alloygbm-predictor` outputs on artifact-backed fixtures.
3. SHAP artifact compatibility coverage includes strict dual-section artifacts, legacy trees-only artifacts, and malformed required-section artifacts.
4. Global importance ordering is deterministic for equal-magnitude contributions.
5. `docs/architecture/v1.0/v0.8/v0.7.3/implementation_notes.md` is created.
6. `docs/architecture/v1.0/v0.8/v0.7.3/verification_report.md` is created.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. Python unittest suite passes.

## Risks and Mitigations
- Risk: predictor-parity tests can be brittle if fixture assumptions are under-specified.
  - Mitigation: use deterministic fixture artifacts and tolerance-based equality checks.
- Risk: compatibility hardening drifts into model-format changes.
  - Mitigation: keep all changes test-focused and preserve existing compatibility mode behavior.
- Risk: scope creep into Python bridge before Rust hardening closeout.
  - Mitigation: defer all Python SHAP bridge work to `v0.7.4`.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Additivity tolerance remains `1e-5` absolute error.
- Required-section compatibility continues to allow legacy trees-only artifacts for SHAP loading.
