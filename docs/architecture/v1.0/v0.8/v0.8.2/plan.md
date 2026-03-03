# AlloyGBM v0.8.2 Plan (v0.8 Test-Gap Closure Slice)

## Summary
- Goal: execute `v0.8.2` as the `v0.8` test-gap closure slice by expanding deterministic edge/compatibility coverage for Python-to-native contract boundaries without changing production behavior.
- Success criteria:
  - targeted missing edge cases from the `v0.8.1` hardening matrix are covered by executable tests,
  - compatibility and error-semantics paths are validated at the Python regressor boundary,
  - `v0.8.2` artifacts are complete and state advances to `v0.8.3`.
- Audience: engineers and reviewers validating release-hardening progress toward `v0.8` parent closeout.

## Scope
### In Scope
- Add targeted contract tests in `bindings/python/tests/test_regressor_contract.py` for:
  - feature-importance feature-count mismatch rejection before native calls,
  - bytes-like artifact payload compatibility (`bytearray`, `memoryview`) in `predict_from_artifact`,
  - categorical feature index bounds validation during fit.
- Produce `v0.8.2` layer artifacts:
  - `docs/architecture/v1.0/v0.8/v0.8.2/plan.md`
  - `docs/architecture/v1.0/v0.8/v0.8.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/v0.8.2/verification_report.md`
- Update `docs/architecture/state/layer_index.yaml` to:
  - mark `v0.8.2` verified,
  - create/advance next target `docs/architecture/v1.0/v0.8/v0.8.3`.

### Out of Scope
- Algorithmic/model behavior changes in Rust crates (`core`, `engine`, `predictor`, `shap`, `categorical`).
- Benchmark reproducibility packaging (`v0.8.3` scope).
- Migration/compatibility narrative finalization (`v0.8.4` scope).
- Parent `v0.8` rollup artifacts.

## Interfaces and Types
- Test/interface boundary in scope:
  - `bindings/python/alloygbm/regressor.py` contract behavior (validated via tests only).
  - `bindings/python/tests/test_regressor_contract.py` as the executable evidence surface.
- State tracking interface:
  - `docs/architecture/state/layer_index.yaml`.

Backward-compatibility expectations:
- no production API signature changes,
- no Rust runtime behavior changes,
- additive coverage only; existing passing tests remain green.

## Implementation Sequence
1. Derive missing compatibility/error-semantics edges from `v0.8.1` matrix commitments.
2. Implement targeted tests in Python contract suite for identified gaps.
3. Run full verification gate commands and ensure non-regression.
4. Write implementation and verification artifacts for this layer.
5. Update layer index status and advance next child target to `v0.8.3`.

## Test Cases and Scenarios
- Contract edge cases:
  - `feature_importances` rejects feature count mismatch without calling native bridge.
  - `predict_from_artifact` accepts `bytearray` payloads and forwards bytes to native bridge.
  - `predict_from_artifact` accepts `memoryview` payloads and forwards bytes to native bridge.
  - fit rejects out-of-bounds `categorical_feature_index`.
- Non-regression gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.8.2/plan.md` is present and decision-complete.
2. Targeted contract tests are added for feature-count mismatch, bytes-like artifact payloads, and categorical index bounds.
3. `docs/architecture/v1.0/v0.8/v0.8.2/implementation_notes.md` is present.
4. `docs/architecture/v1.0/v0.8/v0.8.2/verification_report.md` is present with criterion-to-evidence mapping.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
10. `docs/architecture/state/layer_index.yaml` marks `v0.8.2` verified and advances next target to `docs/architecture/v1.0/v0.8/v0.8.3`.

## Risks and Mitigations
- Risk: coverage additions accidentally assert implementation details that are unstable.
  - Mitigation: test public contract behavior and bridge-routing semantics only.
- Risk: edge-case tests expose latent regressions outside this slice.
  - Mitigation: treat failures as blockers and keep fixes constrained to hardening scope.
- Risk: layer-state transitions drift from produced artifacts.
  - Mitigation: update `layer_index.yaml` only after all required artifacts and gate runs pass.

## Assumptions and Defaults
- `v0.8.2` remains test-centric with minimal/no production code changes.
- Next child target after verification is `docs/architecture/v1.0/v0.8/v0.8.3`.
- Full gate command reruns remain required evidence for slice closeout.
