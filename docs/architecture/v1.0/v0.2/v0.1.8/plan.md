# AlloyGBM v0.1.8 Plan (v0.2 Python Runtime Extension Execution)

## Objective
Close the remaining `v0.2` Python integration evidence gap by adding automated Python-runtime tests that build/import the native extension module and execute predictor-backed inference entry points from Python test runtime.

## Scope
- In scope:
  - Add Python integration tests that build a wheel for `bindings/python` in test runtime and install it into an isolated temporary location.
  - Validate native module import/execution through the installed package (`alloygbm`) rather than mock-only bridge hooks.
  - Execute `predictor_predict_batch` through Python runtime and verify runtime error mapping behavior for invalid artifact payloads.
  - Execute `GBMRegressor.predict_from_artifact(...)` against the installed package and verify it routes through the native extension path.
- Out of scope:
  - Engine training/inference semantic changes.
  - Predictor artifact format changes.
  - New sklearn-surface Python APIs.
  - Parent-layer rollup artifacts for `v0.2` and `v1.0`.

## Deliverables
1. Runtime integration test package:
  - new Python test coverage under `bindings/python/tests/` for wheel-build/import/runtime invocation of native extension APIs.
2. Verification package:
  - `docs/architecture/v1.0/v0.2/v0.1.8/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/v0.1.8/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.1.8` completion and next target.

## Implementation Sequence
1. Create `v0.1.8` plan artifact.
2. Implement Python runtime integration test harness (wheel build + isolated install + import).
3. Add runtime execution tests for native APIs and public regressor bridge path.
4. Run verification command gates and collect criterion-mapped evidence.
5. Write implementation/verification artifacts and update layer state index.

## Acceptance Criteria
1. Python test suite includes at least one test that imports the installed `alloygbm` package from a built wheel and executes `native_runtime_info`.
2. Python test suite executes native `predictor_predict_batch` from Python runtime and validates expected error type on invalid artifact bytes.
3. Python test suite executes `GBMRegressor.predict_from_artifact(...)` from installed package runtime and validates native error propagation for invalid artifact bytes.
4. Existing Python regressor contract tests remain passing.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `cargo doc --workspace --no-deps` passes.
9. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: runtime integration tests become flaky due to global site-package interference.
  - Mitigation: use isolated temporary install target and module-cache cleanup in test setup/teardown.
- Risk: build-tool assumptions differ across environments.
  - Mitigation: invoke maturin via `python3 -m maturin` (same interpreter as test process) and assert wheel output presence explicitly.
- Risk: scope drift into feature expansion beyond evidence closure.
  - Mitigation: keep layer limited to runtime test harness and verification artifacts only.

## Exit Condition
`v0.1.8` is complete when Python-runtime extension import/execution evidence is automated in tests, all verification gates pass, and layer/state artifacts are updated.
