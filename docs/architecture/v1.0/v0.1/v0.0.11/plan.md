# AlloyGBM v0.0.11 Plan (v0.1 Week 11 CI Closeout and Python Contract Smoke)

## Objective
Close the remaining `v0.1` acceptance gaps by hardening CI evidence for Rust docs/build checks and installed-wheel Python contract behavior.

## Scope
- In scope:
  - Add a Rust docs/build verification step to CI on Linux and macOS.
  - Add installed-wheel Python contract smoke in CI to prove `GBMRegressor` constructor/parameter validation is callable from the packaged artifact.
  - Keep existing Rust/Python CI checks unchanged otherwise.
- Out of scope:
  - Trainer algorithm behavior changes.
  - New model IO semantics or artifact format changes.
  - Python estimator feature expansion beyond smoke-level contract validation.
  - Any `v0.2+` functionality.

## Deliverables
1. CI docs/build package:
  - `.github/workflows/ci.yml` includes a Rust docs check (`cargo doc --workspace --no-deps`) in the Rust matrix job.
2. Python wheel contract smoke package:
  - `.github/workflows/ci.yml` includes an installed-wheel smoke step that exercises:
    - `alloygbm.GBMRegressor(...)` constructor
    - invalid parameter rejection via `set_params(...)`
3. Verification package:
  - local verification evidence for fmt/clippy/tests/docs.
  - layer artifacts: `implementation_notes.md`, `verification_report.md`.

## Implementation Plan
1. Add `v0.0.11` plan artifact.
2. Update CI workflow:
  - insert `cargo doc --workspace --no-deps` in Rust matrix job.
  - extend Python smoke job with a small inline contract script after wheel install.
3. Run verification commands and capture evidence.
4. Record implementation and verification artifacts.
5. Update state index to reflect `v0.0.11` status and next suggested layer.

## Acceptance Criteria
1. `.github/workflows/ci.yml` runs `cargo doc --workspace --no-deps` in the Rust matrix job.
2. `.github/workflows/ci.yml` runs an installed-wheel Python smoke script that constructs `GBMRegressor` and confirms invalid parameter validation raises.
3. `cargo fmt -- --check` passes.
4. `cargo clippy --workspace --all-targets -- -D warnings` passes.
5. `cargo test --workspace` passes.
6. `cargo doc --workspace --no-deps` passes locally.
7. Existing Python tests remain passing.

## Risks and Mitigations
- Risk: docs check increases CI time.
  - Mitigation: keep docs build `--no-deps` and in existing Rust matrix only.
- Risk: wheel contract smoke may be brittle.
  - Mitigation: keep smoke minimal and assertion-focused on stable baseline contract behavior.
- Risk: scope drift toward estimator feature work.
  - Mitigation: limit Python smoke to constructor/validation checks only.

## Exit Condition
`v0.0.11` is complete when CI includes docs/build + wheel contract smoke checks, local verification passes, and artifacts are recorded.
