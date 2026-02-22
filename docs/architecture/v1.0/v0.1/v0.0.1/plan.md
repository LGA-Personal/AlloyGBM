# AlloyGBM v0.0.1 Plan (v0.1 Week 1 Bootstrap)

## Objective
Deliver the first executable slice of `v0.1` by establishing repository structure, crate boundaries, and CI quality gates so all later interface work can land on stable scaffolding.

## Scope
- In scope:
  - Rust workspace bootstrap.
  - Initial crate creation and dependency boundaries.
  - Baseline lint/test/toolchain configuration.
  - Linux + macOS CI skeleton.
  - Python packaging scaffold for future bindings.
- Out of scope:
  - Trainer algorithm implementation.
  - Model format implementation details beyond stubs.
  - SHAP/categorical/ranking behavior.
  - Performance optimization.

## Deliverables
1. Root workspace files:
   - `Cargo.toml` workspace manifest with all planned member crates.
   - `rust-toolchain.toml` pinning stable Rust toolchain.
   - `.gitignore` covering Rust, Python, and build artifacts.
2. Crate scaffolding (compilable, minimal stubs):
   - `crates/core`
   - `crates/engine`
   - `crates/backend_cpu`
   - `crates/predictor`
   - `crates/shap`
   - `crates/categorical`
   - `bindings/python` (PyO3/maturin-ready skeleton)
3. Developer quality gates:
   - `cargo fmt` and `cargo clippy` config.
   - Workspace test command (`cargo test --workspace`) passing.
4. CI foundation:
   - Linux and macOS jobs.
   - Rust checks: build, fmt, clippy, tests.
   - Python smoke job for wheel build/import.

## Implementation Plan
1. Initialize workspace and manifests.
   - Create root workspace manifest and list all members.
   - Define shared edition/version policy.
2. Create all crate skeletons with minimal public modules.
   - Expose placeholder types/traits only where needed for cross-crate compile checks.
3. Establish dependency rules.
   - `engine` depends on `core` plus backend trait surface.
   - `backend_cpu` depends on `core` and implements stub ops.
   - `predictor`, `shap`, and `categorical` depend on `core` only at this stage.
4. Add Python binding scaffold.
   - Create importable `alloygbm` package shell.
   - Add minimal `GBMRegressor` class stub and native module import smoke path.
5. Add CI and local task commands.
   - Add workflow(s) for Rust checks and Python wheel smoke test.
   - Document run commands in a short developer section.

## Acceptance Criteria
1. `cargo check --workspace` passes.
2. `cargo test --workspace` passes.
3. `cargo fmt -- --check` passes.
4. `cargo clippy --workspace --all-targets -- -D warnings` passes.
5. Linux and macOS CI jobs both pass.
6. Python package builds a wheel and imports `alloygbm` successfully in CI.

## Risks and Mitigations
- Risk: early crate coupling makes future backend abstraction harder.
  - Mitigation: enforce dependency direction now and keep cross-crate APIs minimal.
- Risk: CI flakiness slows later interface iteration.
  - Mitigation: keep first workflows short, deterministic, and cache-aware.
- Risk: Python scaffold diverges from Rust crate naming/layout.
  - Mitigation: define package/crate naming map in this milestone and reuse consistently.

## Exit Condition
`v0.0.1` is complete when the repository is structurally ready for `v0.0.2` contract-definition work without additional bootstrap tasks.
