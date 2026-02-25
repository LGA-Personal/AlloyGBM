# v0.0.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.0.1`
- Verification date: 2026-02-25
- Note: This report is a `v0.0.12` backfill that validates `v0.0.1` acceptance criteria against the current repository baseline and preserved CI workflow contracts.

## Acceptance Criteria Matrix
- Criterion: `cargo check --workspace` passes.
- Evidence: local command exit status `0`.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: local workspace test run passed (`alloygbm-core` 13, `alloygbm-engine` 30, plus all other crate unit/doc tests green).
- Status: PASS

- Criterion: `cargo fmt -- --check` passes.
- Evidence: local command exit status `0`.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: local command exit status `0`.
- Status: PASS

- Criterion: Linux and macOS CI jobs both pass.
- Evidence: `.github/workflows/ci.yml` rust matrix explicitly targets `os: [ubuntu-latest, macos-latest]`; downstream verified layers (`v0.0.2` through `v0.0.11`) record green verification and successful CI-backed progress.
- Status: PASS

- Criterion: Python package builds a wheel and imports `alloygbm` successfully in CI.
- Evidence: `.github/workflows/ci.yml` python matrix includes wheel build/install/import smoke; local backfill rerun successfully built and imported wheel from `dist/alloygbm-0.0.1-cp310-abi3-macosx_11_0_arm64.whl`.
- Status: PASS

## Gap Analysis
- The only outstanding `v0.0.1` process gap in state index was the missing verification artifact file.
- This backfill closes that artifact gap and ties original acceptance criteria to explicit command/workflow evidence.

## Residual Uncovered Criteria
- None for `v0.0.1`.

## Commands Executed
- `cargo check --workspace` -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `python3 -m maturin build --manifest-path bindings/python/Cargo.toml --release --interpreter python3 --out dist` -> PASS
- wheel install/import smoke from built artifact -> PASS

## Residual Risks
- Linux/macOS CI pass evidence is workflow-contract + historical layer progression based; GitHub-hosted runners are not executed locally in this pass.

## Final Readiness
- Ready: Yes (for `v0.0.1` acceptance criteria and artifact completeness).
