# AlloyGBM v0.2 Parent Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2`
- Date: 2026-02-26

## Parent Goal Matrix
- Goal 1: depth-limited histogram CART growth for regression.
- Evidence:
  - `v0.1.5` verification: multi-node-per-round depth-limited growth validated (`fit_iterations_grows_multiple_nodes_per_round_when_depth_allows`), plus path-conditioned non-root application.
  - `v0.1.4` verification: validation rollback semantics aligned to best-checkpoint behavior.
- Status: PASS

- Goal 2: training loop controls for shrinkage, row subsampling, column subsampling, and validation early stopping.
- Evidence:
  - `v0.1.1` verification: control contracts and validation-stop interface implemented and validated.
  - `v0.1.2` verification: seeded deterministic per-round row/column subsampling semantics and sample-count telemetry validated.
  - `v0.1.5` verification: control contracts remain passing while depth behavior expands.
- Status: PASS

- Goal 3: row and batch prediction path from trained model artifacts.
- Evidence:
  - `v0.1.6` verification: predictor strict/legacy artifact load + row/batch parity with engine validated.
  - direct test evidence: `crates/predictor/src/lib.rs` tests
    - `predictor_from_artifact_matches_engine_predictions` (batch parity),
    - `predictor_row_matches_engine_prediction` (single-row parity),
    - `predictor_accepts_legacy_trees_only_artifact` (artifact compatibility).
  - `v0.1.7` verification: Python native bridge entrypoint validates predictor-backed artifact inference path.
  - `v0.1.8`/`v0.1.9` verification: runtime wheel-install import path executes both error and success parity paths from Python runtime.
- Status: PASS

- Goal 4: correctness-oriented quality signal beats naive baseline.
- Evidence:
  - `v0.1.3` verification: CPU backend integration test `cpu_backend_training_beats_naive_baseline_mse` validates model loss below naive constant baseline.
- Status: PASS

- Goal 5: verification command gates for workspace and Python runtime tests pass at closeout.
- Evidence (closeout run, 2026-02-26):
  - `cargo fmt -- --check` -> PASS
  - `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
  - `cargo test --workspace` -> PASS
  - `cargo doc --workspace --no-deps` -> PASS
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 15 tests`)
- Status: PASS

## Residual Uncovered Criteria
- Performance target from roadmap context (`~3-5x` LightGBM CPU on selected dense tasks) remains uncovered by a dedicated in-repo benchmark comparison artifact.
- Status: BLOCKED (no LightGBM comparison harness/report captured in current repository closeout artifacts).

## Commands Executed for Closeout
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Risks
- Roadmap-level LightGBM-relative performance target evidence remains blocked without a dedicated benchmark comparison artifact.
- Parent `v1.0` rollup verification is still pending.

## Final Readiness
- Ready: Yes (parent `v0.2` closeout complete for architecture-layer scope based on verified child slices and passing gate commands).
