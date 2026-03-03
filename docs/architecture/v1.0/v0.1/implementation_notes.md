# AlloyGBM v0.1 Parent Implementation Notes

## Scope and Purpose
This parent rollup summarizes the completed `v0.1` child layers (`v0.1.1` through `v0.1.10`) and records how the milestone moved from contract scaffolding to a verified minimal histogram GBDT CPU regression baseline.

## Child-Layer Contributions
- `v0.1.1`: locked training-control contracts for `row_subsample`, `col_subsample`, `early_stopping_rounds`, `min_validation_improvement`, and validation-aware stop-path plumbing.
- `v0.1.2`: replaced prefix-only sampling with seeded deterministic per-round row/feature subsampling semantics and summary telemetry.
- `v0.1.3`: added real `Trainer + CpuBackend` integration evidence for baseline quality (`model_mse < naive_mse`) and deterministic artifact reproducibility.
- `v0.1.4`: corrected validation plateau rollback semantics to retain best-checkpoint model state and aligned summary traces.
- `v0.1.5`: implemented depth-limited multi-node-per-round growth and path-conditioned non-root split application semantics.
- `v0.1.6`: implemented predictor artifact import/inference parity (strict + legacy) with row/batch prediction path and input validation behavior.
- `v0.1.7`: exposed predictor-backed Python native bridge (`predictor_predict_batch`) and public regressor artifact inference path.
- `v0.1.8`: added Python runtime wheel-build/install integration tests for native extension import/execution and error-path propagation.
- `v0.1.9`: added Python runtime success-path parity assertions with valid artifact bytes and deterministic native prediction checks.
- `v0.1.10`: consolidated parent closeout artifacts/state-index bookkeeping and added explicit single-row predictor parity test evidence (`predictor_row_matches_engine_prediction`).

## Outcome vs v0.1 Goals
- Histogram-based regression behavior is implemented with depth-limited growth and path-aware application.
- Training loop controls (shrinkage, row/column subsampling, validation early stopping) are implemented and test-backed.
- Row and batch inference are available from serialized artifacts through predictor and Python bridge/runtime paths.
- Correctness-oriented command gates and Python integration coverage are green at closeout time.

## Non-Intuitive Decisions Across v0.1
- Kept predictor runtime independent from engine internals even when decode/path semantics were mirrored, to preserve lightweight inference boundaries.
- Implemented Python runtime checks via wheel-install harness to validate true extension execution rather than source-only imports.
- Used layered child slices to land behavior deterministically before closeout rollup, reducing cross-layer drift risk.

## Known Residuals at Parent Closeout
- LightGBM-relative performance benchmarking for the roadmap target (`3–5x`) is not yet represented in dedicated benchmark artifacts in this closeout.
- Parent `v1.0` rollup artifacts are still pending:
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Closeout Readiness
- Parent `v0.1` implementation scope is considered complete for architecture-layer closeout based on child-layer evidence and gate-command health.
