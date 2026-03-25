# v0.1.6 Implementation Notes

## Summary of What Was Built
- Implemented real predictor inference in `crates/predictor/src/lib.rs`:
  - artifact import via `Predictor::from_artifact_bytes`,
  - strict required-section handling (`Trees`, `PredictorLayout`) plus legacy trees-only fallback,
  - trained-model payload decoding for baseline and stump entries,
  - `predict_row` and `predict_batch` inference methods.
- Added path-aware non-root stump gating in predictor to match `v0.1.5` engine semantics:
  - tree/local node decoding from `node_id`,
  - ancestor path checks before applying child stump leaf values.
- Replaced placeholder-only behavior by keeping `predict_row_stub` and `predict_batch_stub` as compatibility aliases to real methods.
- Added predictor tests for parity/compatibility/validation:
  - `predictor_from_artifact_matches_engine_predictions`
  - `predictor_accepts_legacy_trees_only_artifact`
  - `predictor_row_rejects_feature_count_mismatch`
  - `batch_rejects_empty_rows`
- Added predictor crate test dependencies in `crates/predictor/Cargo.toml` under `[dev-dependencies]` for engine/backend parity fixtures.

## Non-Intuitive Decisions
- Decision: duplicate minimal artifact/payload decode logic inside predictor instead of importing engine internals.
- Reason: predictor must stay dependency-light and not pull in training-layer runtime dependencies.
- Impact: predictor remains independent at runtime, with cross-crate parity guaranteed by tests.

- Decision: preserve legacy trees-only compatibility in predictor.
- Reason: engine import path already supports legacy artifacts; predictor should remain interoperable with the same existing artifact corpus.
- Impact: predictor can serve both strict modern artifacts and legacy trees-only payloads without format migration in this layer.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.1.6/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No public Python/bindings interface changes.
- Predictor Rust crate API expanded from stubs to usable artifact inference methods.

## Known Gaps Deferred to Next Layer
- Parent rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/verification_report.md`
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Follow-Up Actions
- Define `v0.1.7` to continue remaining `v0.1` closeout scope (for example stronger end-to-end coverage from Python entry points into predictor-backed inference and any remaining parent rollup prerequisites).
