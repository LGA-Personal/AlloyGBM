# AlloyGBM v0.6.3 Implementation Notes

## Summary of What Was Built
- Executed `v0.6.3` engine/predictor categorical integration slice.
- Updated engine dependencies in [Cargo.toml](/Users/lashby/Projects/AlloyGBM/crates/engine/Cargo.toml) to include `alloygbm-categorical`.
- Added engine-side categorical preprocessing and integration in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/engine/src/lib.rs):
  - `CategoricalTargetEncodingSpec`,
  - `Trainer::fit_iterations_with_single_target_encoded_feature`,
  - deterministic helper flow for single-feature target encoding and encoded-bin mapping.
- Extended model artifact integration in engine:
  - `TrainedModel` now carries optional `categorical_state`,
  - `to_artifact_bytes` emits `ModelSectionKind::CategoricalState` when present,
  - `from_artifact_bytes_with_mode` decodes optional categorical section via core helper and validates feature bounds.
- Extended predictor artifact loading in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/predictor/src/lib.rs):
  - predictor now decodes and stores optional categorical state while preserving prediction behavior.
- Added/updated integration tests in engine/predictor to lock:
  - state attachment from categorical-aware training wrapper,
  - artifact roundtrip with optional categorical state,
  - predictor replay parity for artifacts containing categorical state.

## Non-Intuitive Decisions
- Decision: `v0.6.3` training-path integration is limited to a single-feature target-encoding wrapper rather than full multi-feature orchestration.
- Reason: this layer is integration scaffolding; full pipeline breadth and Python exposure are deferred by plan to `v0.6.4+`.
- Impact: categorical integration is testable and artifact-aware now, but broader multi-feature ergonomics remain follow-on work.

- Decision: encoded float values are deterministically mapped to ordinal bins by sorted unique values before reuse in existing histogram path.
- Reason: current trainer consumes `BinnedMatrix`; this mapping provides deterministic compatibility without redesigning split primitives in this layer.
- Impact: integration correctness is established, while binning strategy optimization remains future work.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Scope stayed inside Rust crates `engine` and `predictor` plus layer docs/state.
- Existing numeric-only training and predictor APIs remained unchanged.
- Python binding behavior was intentionally not expanded in this slice.

## Known Gaps Deferred to Next Layer
- Python categorical fit/predict bridge and ergonomic surface are still deferred (`v0.6.4`).
- Multi-feature categorical preprocessing orchestration is not yet implemented.
- Parent `v0.6` rollup artifacts remain pending until all child slices complete.

## Follow-Up Actions
- Create `docs/architecture/v1.0/v0.6/v0.6.4/plan.md` for Python bridge + end-to-end categorical flow exposure.
- Expand engine categorical wrapper from single-feature scope to broader feature-set orchestration where required.
