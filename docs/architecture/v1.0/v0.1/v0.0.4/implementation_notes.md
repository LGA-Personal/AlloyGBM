# v0.0.4 Implementation Notes

## Summary of What Was Built
- Added initial model artifact binary utilities in `crates/core/src/lib.rs`:
  - `serialize_model_artifact_v1`
  - `deserialize_model_artifact_v1`
  - artifact section/container types (`ModelArtifactSection`, `ParsedModelArtifactV1`)
- Added iterative multi-round training support in `crates/engine/src/lib.rs`:
  - `Trainer::fit_iterations` for repeated stump-style updates
  - helper validation for gradient shape, dataset/bin alignment, and partition coverage
- Added trained-model representation and inference helpers in engine:
  - `TrainedModel` and `TrainedStump`
  - `predict_row` and `predict_batch`
- Added engine-level model artifact roundtrip:
  - `TrainedModel::to_artifact_bytes`
  - `TrainedModel::from_artifact_bytes`
  - payload encoder/decoder for initial trees section binary format

## Non-Intuitive Decisions
- Kept artifact export confined to a single `Trees` section for this layer. This satisfies initial end-to-end emission without prematurely locking a broader multi-section production layout.
- Retained `fit_one_round` and `fit_stub` behavior for compatibility while introducing `fit_iterations` as the new iterative path.
- Stored minimal stump metadata (split + leaf values + row counts) in payload to keep binary parsing deterministic and small.

## Plan Contradictions and Why
- No contradictions with `v0.0.4/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate boundary changes were made.
- Planned evolution occurred within existing boundaries:
  - `core`: artifact IO primitives
  - `engine`: iterative training/model + artifact glue
  - `backend_cpu`: unchanged behavior, reused as primitive backend

## Known Gaps Deferred to v0.0.5+
- Iterative training still uses stump-level updates, not full depth-controlled tree growth.
- Artifact payload format is initial and engine-specific; predictor/shap/categorical sections are not yet emitted.
- Python bindings are still baseline-only and do not yet load engine-emitted artifacts.

## Follow-Up Actions
- Plan `v0.0.5` for richer tree-growth loop controls and first predictor-facing artifact compatibility.
- Add negative tests for malformed section ordering/duplication in engine artifact loaders.
- Extend artifact metadata to carry explicit model hyperparameter snapshot.
