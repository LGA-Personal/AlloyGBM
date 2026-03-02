# AlloyGBM v0.6.2 Implementation Notes

## Summary of What Was Built
- Executed `v0.6.2` by replacing categorical placeholder logic with deterministic target and frequency encoder implementations in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/categorical/src/lib.rs).
- Added target encoder runtime:
  - state types: `TargetEncoderState`, `CategoryTargetStats`,
  - APIs: `fit_target_encoder`, `transform_target_encoder`, `fit_transform_target_encoder`,
  - leakage-safe time-aware fit-transform path with timestamp-grouped processing.
- Added frequency/count encoder runtime:
  - state types: `FrequencyEncoderState`, `CategoryFrequency`,
  - APIs: `fit_frequency_encoder`, `transform_frequency_encoder`, `fit_transform_frequency_encoder`.
- Added deterministic validation and error semantics:
  - input/config validation for value/target/time-index shape and non-finite targets,
  - deterministic category ordering via `BTreeMap`,
  - explicit unknown-category fallback (target -> global mean, frequency -> `0.0`).
- Added test coverage (10 categorical tests) that locks determinism and leakage-safe defaults.

## Non-Intuitive Decisions
- Decision: in time-aware fit-transform mode, rows sharing the same timestamp are encoded using only strictly earlier timestamps.
- Reason: this avoids within-timestamp target leakage while preserving deterministic behavior.
- Impact: same-timestamp rows do not influence each other’s encoded values during training transform.

- Decision: cold-start prior mean in time-aware mode uses `0.0` before any historical observations exist.
- Reason: avoids introducing future-information leakage at the first timestamp.
- Impact: first timestamp groups may encode to `0.0` when no prior history is available.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Scope remained inside `crates/categorical`; no `engine`, `predictor`, or Python binding runtime changes in this slice.
- Replaced old `fit_transform_stub`/`NotImplemented` placeholder flow with concrete deterministic APIs.
- No parent-layer API breakage outside categorical crate.

## Known Gaps Deferred to Next Layer
- Engine training path is not yet wired to categorical state/encoders (target `v0.6.3`).
- Predictor artifact replay with categorical execution state remains deferred to `v0.6.3`.
- Python-level categorical fit/predict controls remain deferred to `v0.6.4`.

## Follow-Up Actions
- Create `docs/architecture/v1.0/v0.7/v0.6.3/plan.md` for engine integration.
- Thread `v0.6.1` core categorical contract helpers with the new categorical runtime states during `v0.6.3`.
