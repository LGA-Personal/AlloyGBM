# AlloyGBM v0.6.2 Plan (Deterministic Categorical Encoder Slice)

## Summary
- Goal: execute the `v0.6.2` slice by replacing categorical placeholders with deterministic target and frequency/count encoder implementations in `crates/categorical`.
- Success criteria:
  - target encoder supports deterministic fit/transform and leakage-safe training transforms when `time_aware=true`,
  - frequency encoder supports deterministic fit/transform with explicit unknown-category behavior,
  - unit coverage locks config validation, deterministic ordering, and leakage-safe defaults.
- Audience: engineers implementing `v0.7` categorical internals and reviewers gating readiness for engine integration in `v0.6.3`.

## Scope
### In Scope
- Implement target encoding runtime in `crates/categorical/src/lib.rs`:
  - deterministic state fitting (category stats + global mean),
  - transform from fitted state,
  - fit-transform behavior for time-aware and non-time-aware modes.
- Implement frequency/count encoding runtime in `crates/categorical/src/lib.rs`:
  - deterministic state fitting with per-category counts/frequencies,
  - transform with explicit unknown-category fallback.
- Validation and determinism hardening:
  - target/config/shape checks with deterministic error semantics,
  - deterministic category ordering and stable outputs for same input.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.7/v0.6.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.7/v0.6.2/verification_report.md`

### Out of Scope
- Wiring encoders into engine training flow and artifact replay (`v0.6.3` scope).
- Python API/bridge expansion for categorical fit/predict controls (`v0.6.4` scope).
- Learned categorical embeddings and ranking-specific categorical handling.
- Model format version changes.

## Interfaces and Types
- `crates/categorical/src/lib.rs`:
  - `TargetEncoderConfig`,
  - target-encoding state types and fit/transform APIs,
  - frequency-encoding state types and fit/transform APIs,
  - deterministic validation + error behavior.
- `crates/core/src/lib.rs`:
  - no new changes required in this slice; `v0.6.1` contract helpers remain the integration anchor.

Backward-compatibility expectations:
- no public API breakage outside `crates/categorical`,
- numeric-only training/inference behavior in engine/predictor/python remains unchanged in this layer.

## Deliverables
1. Target encoding package:
  - deterministic fit/transform APIs with time-aware fit-transform mode.
2. Frequency encoding package:
  - deterministic fit/transform APIs with unknown-category fallback policy.
3. Test package:
  - unit tests for validation, determinism, leakage-safe defaults, and unknown-category behavior.
4. Verification package:
  - criterion-to-evidence mapping in `verification_report.md`.
5. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.6.2` completion and `v0.6.3` next-target suggestion.

## Implementation Sequence
1. Add `v0.6.2` plan and lock scope to `crates/categorical`.
2. Replace placeholder categorical stub with concrete target/frequency encoder state + fit/transform APIs.
3. Add deterministic time-aware fit-transform behavior for target encoding and enforce required `time_index` shape in time-aware mode.
4. Add/expand unit tests for:
  - input/config validation,
  - deterministic category ordering,
  - leakage-safe same-timestamp handling,
  - unknown-category fallback semantics.
5. Run targeted and full verification gates.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `layer_index.yaml` to mark `v0.6.2` verified and set `v0.6.3` next target.

## Test Cases and Scenarios
- Unit cases:
  - target encoder rejects invalid config and shape mismatches,
  - target encoder fit state is deterministic and sorted,
  - time-aware fit-transform does not consume same-timestamp targets for current-row encodings,
  - frequency encoder produces deterministic counts/frequencies.
- Integration cases:
  - `fit_transform_target_encoder` and `fit_transform_frequency_encoder` output lengths match input rows and produce deterministic values.
- Failure and edge cases:
  - missing `time_index` in `time_aware=true` mode,
  - non-finite targets rejection,
  - unknown-category fallback behavior for both encoders.
- Acceptance test mapping:
  - `cargo test -p alloygbm-categorical`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. Placeholder categorical stub is replaced with deterministic target and frequency/count encoder implementations.
2. Time-aware target fit-transform mode is implemented with deterministic, leakage-safe ordering semantics.
3. Unknown-category fallback behavior is explicit and tested for both target and frequency transforms.
4. `docs/architecture/v1.0/v0.7/v0.6.2/implementation_notes.md` is created.
5. `docs/architecture/v1.0/v0.7/v0.6.2/verification_report.md` is created.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: leakage-safe behavior is underspecified for same-timestamp rows.
  - Mitigation: lock deterministic group-by-time semantics in tests.
- Risk: categorical API churn creates integration rework in `v0.6.3`.
  - Mitigation: keep APIs small/state-driven and document deferred integration boundaries.
- Risk: stricter validation breaks downstream assumptions.
  - Mitigation: return explicit error messages and preserve deterministic defaults.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.6.2` is `crates/categorical` only; engine/predictor/python integration remains deferred.
- Time-aware mode defaults to leakage-safe grouping by timestamp.
