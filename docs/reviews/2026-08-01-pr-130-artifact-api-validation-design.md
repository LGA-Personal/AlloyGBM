# PR #130 Artifact and API Validation Design

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-01 | OpenAI Codex | `main` after PR #129 | `1af8df3` | Approved for implementation |

## Objective

Close the remaining artifact resource-validation finding from the 2026-07-02 core review and
the adjacent public API dimension hazards without changing valid model behavior or estimator
lifecycle semantics. Full sklearn estimator conformance is deliberately reserved for PR #131.

## Scope

PR #130 will harden four boundaries:

1. Top-level artifact metadata and aggregate artifact resource budgets.
2. Counted section payloads and references between artifact sections.
3. Loaded engine and predictor model structure before it becomes callable.
4. Public dense, quantized, SHAP-binning, and persistence API dimensions.

The implementation will preserve the v1 binary format and valid existing artifacts. Unknown
metadata fields and bounded unknown objective labels remain forward-compatible.

## Resource Contracts

Shared constants in `alloygbm_core` will define conservative upper bounds for metadata bytes,
aggregate artifact bytes, model features, model stumps, classes/outputs, feature-name bytes, and
cuts per feature. Existing section-count, section-size, and per-tree node-slot limits remain in
force. Serializer, contract validator, and deserializer must enforce the same limits so AlloyGBM
cannot emit an artifact that its own loaders reject.

Declared counts must also be feasible for the bytes that remain in their containing payload.
Decoders must perform checked length arithmetic and reject impossible declarations before
`Vec::with_capacity`, iteration, or payload copying. A global maximum is not a substitute for this
local feasibility check.

## Metadata And Payload Validation

`ModelMetadata` validation will require a supported format version, a nonempty bounded feature
list, bounded individual feature names, a bounded nonempty objective label, and valid class-count
semantics. Unknown bounded objective labels remain accepted for custom and forward-compatible
objectives. Multiclass metadata, payload class count, and section kind must agree.

Counted payload decoders will reject unknown flag bits, non-finite persisted numeric state,
duplicate references, out-of-range feature/stump references, inconsistent output dimensions, and
trailing bytes. In particular, native-categorical, linear-leaf, DART, multi-output, feature
baseline, MorphBoost, and multiclass payloads will no longer silently ignore malformed overlays or
extra data.

Cross-section validation will happen after the primary tree payload establishes feature, stump,
class, and output counts but before a loaded model is returned. Both the engine and predictor must
reject malformed references consistently. Shared parsing/validation helpers will be used where
they remove duplicated wire-format logic; loader-specific assembly will remain local.

## Public API Validation

One checked dense-shape helper will validate `row_count * feature_count` and the expected element
or byte length before allocation or slicing. Prediction, quantized prediction, SHAP, and Python
bridge paths that accept explicit dimensions will use this contract.

Per-feature quantile cuts and sorted-value inputs will be bounded by the supported u16 bin domain,
finite, and strictly increasing where the algorithm requires ordered cuts. Python `load_model`
will validate the embedded native artifact during load and preserve the underlying error instead
of returning an object marked fitted with an unusable artifact.

## Compatibility And Errors

Valid artifacts produced by current and supported legacy releases must continue to load and yield
unchanged predictions. Strict and legacy-trees-only compatibility modes remain available. New
errors will use the existing serialization, validation, contract-violation, and Python exception
mapping conventions, and will identify the rejected field and applicable limit.

This PR will not delay constructor validation, reorder estimator inheritance, change fitted-state
attributes, add sparse input support, or mark sklearn checks as expected failures. Those concerns
belong to PR #131.

## Verification

Tests will be written before each implementation slice and will cover:

- oversized metadata and aggregate artifacts on both serialization and deserialization;
- tiny payloads with huge declared counts, proving rejection occurs before allocation;
- malformed metadata and every hardened counted section;
- engine/predictor parity for invalid cross-section references;
- overflowed and mismatched dense/quantized dimensions;
- unsorted, non-finite, and excessive cut arrays;
- fail-fast Python model loading;
- round-trip loading and prediction parity for scalar, multiclass, DART, PL, categorical,
  MorphBoost, and joint multi-output artifacts.

The final gate is `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`, the full Python suite, and Sphinx warnings-as-errors. No performance
benchmark is required because valid training and prediction behavior is unchanged, but rejection
tests must avoid actually allocating near the configured limits.
