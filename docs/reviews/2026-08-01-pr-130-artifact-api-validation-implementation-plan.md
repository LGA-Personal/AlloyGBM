# PR #130 Artifact and API Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-01 | OpenAI Codex | `main` after PR #129 | `1af8df3` | Approved for implementation |

**Goal:** Reject malformed or resource-amplifying model artifacts and public API dimensions before
allocation, iteration, or model use while preserving valid artifact bytes and predictions.

**Architecture:** `alloygbm_core` owns shared resource limits, checked shape arithmetic, metadata
validation, and counted-section validation. Engine, predictor, SHAP, and PyO3 loaders consume those
contracts and add only the cross-section invariants that require a fully decoded model. Python
persistence validates the native artifact before exposing a loaded estimator.

**Tech Stack:** Rust 1.92, serde/serde_json, PyO3/numpy, Python 3.13, pytest, Sphinx.

## Global Constraints

- Preserve the v1 binary wire format and valid legacy Trees-only compatibility.
- Preserve valid artifact bytes and predictions; validation errors are the only behavior change.
- Keep unknown bounded metadata fields and objective labels forward-compatible.
- Enforce every emitted artifact limit symmetrically in serializer and loader.
- Validate declared counts against both shared budgets and available payload bytes before allocation.
- Use checked arithmetic for all caller-controlled dimensions.
- Keep sklearn estimator lifecycle changes out of PR #130.
- Keep `unsafe_code = "forbid"` and introduce no unsafe Rust.

---

### Task 1: Core resource budgets and metadata contract

**Files:**
- Modify: `crates/core/src/artifact_format.rs`
- Modify: `crates/core/src/validation.rs`
- Modify: `crates/core/src/lib.rs`
- Test: `crates/core/src/tests/main.rs`

**Interfaces:**
- Produces exported `MAX_MODEL_ARTIFACT_BYTES`, `MAX_MODEL_METADATA_BYTES`,
  `MAX_MODEL_FEATURES`, `MAX_MODEL_STUMPS`, `MAX_MODEL_CLASSES`, `MAX_MODEL_OUTPUTS`,
  `MAX_MODEL_FEATURE_NAME_BYTES`, and `MAX_MODEL_OBJECTIVE_BYTES` constants.
- Produces `validate_model_metadata(metadata: &ModelMetadata) -> CoreResult<()>`.
- Produces `checked_dense_element_count(row_count, feature_count) -> CoreResult<usize>` for Task 4.

- [ ] **Step 1: Add failing metadata and top-level limit tests**

Add tests that construct metadata with zero features, too many features without allocating the
limit, an oversized feature name, an empty/oversized objective, invalid `num_classes`, and
multiclass metadata without a valid class count. Mutate the binary header to declare
`MAX_MODEL_METADATA_BYTES + 1` and build a `ModelIoContractV1` whose final section offset exceeds
`MAX_MODEL_ARTIFACT_BYTES`.

```rust
#[test]
fn model_artifact_rejects_oversized_metadata_before_json_decode() {
    let mut bytes = ModelBinaryHeader::new(1, (MAX_MODEL_METADATA_BYTES + 1) as u32)
        .encode()
        .to_vec();
    bytes.extend_from_slice(&[0; MODEL_SECTION_DESCRIPTOR_LEN]);
    let err = deserialize_model_artifact_v1(&bytes).expect_err("metadata budget must fail");
    assert!(err.to_string().contains("metadata json length"));
}
```

- [ ] **Step 2: Verify the new tests fail for missing limits**

Run: `cargo test -p alloygbm-core model_artifact_rejects_oversized_metadata_before_json_decode`

Expected: compilation failure for the missing exported constants or assertion failure because the
decoder reaches truncation handling instead of the metadata budget.

- [ ] **Step 3: Implement shared budgets and symmetric validation**

Add the constants, validate metadata in `validate_model_contract_v1`, reject oversized metadata
before slicing/copying in `deserialize_model_artifact_v1`, reject it before header construction in
`serialize_model_artifact_v1`, and reject an aggregate final offset over the artifact budget.
Use `checked_add`/`checked_mul` for every budget calculation.

- [ ] **Step 4: Run the core suite**

Run: `cargo test -p alloygbm-core`

Expected: all core tests pass.

- [ ] **Step 5: Commit the core contract**

```bash
git add crates/core/src/artifact_format.rs crates/core/src/validation.rs \
  crates/core/src/lib.rs crates/core/src/tests/main.rs
git commit -m "fix: bound model artifact metadata resources"
```

### Task 2: Counted section decoder hardening

**Files:**
- Modify: `crates/core/src/artifact_format.rs`
- Test: `crates/core/src/tests/main.rs`

**Interfaces:**
- Produces counted payload decoders that reject impossible counts before allocation and consume
  their payload exactly.
- Produces validation helpers for native-categorical and linear-leaf references that accept known
  feature/stump counts for Task 3.

- [ ] **Step 1: Add failing tiny-payload/huge-count tests**

Cover native categorical `stump_count`, linear-leaf `entry_count`, and multi-output `n_stumps` with
`u32::MAX` declarations in minimum-size payloads. Add malformed tests for unknown linear flags,
trailing bytes, non-finite DART weights, zero outputs, non-divisible multi-output leaf vectors,
non-finite feature baselines, malformed MorphBoost state, and duplicate native/linear references.

```rust
#[test]
fn native_categorical_decoder_rejects_impossible_stump_count_before_allocation() {
    let mut bytes = 0_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = decode_native_categorical_splits_payload(&bytes)
        .expect_err("declared stump headers do not fit");
    assert!(err.to_string().contains("stump count"));
}
```

- [ ] **Step 2: Verify representative tests fail**

Run: `cargo test -p alloygbm-core impossible_stump_count_before_allocation`

Expected: the current decoder attempts `Vec::with_capacity(u32::MAX)` and panics/aborts or returns a
different late truncation error; run the test in isolation.

- [ ] **Step 3: Add minimum-size feasibility checks before allocation**

For each counted decoder, derive a nonzero minimum encoded size per entry, compare the declared
count with `remaining_bytes / minimum_size`, then compare with the applicable shared maximum.
Replace unchecked `cursor + len * width` with checked arithmetic.

- [ ] **Step 4: Enforce exact payload and numeric semantics**

Reject unknown flag bits, duplicate indices, zero/oversized outputs, non-finite persisted values,
invalid scaler values, and any `cursor != bytes.len()` after decoding. Keep supported v1/v2
backward compatibility explicit.

- [ ] **Step 5: Run the core suite and commit**

Run: `cargo test -p alloygbm-core`

```bash
git add crates/core/src/artifact_format.rs crates/core/src/tests/main.rs
git commit -m "fix: validate counted artifact payloads before allocation"
```

### Task 3: Engine and predictor structural validation

**Files:**
- Modify: `crates/engine/src/artifact.rs`
- Modify: `crates/engine/src/trained_model.rs`
- Modify: `crates/engine/src/multiclass_model.rs`
- Modify: `crates/predictor/src/lib.rs`
- Test: `crates/engine/src/tests/main.rs`
- Test: `crates/predictor/src/lib.rs`

**Interfaces:**
- Consumes Task 1 limits and Task 2 reference validators.
- Produces loaders that validate feature, stump, class, output, overlay, and metadata relationships
  before returning a callable model.

- [ ] **Step 1: Add failing crafted-artifact tests**

Create valid fixture artifacts and mutate one field at a time: out-of-range stump feature,
metadata/payload class mismatch, invalid multiclass count arithmetic, out-of-range linear
regressor feature, duplicate/out-of-range categorical stump overlays, DART count mismatch, and
feature-baseline count mismatch. Assert both engine and predictor return contract errors rather
than ignoring an overlay or allowing a later indexing panic.

- [ ] **Step 2: Verify the engine panic-path test fails**

Run: `cargo test -p alloygbm-engine loaded_model_rejects_out_of_range_stump_feature`

Expected: artifact loading currently succeeds, causing the new assertion to fail.

- [ ] **Step 3: Add checked multiclass parsing and metadata agreement**

Use checked class-table and stump-size arithmetic. Require `2..=MAX_MODEL_CLASSES`, total stumps
within `MAX_MODEL_STUMPS`, exact payload size, `metadata.num_classes == payload.num_classes`, and a
multiclass objective/section pairing. Share helpers where engine and predictor currently duplicate
wire-format calculations.

- [ ] **Step 4: Validate all references before overlays are applied**

Check every stump/debug/linear/categorical feature index and every overlay stump index against the
decoded primary payload. Reject duplicates and mismatches rather than filtering or overwriting.
Require feature baseline length to equal metadata feature count.

- [ ] **Step 5: Run loader suites and commit**

Run: `cargo test -p alloygbm-engine -p alloygbm-predictor`

```bash
git add crates/engine/src/artifact.rs crates/engine/src/trained_model.rs \
  crates/engine/src/multiclass_model.rs crates/engine/src/tests/main.rs \
  crates/predictor/src/lib.rs
git commit -m "fix: reject inconsistent model artifact structure"
```

### Task 4: Checked public dense and quantized dimensions

**Files:**
- Modify: `crates/core/src/validation.rs`
- Modify: `crates/predictor/src/lib.rs`
- Modify: `crates/shap/src/lib.rs`
- Modify: `bindings/python/src/predict.rs`
- Test: `crates/core/src/tests/main.rs`
- Test: `crates/predictor/src/lib.rs`
- Test: `crates/shap/src/tests/main.rs`
- Test: `bindings/python/src/tests/main.rs`

**Interfaces:**
- Consumes `checked_dense_element_count` from Task 1.
- Produces exact element/byte-length validation before allocation or slicing on every explicit
  `(row_count, feature_count)` path.

- [ ] **Step 1: Add failing overflow and mismatch tests**

Cover `usize::MAX * 2`, a byte payload divisible by four but inconsistent with dimensions, and
zero-feature/nonzero-row inputs. For the PyO3 quantized path, factor dimension validation into a
Rust helper so tests do not need to allocate Python arrays.

```rust
#[test]
fn dense_shape_rejects_element_count_overflow() {
    let err = checked_dense_element_count(usize::MAX, 2).expect_err("shape must overflow");
    assert!(err.to_string().contains("row_count * feature_count"));
}
```

- [ ] **Step 2: Verify the shared-helper test fails to compile**

Run: `cargo test -p alloygbm-core dense_shape_rejects_element_count_overflow`

Expected: compilation failure until `checked_dense_element_count` is implemented/exported.

- [ ] **Step 3: Route explicit dimensions through checked helpers**

Validate element count, convert to bytes with checked multiplication, and require exact input
length before output or scratch allocation. Apply this to core matrix views, predictor dense/bytes
methods, SHAP dense methods, and both quantized Python prediction modes.

- [ ] **Step 4: Run affected Rust suites and commit**

Run: `cargo test -p alloygbm-core -p alloygbm-predictor -p alloygbm-shap -p alloygbm-python`

```bash
git add crates/core/src/validation.rs crates/core/src/tests/main.rs \
  crates/predictor/src/lib.rs crates/shap/src/lib.rs crates/shap/src/tests/main.rs \
  bindings/python/src/predict.rs bindings/python/src/tests/main.rs
git commit -m "fix: check public dense dimensions before allocation"
```

### Task 5: SHAP binning and fail-fast Python persistence

**Files:**
- Modify: `crates/shap/src/binning.rs`
- Modify: `crates/shap/src/tests/main.rs`
- Modify: `bindings/python/alloygbm/_regressor/_persistence.py`
- Modify: `bindings/python/tests/test_native_runtime_integration.py`

**Interfaces:**
- Consumes `MAX_MODEL_CUTS_PER_FEATURE` or the existing u16 data-bin bound exported from core.
- Produces validated finite, strictly increasing cuts/sorted values and `load_model()` that proves
  its embedded artifact is usable before returning.

- [ ] **Step 1: Add failing SHAP binning tests**

Construct `BinningContext` inputs with NaN/Inf cuts, duplicate/descending cuts, and one more cut than
the supported u16 threshold domain. Assert construction/validation returns a descriptive error.

- [ ] **Step 2: Add a failing corrupt-container load test**

Train and save a valid model, replace the container's artifact bytes with a malformed AGBM payload
while preserving the Python envelope, then assert `GBMRegressor.load_model(path)` raises the native
artifact validation error during load rather than returning a fitted object.

- [ ] **Step 3: Verify both tests fail for current late validation**

Run: `cargo test -p alloygbm-shap binning_context_rejects_non_finite_cuts`

Run: `PYTHONPATH=bindings/python .venv/bin/python -m pytest \
bindings/python/tests/test_native_runtime_integration.py -k corrupt_artifact -q`

Expected: SHAP accepts malformed inner cuts and Python load returns an estimator or suppresses the
native handle error.

- [ ] **Step 4: Implement boundary validation**

Validate all cut/sorted-value vectors before explanation work. In `_restore_state`, require native
handle construction for fitted native artifacts and re-raise the original validation exception;
retain fallback behavior only for explicitly supported legacy states.

- [ ] **Step 5: Rebuild the extension, run targeted Python/Rust tests, and commit**

Run: `maturin develop --release`

Run: `cargo test -p alloygbm-shap`

Run: `.venv/bin/python -m pytest bindings/python/tests/test_native_runtime_integration.py -q`

```bash
git add crates/shap/src/binning.rs crates/shap/src/tests/main.rs \
  bindings/python/alloygbm/_regressor/_persistence.py \
  bindings/python/tests/test_native_runtime_integration.py
git commit -m "fix: validate SHAP cuts and loaded model artifacts"
```

### Task 6: Review resolution, compatibility matrix, and final gate

**Files:**
- Modify: `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md`
- Modify: `docs/reviews/2026-08-01-pr-130-artifact-api-validation-design.md`
- Modify: `docs/reviews/2026-08-01-pr-130-artifact-api-validation-implementation-plan.md`
- Modify if user-facing limits need disclosure: `docs/limitations.md`

**Interfaces:**
- Produces traceable closure evidence for core review finding 5.2 and records PR #131 as the owner
  of sklearn conformance.

- [ ] **Step 1: Run valid artifact compatibility tests before documentation claims**

Run targeted round trips for scalar, multiclass, DART, PL, native categorical, MorphBoost, and
joint multi-output models. Confirm artifact bytes are unchanged for deterministic fixtures and
predictions match before/after loading.

- [ ] **Step 2: Update the core resolutions document**

Replace “Additional artifact hardening remains open” with the limits, pre-allocation checks,
cross-section invariants, checked API dimensions, fail-fast load behavior, and exact regression
test names. Keep sklearn conformance explicitly open for PR #131.

- [ ] **Step 3: Run formatting and static analysis**

Run: `cargo fmt --check`

Run: `cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 4: Run the complete verification matrix**

Run: `cargo test --workspace`

Run: `.venv/bin/python -m pytest bindings/python/tests/ -q`

Run: `.venv/bin/python -m pytest benchmarks/tests/ -q`

Run: `.venv/bin/python -m sphinx -W -b html docs/site/source docs/site/_build/html`

Expected: every command succeeds with no warnings promoted to errors.

- [ ] **Step 5: Commit resolution evidence**

```bash
git add docs/reviews/2026-07-02-v0.12.10-core-resolutions.md \
  docs/reviews/2026-08-01-pr-130-artifact-api-validation-design.md \
  docs/reviews/2026-08-01-pr-130-artifact-api-validation-implementation-plan.md \
  docs/limitations.md
git commit -m "docs: close artifact and API validation review findings"
```
