# Plan: Multiple Categorical Column Support

## Status: Not Started

## Summary

AlloyGBM currently supports at most one categorical feature via target encoding. The `GBMRegressor` constructor accepts `categorical_feature_index: int | None` (singular), `CategoricalTargetEncodingSpec` carries a single `feature_index: usize`, and the engine function `apply_single_categorical_target_encoding` processes exactly one column. However, the artifact format's `CategoricalStatePayloadV1` already stores `categorical_feature_indices: Vec<u32>` (plural), and `DatasetSchema` has `categorical_feature_indices: Vec<usize>` -- so multi-categorical was clearly envisioned at the data layer.

This plan covers extending the entire pipeline to support N categorical features with independent target encoding configurations.

---

## Questions to Resolve Before Starting

1. **Shared vs. per-feature encoding config**: Should all categorical features share the same `smoothing`, `min_samples_leaf`, and `time_aware` settings? Or should each feature have its own config? Recommendation: shared config initially (simpler API, matches current params), with per-feature override as a future extension.

2. **API shape for specifying categoricals**: Options:
   - `categorical_feature_indices: list[int] | None` (plural) -- simple, explicit
   - `categorical_features: dict[int, dict] | None` -- per-feature config, more complex
   - Keep auto-inference from pandas `category` dtype (already works for single, extend to multi)
   Recommendation: `categorical_feature_indices: list[int] | None` replacing the current singular parameter, with backward compatibility for `categorical_feature_index` (deprecated).

3. **How to pass categorical values**: Currently `categorical_feature_values: list[str]` is a flat list for one column. For N columns, options:
   - `categorical_feature_values: dict[int, list[str]]` -- keyed by feature index
   - `categorical_feature_values: list[list[str]]` -- parallel to `categorical_feature_indices`
   Recommendation: `dict[int, list[str]]` keyed by feature index -- more explicit and less error-prone than positional.

4. **Backward compatibility**: Should `categorical_feature_index` (singular) still work? Recommendation: yes, as a convenience that converts to a single-element list internally. Deprecation warning optional.

5. **Encoding order independence**: When encoding multiple categoricals, does encoding order matter? With target encoding it shouldn't (each feature is encoded independently against the target). But with time-aware encoding, the implementation processes rows chronologically -- confirm that encoding feature A doesn't affect encoding feature B.

---

## Architecture Overview

### Current Single-Categorical Flow

```
Python: categorical_feature_index (int) + categorical_feature_values (list[str])
  -> resolve_categorical_spec() -> Option<CategoricalTargetEncodingSpec>
  -> train_regression_artifact_with_summary_dense_impl()
    -> apply_categorical_encoding_to_training_matrices() [one column]
    -> apply_categorical_encoding_to_validation_matrices() [one column]
    -> model.with_categorical_state(CategoricalStatePayloadV1 { indices: vec![idx] })
```

### Target Multi-Categorical Flow

```
Python: categorical_feature_indices (list[int]) + categorical_feature_values (dict[int, list[str]])
  -> resolve_categorical_specs() -> Vec<CategoricalTargetEncodingSpec>
  -> train_regression_artifact_with_summary_dense_impl()
    -> for each spec: apply_categorical_encoding_to_training_matrices() [iterative]
    -> for each spec: apply_categorical_encoding_to_validation_matrices() [iterative]
    -> model.with_categorical_state(CategoricalStatePayloadV1 { indices: vec![idx0, idx1, ...] })
```

### What Already Supports Multiple Categoricals (No Changes Needed)

- **`CategoricalStatePayloadV1`** (`core/src/lib.rs:554`): Already stores `categorical_feature_indices: Vec<u32>`. Serialization/deserialization already handles vectors of indices.
- **`validate_categorical_state_payload_v1()`** (`core/src/lib.rs:722`): Already validates strictly increasing ordering for multiple indices.
- **`DatasetSchema`** (`core/src/lib.rs:70`): Already has `categorical_feature_indices: Vec<usize>`.
- **`validate_dataset_schema()`** (`core/src/lib.rs:864`): Already validates multiple categorical indices are strictly increasing and in-bounds.
- **The `categorical` crate**: All functions (`fit_target_encoder`, `transform_target_encoder`, `fit_transform_target_encoder`) operate on a single feature's values -- they're stateless and can be called independently per feature.

### What Needs Changes

Listed by layer, bottom-up.

---

## Phase 1: Engine -- Multi-Categorical Encoding

### Files to Modify

**`crates/engine/src/lib.rs`**

#### Step 1.1: Replace `CategoricalTargetEncodingSpec` with a multi-spec approach

Current struct (line ~294):
```rust
pub struct CategoricalTargetEncodingSpec {
    pub feature_index: usize,
    pub values: Vec<String>,
    pub config: TargetEncoderConfig,
}
```

Option A (recommended): Keep the struct as-is but accept `Vec<CategoricalTargetEncodingSpec>` everywhere instead of a single spec. Each spec carries its own feature_index and values.

Option B: Create a new `CategoricalEncodingPlan` struct that bundles multiple specs. More encapsulated but adds a new type.

Recommendation: Option A -- minimal new types, each function just iterates over specs.

#### Step 1.2: Generalize `apply_single_categorical_target_encoding` (line ~1722)

Rename to `apply_categorical_target_encoding` and make it accept `&[CategoricalTargetEncodingSpec]`. For each spec in the slice:
1. Validate feature_index is in-bounds and values length matches row_count
2. Call `fit_transform_target_encoder` for that feature's values
3. Replace the corresponding column in both `encoded_dense_values` and `encoded_bins_payload`

Key detail: Each encoding replaces one column in the dense matrix. The columns are independent, so order doesn't matter. But the BinnedMatrix's `max_bin` must be updated to the maximum across all encoded features.

```rust
fn apply_categorical_target_encoding(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    specs: &[CategoricalTargetEncodingSpec],
) -> EngineResult<(TrainingDataset, BinnedMatrix)> {
    // Clone once, then mutate for each spec
    let mut encoded_dense_values = dataset.matrix.values.clone();
    let mut encoded_bins_payload = binned_matrix.bins.clone();
    let mut max_bin = binned_matrix.max_bin;

    for spec in specs {
        // validate bounds, encode, update column in-place
    }
    // Build final TrainingDataset + BinnedMatrix
}
```

#### Step 1.3: Update `fit_iterations_with_single_target_encoded_feature` family

Three functions reference "single" (lines ~804, ~886, ~911):
- `fit_iterations_with_single_target_encoded_feature`
- `fit_iterations_with_single_target_encoded_feature_summary`
- `fit_iterations_with_single_target_encoded_feature_and_policy_request`

Rename to `fit_iterations_with_target_encoded_features` (plural). Change spec parameter from `spec: &CategoricalTargetEncodingSpec` to `specs: &[CategoricalTargetEncodingSpec]`.

Update `categorical_state` construction:
```rust
let categorical_state = CategoricalStatePayloadV1 {
    format_version: CATEGORICAL_STATE_FORMAT_V1,
    leakage_safe_target_encoding: specs.iter().any(|s| s.config.time_aware),
    categorical_feature_indices: specs.iter().map(|s| s.feature_index as u32).collect(),
};
```

#### Step 1.4: Validation

Add validation that:
- All `feature_index` values are unique across specs
- All `feature_index` values are within `[0, feature_count)`
- Specs are sorted by feature_index (or sort them internally) to match `CategoricalStatePayloadV1`'s strictly-increasing requirement

### Success Criteria (Phase 1)

- Existing single-categorical engine tests pass unchanged (pass a 1-element slice)
- New test: 2+ categorical features, each encoded independently, artifact roundtrips correctly
- `CategoricalStatePayloadV1` stores all categorical indices
- Encoded dataset dimensions unchanged (same row_count, same feature_count)

---

## Phase 2: Python Bridge -- Multi-Categorical Arguments

### Files to Modify

**`bindings/python/src/lib.rs`**

#### Step 2.1: Update `resolve_categorical_spec` -> `resolve_categorical_specs`

Current signature (line ~1082):
```rust
fn resolve_categorical_spec(
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    row_count: usize,
) -> Result<Option<CategoricalTargetEncodingSpec>, EngineError>
```

New signature:
```rust
fn resolve_categorical_specs(
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values: Option<HashMap<usize, Vec<String>>>,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    row_count: usize,
) -> Result<Vec<CategoricalTargetEncodingSpec>, EngineError>
```

Return `Vec` (empty if no categoricals) instead of `Option<single>`.

#### Step 2.2: Update all 5 training pyfunctions

Each of the 5 training pyfunctions (`train_regression_artifact`, `train_regression_artifact_dense`, `train_regression_artifact_with_summary`, `train_regression_artifact_dense_with_summary`, `train_regression_artifact_dense_with_summary_bytes`) currently accepts:
- `categorical_feature_index: Option<usize>`
- `categorical_feature_values: Option<Vec<String>>`

Change to:
- `categorical_feature_indices: Option<Vec<usize>>`
- `categorical_feature_values: Option<HashMap<usize, Vec<String>>>`

Also support backward compatibility by accepting the old singular form and converting internally.

#### Step 2.3: Update `train_regression_artifact_with_summary_dense_impl`

Current (line ~1344):
```rust
categorical_spec: Option<CategoricalTargetEncodingSpec>,
validation_categorical_values: Option<Vec<String>>,
```

Change to:
```rust
categorical_specs: Vec<CategoricalTargetEncodingSpec>,
validation_categorical_values: Option<HashMap<usize, Vec<String>>>,
```

Update the body:
- `need_dense_values = !categorical_specs.is_empty()`
- Loop over specs for training encoding
- Loop over specs for validation encoding
- Collect all `CategoricalStatePayloadV1` indices

#### Step 2.4: Update `apply_categorical_encoding_to_training_matrices` and `apply_categorical_encoding_to_validation_matrices`

These bridge-level functions (lines ~1181, ~1244) currently handle a single spec. Generalize to accept `&[CategoricalTargetEncodingSpec]` or call per-spec in a loop.

### Success Criteria (Phase 2)

- Bridge tests pass with 0, 1, and 2+ categorical specs
- Old single-categorical Python calls still work (backward compat)
- `train_bridge_categorical_path_matches_engine_predictions` test updated for multi-categorical

---

## Phase 3: Python API -- Multi-Categorical Interface

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 3.1: Update constructor parameters

Current:
```python
categorical_feature_index: int | None = None,  # singular
```

New:
```python
categorical_feature_indices: list[int] | None = None,  # plural
```

Keep `categorical_feature_index` as a deprecated alias that converts `int -> [int]` internally.

Validation:
- All indices must be non-negative integers
- No duplicates
- If both `categorical_feature_index` and `categorical_feature_indices` are provided, raise ValueError

#### Step 3.2: Update `categorical_feature_values` parameter in `fit()`

Current `fit()` signature:
```python
def fit(self, X, y, *, categorical_feature_values: object | None = None, ...)
```

Change the semantics:
- If `categorical_feature_indices` has 1 element: `categorical_feature_values` can still be `list[str]` (backward compat) or `dict[int, list[str]]`
- If `categorical_feature_indices` has N>1 elements: `categorical_feature_values` must be `dict[int, list[str]]`
- Auto-inference from pandas `category` dtypes should detect all categorical columns, not just the first one

#### Step 3.3: Update `_infer_explicit_categorical_feature` (line ~1922)

Currently raises an error if multiple categorical columns are detected:
```python
if len(categorical_indices) > 1:
    raise ValueError("X contains multiple explicit categorical columns; set categorical_feature_index explicitly")
```

Change to: infer all categorical columns and return `dict[int, list[str]]`.

#### Step 3.4: Update `_extract_categorical_values_for_index` (line ~1890)

Generalize to extract values for multiple indices. Could become `_extract_categorical_values_for_indices` returning `dict[int, list[str]]`.

#### Step 3.5: Update `get_params()` / `set_params()` / `__repr__`

Add `categorical_feature_indices` to the param list. Handle backward compat for `categorical_feature_index`.

#### Step 3.6: Update validation data handling

Currently (lines ~638-641):
```python
if effective_categorical_feature_index is not None:
    validation_categorical_values = self._extract_categorical_values_for_index(
        eval_X, effective_categorical_feature_index, ...)
```

Generalize to extract values for all categorical indices in the validation set.

### Success Criteria (Phase 3)

- `GBMRegressor(categorical_feature_index=3)` still works (deprecated but functional)
- `GBMRegressor(categorical_feature_indices=[1, 3, 5])` works with `fit(X, y, categorical_feature_values={1: [...], 3: [...], 5: [...]})`
- Auto-inference from pandas DataFrames with multiple `category` columns works
- `get_params()` returns `categorical_feature_indices` as a list
- Validation set categorical values are correctly extracted for all indices

---

## Phase 4: Prediction Path Updates

### Files to Modify

The prediction path should require **no changes** for multi-categorical support. Here's why:

- Categorical encoding happens at training time: the categorical column's raw string values are replaced with their target-encoded float values, which are then binned like any other continuous feature
- The `BinnedMatrix` and the trained trees have no concept of "categorical" -- they just see bins
- At prediction time, the user must provide float values for all features (the target-encoded values are what the model was trained on)
- The `CategoricalStatePayloadV1` in the artifact records *which* features were categorically encoded, but the predictor doesn't use this -- it's metadata for the caller

**However**, if the Python `predict()` method needs to accept raw categorical values and encode them at prediction time, that's a separate feature (online categorical encoding). Currently, users must pre-encode categorical features before calling `predict()`. This limitation exists today and is orthogonal to multi-categorical support.

### Verify

- Prediction produces identical results whether data had 1 or N categorical features
- No prediction code references `CategoricalStatePayloadV1`

---

## Phase 5: Artifact Format Compatibility

### Already Supported

`CategoricalStatePayloadV1` already serializes a `Vec<u32>` of feature indices. The format works for any number of indices:

```rust
pub struct CategoricalStatePayloadV1 {
    pub format_version: u32,
    pub leakage_safe_target_encoding: bool,
    pub categorical_feature_indices: Vec<u32>,  // already a vector
}
```

The `encode_categorical_state_payload_v1` and `decode_categorical_state_payload_v1` functions already serialize/deserialize arbitrary-length vectors. The `validate_categorical_state_payload_v1` function already validates that indices are strictly increasing.

### Backward Compatibility

- Old artifacts with 1 categorical index: load fine (1-element vector)
- Old artifacts with 0 categorical indices (no categorical section): load fine (section is optional)
- New artifacts with N>1 categorical indices: old code would also load them fine since the format already supports it

**No artifact format changes needed.**

---

## Implementation Order

1. **Phase 1** (Engine): Generalize encoding functions to accept `&[CategoricalTargetEncodingSpec]`
2. **Phase 2** (Bridge): Update PyO3 functions to pass vectors of specs
3. **Phase 3** (Python API): Update `GBMRegressor` parameter interface
4. **Phase 4** (Verify): Confirm prediction path needs no changes
5. **Phase 5** (Verify): Confirm artifact format needs no changes

Phases 1-2 are the core work. Phase 3 is the user-facing API change. Phases 4-5 are verification only.

---

## Risk Areas and Troubleshooting

### Performance with Many Categoricals

Each categorical feature requires:
- A full pass through `fit_transform_target_encoder` (O(N) per feature)
- Replacing one column in the dense matrix and binned matrix
- If time-aware: sorting rows by time index and processing chronologically

With K categorical features, this is O(K * N) total. For K < 20 and N < 10M this should be negligible relative to tree building time.

**Optimization**: Instead of cloning the full dense matrix once and then modifying columns one at a time, could modify columns in-place on the same clone. The current code already does this -- just need to ensure the multi-feature loop doesn't re-clone.

### Column Encoding Independence

Target encoding for feature A uses only the target values (y), not any other feature's values. So encoding order doesn't matter and features can be encoded independently. **However**, time-aware encoding for feature A could theoretically interact with feature B if they share the same time index grouping -- but since each feature's encoding only looks at its own (value, target) pairs, they're truly independent.

### Cardinality Overflow

Each target-encoded feature produces a set of unique encoded float values, which are mapped to u8 bins (max 256). If a categorical feature has more than 256 unique categories, the encoded values *might* fit in 256 bins (multiple categories could share the same encoded value due to smoothing), but if not, `encode_bins_from_encoded_values` will return an error. This is a pre-existing limitation (see Limitation #7: Bins Capped at 256) and is not worsened by multi-categorical support.

### Validation Set Encoding

For the validation set, target encoding uses the training set's fitted encoder state (via `transform_target_encoder`, not `fit_transform_target_encoder`). This correctly prevents target leakage. With multiple categoricals, each validation column must be transformed using the corresponding training encoder state. The implementation must pair each validation feature's values with the correct encoder.

**Implementation detail**: In `fit_transform_target_encoder`, the `TargetEncoderState` is returned alongside the encoded values. For multi-categorical, store each feature's encoder state in a `Vec<(usize, TargetEncoderState)>` (keyed by feature_index) and use the corresponding state for validation encoding.

Currently, the bridge function `apply_categorical_encoding_to_validation_matrices` (line ~1244) fits a *new* encoder on the training data and then transforms the validation data:
```rust
let (encoder_state, _) = fit_transform_target_encoder(
    &categorical_spec.config,
    &categorical_spec.values,
    training_targets,
    training_time_index,
)?;
let encoded_values = transform_target_encoder(&encoder_state, &categorical_spec.values)?;
```

This approach works but re-fits the encoder for validation. A cleaner approach would be to pass the already-fitted encoder state from training. Consider refactoring to return encoder states from the training encoding step and reuse them for validation.

---

## Testing Strategy

### Unit Tests (Engine)

1. **Multi-categorical encoding roundtrip**: Create dataset with 2 string columns, encode both, verify binned matrix has correct encoded values in both columns while other columns are unchanged
2. **Order independence**: Encode features [1, 3] vs [3, 1] and verify identical results
3. **Validation encoding**: Verify validation set uses training encoder state for both features
4. **Artifact roundtrip**: Train with 2 categoricals, serialize/deserialize, verify `categorical_feature_indices` has both indices
5. **Edge cases**: 0 specs (no-op), 1 spec (backward compat), all features categorical

### Integration Tests (Bridge)

1. **Bridge consistency**: Train via engine with 2 categoricals, train via bridge with same data, verify predictions match
2. **Backward compat**: Old-style single `categorical_feature_index` still works through bridge

### Python Tests

1. **Constructor**: `GBMRegressor(categorical_feature_indices=[1, 3])` sets params correctly
2. **Deprecated param**: `GBMRegressor(categorical_feature_index=3)` converts to `[3]`
3. **Fit with dict values**: `fit(X, y, categorical_feature_values={1: [...], 3: [...]})`
4. **Auto-inference**: DataFrame with 2 `category` columns infers both
5. **Eval set**: Validation categorical values extracted for all indices
6. **get_params/set_params**: Roundtrip with new param names

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed (est.) | Risk |
|-------|--------------|---------------------|------|
| Phase 1: Engine | 1 (`engine/src/lib.rs`) | ~80-120 | Low -- mostly loop generalization |
| Phase 2: Bridge | 1 (`bindings/python/src/lib.rs`) | ~100-150 | Medium -- many functions to update |
| Phase 3: Python API | 1 (`regressor.py`) | ~120-180 | Medium -- backward compat logic |
| Phase 4: Prediction | 0 | 0 (verify only) | None |
| Phase 5: Artifact | 0 | 0 (verify only) | None |

Total: ~300-450 lines changed across 3 files. No new crates or modules needed.

---

## Non-Goals (Out of Scope)

- **Native categorical splits** (like LightGBM's optimal histogram split): This would be a fundamentally different approach where the tree builder natively handles categorical features without pre-encoding. Much more complex and a separate initiative.
- **Per-feature encoding config**: Each feature using different smoothing/min_samples_leaf. Could be added later but overcomplicates the initial API.
- **Online prediction-time encoding**: Having `predict()` accept raw string values and encode them. Currently users must pre-encode; this is orthogonal to multi-categorical.
- **Frequency encoding from Python**: The `categorical` crate supports frequency encoding, but it's not exposed through the Python API. Separate concern.
- **Ordinal encoding**: Another encoding strategy not currently exposed.
