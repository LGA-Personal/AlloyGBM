# Plan: Sample Weight Support from Python

## Status: Not Started

## Summary

The Rust engine's `TrainingDataset` already has `sample_weights: Option<Vec<f32>>` and the `SquaredErrorObjective` correctly handles weights in gradient/hessian computation (`grad = residual * weight`, `hess = weight`). However, the Python bridge **always passes `sample_weights: None`** -- the `fit()` method has no `sample_weight` parameter. This plan exposes the existing Rust-side capability to Python.

---

## Questions to Resolve Before Starting

1. **Parameter naming**: `sample_weight` (sklearn convention) vs `sample_weights` (plural, matches Rust field)? Recommendation: `sample_weight` (singular) to match sklearn's `fit(X, y, sample_weight=...)` convention.

2. **Validation set weights**: Should the validation set also accept sample weights? This affects early stopping loss computation. Recommendation: yes, accept `eval_sample_weight` alongside `eval_set`. Weighted loss on validation should use the same weighting.

3. **Weight normalization**: Should weights be normalized (sum to N) or used as-is? XGBoost/LightGBM use raw weights. The Rust engine uses raw weights. Recommendation: use raw weights, document that they affect the scale of gradients/hessians and therefore regularization sensitivity.

---

## Architecture Overview

### Current State

**Rust engine** (`engine/src/lib.rs`):
- `SquaredErrorObjective::compute_gradients()` (line ~168): `grad = (prediction - target) * weight`, `hess = weight`
- `TrainingDataset` (core/src/lib.rs): `sample_weights: Option<Vec<f32>>`
- `squared_error_loss()` (engine/src/lib.rs ~2694): Does NOT use weights -- this computes unweighted MSE for monitoring

**Python bridge** (`bindings/python/src/lib.rs`):
- All training functions construct `TrainingDataset` with `sample_weights: None`
- No parameter for weights exists on any pyfunction

**Python API** (`regressor.py`):
- `fit()` has no `sample_weight` parameter

### What Needs to Change

1. Add `sample_weight` parameter to Python `fit()`
2. Pass weights through the bridge to the engine
3. Optionally: update loss monitoring to use weighted loss

---

## Phase 1: Python API

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 1.1: Add `sample_weight` to `fit()`

```python
def fit(
    self,
    X,
    y,
    *,
    sample_weight: object | None = None,  # NEW
    eval_set: tuple | None = None,
    eval_sample_weight: object | None = None,  # NEW
    # ... existing params ...
):
```

#### Step 1.2: Validate weights

```python
if sample_weight is not None:
    sample_weight = self._validate_sample_weight(sample_weight, n_rows)
```

Validation:
- Must be array-like of floats, length matching `n_rows`
- All values must be finite and non-negative (or strictly positive, depending on convention)
- Convert to `list[float]` or numpy array for passing to Rust

#### Step 1.3: Pass weights through to native training call

In the `_fit_with_*` code paths, add `sample_weight` to the arguments passed to the native bridge functions.

---

## Phase 2: Python Bridge

### Files to Modify

**`bindings/python/src/lib.rs`**

#### Step 2.1: Add `sample_weights` parameter to all 5 training pyfunctions

Each training pyfunction currently constructs `TrainingDataset` with `sample_weights: None`. Add `sample_weights: Option<Vec<f32>>` parameter.

For the key implementation function `train_regression_artifact_with_summary_dense_impl` (line ~1332):

```rust
fn train_regression_artifact_with_summary_dense_impl(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    sample_weights: Option<Vec<f32>>,  // NEW
    // ... rest of params ...
```

#### Step 2.2: Pass weights into `TrainingDataset` construction

In `prepare_training_matrices_from_dense_values` or wherever the `TrainingDataset` is built, pass the weights through:

```rust
let dataset = TrainingDataset {
    matrix: ...,
    targets: ...,
    sample_weights,  // was: sample_weights: None
    time_index: ...,
    group_id: ...,
};
```

#### Step 2.3: Validation set weights

Similarly, add `validation_sample_weights: Option<Vec<f32>>` for the validation dataset construction.

---

## Phase 3: Loss Monitoring (Optional but Recommended)

### Current Issue

`squared_error_loss()` (engine/src/lib.rs:2694) computes unweighted MSE:
```rust
loss += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
// ... divided by N
```

When sample weights are used, the training loss reported during fitting doesn't reflect the weighted objective being optimized. The early stopping criterion compares unweighted validation loss.

### Recommended Fix

Add a `weighted_squared_error_loss()` function or modify `squared_error_loss()` to optionally accept weights:

```rust
fn squared_error_loss(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    // If weights provided: sum(w_i * (pred_i - target_i)^2) / sum(w_i)
    // If no weights: sum((pred_i - target_i)^2) / N
}
```

This ensures:
- `train_rmse` in the training summary reflects the weighted loss
- Early stopping uses weighted validation loss (consistent with the weighted gradients)

### Files to Modify

**`crates/engine/src/lib.rs`**: Update `squared_error_loss()` and all 6 call sites where it's used.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Python API | 1 (`regressor.py`) | ~30-40 | Very Low |
| Phase 2: Bridge | 1 (`bindings/python/src/lib.rs`) | ~40-60 | Low |
| Phase 3: Loss monitoring | 1 (`engine/src/lib.rs`) | ~30-50 | Low-Medium |

Total: ~100-150 lines across 3 files. The Rust engine **already supports weights** -- the work is purely plumbing.

---

## Testing Strategy

1. **Basic weight support**: Train with uniform weights (all 1.0), verify identical results to no weights
2. **Non-uniform weights**: Train with weights emphasizing certain samples, verify predictions shift toward heavily-weighted samples
3. **Zero weights**: Samples with weight 0.0 should be effectively ignored
4. **Validation set weights**: Verify early stopping uses weighted validation loss
5. **Edge cases**: All-zero weights (should error), negative weights (should error), NaN weights (should error)

---

## Non-Goals

- **Automatic class weight computation** (`class_weight='balanced'`): This is a classification concern (Limitation #1)
- **Boosting-specific weight schemes** (AdaBoost-style reweighting): AlloyGBM uses gradient boosting, not AdaBoost
- **Per-round weight updates**: Weights are fixed for the entire training run
