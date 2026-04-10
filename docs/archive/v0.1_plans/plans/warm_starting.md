# Plan: Warm-Starting / Incremental Training

## Status: Not Started

## Summary

There is no way to continue training from a previously fitted model. Each call to `fit()` starts fresh -- no `init_model` / `keep_training_booster` equivalent. This means users can't:
- Add more boosting rounds to an existing model
- Fine-tune a model on new data
- Implement staged training workflows

---

## Questions to Resolve Before Starting

1. **API approach**: Options:
   - `model.fit(X, y, init_model=previous_model)` -- pass an existing model to continue from
   - `model.fit(X, y, warm_start=True)` -- use the model's own previous state (sklearn convention with `warm_start` attribute)
   - `model.update(X, y, n_more_rounds=50)` -- explicit continuation method
   Recommendation: Both `warm_start=True` attribute (sklearn-style) and `init_model` parameter (LightGBM-style). `warm_start=True` means `fit()` continues from self; `init_model` means continue from a different model.

2. **Compatibility constraints**: When continuing training, must the new data have the same features? Same number of features and same binning? Recommendation: strict -- same feature count, same binning thresholds. Different data rows are fine.

3. **Prediction state**: The training loop maintains `candidate_predictions` (current ensemble predictions). To continue training, we need these initial predictions. They can be recomputed by running the existing trees on the training data, or stored from the previous training run.

---

## Implementation

### Phase 1: Engine Support

**`crates/engine/src/lib.rs`**

#### Step 1.1: Accept initial model in training functions

The key is the `fit_iterations_with_optional_validation_summary` function (line ~1110). Currently it starts with `initial_prediction` from the objective (typically the mean of targets for MSE). For warm-starting:

```rust
pub fn fit_iterations_warm(
    &self,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    backend: &B,
    objective: &O,
    controls: IterationControls,
    initial_model: &TrainedModel,  // existing trees
) -> EngineResult<IterationRunSummary>
```

#### Step 1.2: Compute initial predictions from existing model

Before the boosting loop starts, run the existing model's trees on the training data to get `candidate_predictions`:

```rust
let predictor = Predictor::from_artifact_bytes(&initial_model.to_artifact_bytes()?)?;
let mut candidate_predictions = vec![0.0_f32; row_count];
for i in 0..row_count {
    candidate_predictions[i] = predictor.predict_row(...);
}
```

Or more efficiently, use `predict_batch_dense` if available at the engine level.

Alternative: the `TrainedModel` already has `stumps` (the tree structures). Traverse them directly without going through the artifact serialization path.

#### Step 1.3: Append new trees to existing model

The training loop produces new `TreeStump` entries. For warm-starting, prepend the existing model's stumps and adjust round indexing:

```rust
let mut all_stumps = initial_model.stumps.clone();
// ... train new rounds ...
all_stumps.extend(new_stumps);
```

The `initial_prediction` for the ensemble should come from the initial model, not recomputed from the objective.

### Phase 2: Bridge Support

**`bindings/python/src/lib.rs`**

Add `init_artifact_bytes: Option<&[u8]>` to training functions. If provided, deserialize the artifact, extract the model, and pass to the engine's warm-start path.

### Phase 3: Python API

**`bindings/python/alloygbm/regressor.py`**

#### Step 3.1: Add `warm_start` attribute

```python
GBMRegressor.__init__(
    # ...
    warm_start: bool = False,
)
```

When `warm_start=True` and the model is already fitted, `fit()` continues from the existing model instead of starting fresh.

#### Step 3.2: Add `init_model` parameter to `fit()`

```python
def fit(self, X, y, *, init_model: 'GBMRegressor | None' = None, ...):
```

If `init_model` is provided, use its artifact bytes as the starting point.

#### Step 3.3: Track round counts

After warm-starting, `n_estimators` refers to the *additional* rounds to train. The total rounds in the model is `previous_rounds + n_estimators`.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Engine | 1 (`engine/src/lib.rs`) | ~80-120 | Medium-High |
| Phase 2: Bridge | 1 (`bindings/python/src/lib.rs`) | ~30-50 | Medium |
| Phase 3: Python API | 1 (`regressor.py`) | ~40-60 | Low |

Total: ~150-230 lines across 3 files. The engine work is the most complex -- computing initial predictions efficiently and merging tree structures.

---

## Risk Areas

### Binning Compatibility

The existing model was trained with specific bin thresholds. The new training data must use the **same** bin thresholds for the trees to be consistent. If the user provides different data, the continuous features must be binned using the original thresholds, not recomputed.

This means the warm-start path needs access to the original binning metadata (`continuous_feature_mins`, `continuous_feature_maxs`, `continuous_feature_sorted_values`). The Python `GBMRegressor` stores these after `fit()`, so they're available for `warm_start=True`. For `init_model`, the binning metadata must be transferable.

### Round Indexing

The tree node IDs encode the round index (`encode_tree_node_id`). When appending new trees, the round index must continue from where the previous model left off, not restart from 0. Verify that the predictor correctly handles non-zero-based round indices (it should, since it processes trees sequentially).

### Initial Prediction Consistency

The `initial_prediction` (bias term) must be the same as the original model's. Don't recompute from the new data's target mean -- use the stored `initial_prediction` from the existing model.

---

## Testing Strategy

1. **Warm-start equivalence**: Training 100 rounds at once vs. 50 + warm-start 50 should produce identical results (same data, same seed)
2. **init_model**: Train model A, use as init_model for model B, verify B has A's trees plus new ones
3. **Binning consistency**: Verify that warm-started model uses original bin thresholds
4. **Prediction**: Predictions from warm-started model are correct (not double-counting initial bias)
5. **Round count**: `n_estimators_` reflects total rounds (previous + new)

---

## Non-Goals

- **Transfer learning**: Training on different features or different feature distributions
- **Model distillation**: Using one model's predictions as soft targets for another
- **Online learning**: Adding single samples incrementally (this is batch warm-starting)
