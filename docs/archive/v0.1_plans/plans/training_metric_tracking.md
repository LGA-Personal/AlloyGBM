# Plan: Flexible Training Metric Tracking

## Status: Not Started

## Summary

The training loop only tracks RMSE via `squared_error_loss()`. The `evals_result_` only contains RMSE per round. There's no callback system, no custom evaluation metric during training, and no way to track MAE or other metrics alongside RMSE. This limits observability into the training process.

---

## Questions to Resolve Before Starting

1. **Which metrics to add**: Minimum set for regression: RMSE (existing), MAE, R². For classification (future): log_loss, accuracy, AUC. Recommendation: start with a configurable set of named metrics.

2. **Custom metric callbacks**: Should users be able to pass Python callables as evaluation metrics? This would be called once per round with (predictions, targets) and return a float. Significant performance concern: crossing the Rust-Python boundary per round is expensive. Recommendation: support named built-in metrics first, defer custom callbacks.

3. **API shape**: How does the user specify which metrics to track?
   - `eval_metric: str | list[str] = "rmse"` (like LightGBM/XGBoost)
   - Always track all available metrics (simpler, slight overhead)
   Recommendation: `eval_metric` parameter, default `"rmse"` for backward compat.

---

## Implementation

### Phase 1: Engine-Side Metric Infrastructure

**`crates/engine/src/lib.rs`**

#### Step 1.1: Define metric enum

```rust
pub enum EvalMetric {
    Rmse,
    Mae,
    // Future: LogLoss, Auc, Ndcg, etc.
}
```

#### Step 1.2: Implement metric computation functions

```rust
fn mae_loss(predictions: &[f32], targets: &[f32]) -> EngineResult<f32> {
    // sum(|pred - target|) / N
}
```

`squared_error_loss()` already exists for RMSE.

#### Step 1.3: Generalize loss tracking in training loop

Currently, the 6 call sites to `squared_error_loss()` compute loss for:
- Initial training loss
- Per-round training loss
- Initial validation loss
- Per-round validation loss
- Early stopping comparison
- Final loss

Replace with a configurable metric set. The training loop should compute all requested metrics per round.

Change `IterationRunSummary`:
```rust
pub struct IterationRunSummary {
    // ... existing fields ...
    pub metrics_per_round: HashMap<String, Vec<f32>>,  // metric_name -> per-round values
    pub validation_metrics_per_round: HashMap<String, Vec<f32>>,
}
```

Or keep it simpler with parallel vectors:
```rust
pub loss_per_completed_round: Vec<f32>,  // keep existing (primary loss for early stopping)
pub additional_metrics_per_round: Vec<(String, Vec<f32>)>,  // extra metrics
pub additional_validation_metrics_per_round: Vec<(String, Vec<f32>)>,
```

#### Step 1.4: Early stopping metric

The early stopping logic should use one designated metric (the primary `eval_metric`). Other metrics are tracked for reporting only.

### Phase 2: Bridge Updates

**`bindings/python/src/lib.rs`**

Pass `eval_metric` through from Python to engine. Extend `NativeTrainingSummary` to include additional metric arrays.

### Phase 3: Python API

**`bindings/python/alloygbm/regressor.py`**

#### Step 3.1: Add `eval_metric` parameter

```python
GBMRegressor.__init__(
    # ...
    eval_metric: str | list[str] = "rmse",
)
```

#### Step 3.2: Expose metrics in `evals_result_`

```python
model.evals_result_
# Returns: {
#     "training": {"rmse": [...], "mae": [...]},
#     "validation": {"rmse": [...], "mae": [...]}
# }
```

Currently `evals_result_` only has `train_rmse` and `validation_rmse`.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Engine | 1 (`engine/src/lib.rs`) | ~60-100 | Medium |
| Phase 2: Bridge | 1 (`bindings/python/src/lib.rs`) | ~30-50 | Low |
| Phase 3: Python API | 1 (`regressor.py`) | ~30-40 | Low |

Total: ~120-190 lines across 3 files. Medium complexity due to the 6 loss call sites in the training loop.

---

## Testing Strategy

1. **Default behavior**: `eval_metric="rmse"` produces identical `evals_result_` to current behavior
2. **MAE tracking**: `eval_metric=["rmse", "mae"]` returns both metrics per round
3. **Early stopping**: Primary metric is used for early stopping, additional metrics are passive
4. **Metric consistency**: Manual computation of MAE on predictions matches tracked MAE

---

## Non-Goals

- **Custom Python callable metrics**: Too expensive per round (Rust-Python boundary). Defer to a future callback system.
- **Classification metrics**: Deferred to Classification & Ranking plan (#1)
- **Metric-based model selection**: Choosing best round by a non-primary metric. Could be added later.
