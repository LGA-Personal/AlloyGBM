# Plan: Classification and Ranking Support

## Status: Not Started

## Summary

AlloyGBM currently only supports regression via `SquaredErrorObjective`. This plan covers adding binary classification, multi-class classification, and learning-to-rank objectives. The engine's `ObjectiveOps` trait and Newton-Raphson leaf value formula are already general-purpose -- the core work is wiring new objectives through the training loop, extending the artifact format, adding post-prediction transforms, and building new Python-facing classes.

---

## Questions to Resolve Before Starting

These must be answered by the user/prompter before implementation begins:

1. **Scope priority**: Should this be done incrementally (binary classification first, then multi-class, then ranking)? Or all at once?
2. **Python API shape**: Should classification be a separate `GBMClassifier` class or a mode on `GBMRegressor`? Convention (XGBoost, LightGBM, sklearn) strongly favors separate classes. Recommendation: separate `GBMClassifier` and `GBMRanker`.
3. **Multi-class strategy**: One-vs-rest (K independent binary models) vs. native multi-class (K outputs per tree, softmax loss)? One-vs-rest is simpler but slower and weaker. Native multi-class is what XGBoost/LightGBM do. Recommendation: native multi-class (but could defer to phase 2).
4. **Ranking objective**: Which ranking loss? LambdaMART (pairwise) is the industry standard. LambdaRank is simpler. Recommendation: LambdaMART with NDCG, but ranking could be a completely separate phase.
5. **Artifact backward compatibility**: Should old v1 artifacts (no objective field) be loadable by new code? Recommendation: yes, default to `"squared_error"` when the field is absent.
6. **SHAP compatibility**: The SHAP module is hardcoded to single-output models. Should multi-class SHAP be in scope? Recommendation: defer, document as known limitation.

---

## Architecture Overview

### What's Already General-Purpose (No Changes Needed)

- **Leaf value computation** (`engine/src/lib.rs:1270-1290`): Uses Newton-Raphson formula `leaf = -lr * grad_sum / (hess_sum + lambda + eps)`. This works for any objective that provides correct (grad, hess) pairs.
- **Tree building**: Histogram construction, split finding, partitioning -- all operate on `GradientPair` arrays. Completely objective-agnostic.
- **BackendOps trait**: CPU backend builds histograms from arbitrary gradient pairs.
- **Validation split system** (`validation.py`): Splits by row indices, objective-agnostic.

### What Must Change

| Component | File(s) | Change Required |
|---|---|---|
| ObjectiveOps trait | `crates/engine/src/lib.rs` | Add `loss()` method |
| New objective implementations | `crates/engine/src/lib.rs` | `BinaryCrossEntropyObjective`, `MultiClassObjective`, `LambdaMARTObjective` |
| Training loop loss monitoring | `crates/engine/src/lib.rs` | Replace 6 hardcoded `squared_error_loss()` calls |
| Leaf refinement | `crates/engine/src/lib.rs` | Guard or generalize `refine_regression_leaf_values()` |
| ModelMetadata | `crates/core/src/lib.rs` | Add `objective` and `num_classes` fields |
| Metadata JSON serde | `crates/core/src/lib.rs` | Extend serializer/deserializer with backward compat |
| Predictor | `crates/predictor/src/lib.rs` | Add post-prediction transforms (sigmoid, softmax) |
| Python bridge | `bindings/python/src/lib.rs` | Accept `objective` param, dispatch to correct objective |
| Python classifier class | `bindings/python/alloygbm/classifier.py` | New file: `GBMClassifier` |
| Python ranker class | `bindings/python/alloygbm/ranker.py` | New file: `GBMRanker` |
| Python evaluation | `bindings/python/alloygbm/evaluation.py` | Add `log_loss`, `accuracy`, `auc_roc`, `ndcg` |
| Python `__init__.py` | `bindings/python/alloygbm/__init__.py` | Export new classes and metrics |

---

## Implementation Steps

### Phase 1: Engine-Level Objective Generalization

#### Step 1.1: Extend ObjectiveOps with a Loss Method

**File:** `crates/engine/src/lib.rs`, trait `ObjectiveOps` (line 108)

Add a required method:

```rust
fn loss(
    &self,
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32>;
```

And update `SquaredErrorObjective` to implement it by moving the existing `squared_error_loss()` logic (lines 2694-2761) into this method. The standalone `squared_error_loss()` function can then delegate to `SquaredErrorObjective.loss()` or be kept as a convenience.

**Why a trait method instead of a standalone function:** The training loop (`fit_iterations_with_optional_validation_summary`) is already generic over `O: ObjectiveOps`. Making loss a trait method means the loop naturally dispatches to the correct loss without branching.

**Success criteria:** All existing tests pass unchanged. The `squared_error_loss()` call sites in the training loop now go through the trait.

#### Step 1.2: Replace Hardcoded Loss Calls in the Training Loop

**File:** `crates/engine/src/lib.rs`, function `fit_iterations_with_optional_validation_summary` (line 1110)

There are exactly 6 call sites to replace:

| Line | Context | Change |
|------|---------|--------|
| ~1155 | `initial_loss = squared_error_loss(...)` | `objective.loss(...)` |
| ~1166 | Initial validation loss | `objective.loss(...)` |
| ~1379 | Candidate round loss | `objective.loss(...)` |
| ~1415 | Validation loss per round | `objective.loss(...)` |
| ~1505 | Loss after leaf refinement | `objective.loss(...)` |
| ~1521 | Validation loss after leaf refinement | `objective.loss(...)` |

Note: the `objective` parameter is already in scope as `O: ObjectiveOps` throughout this function.

**Troubleshooting:** After this change, run the full engine test suite. Every existing regression test should produce identical results since `SquaredErrorObjective.loss()` is the same computation as `squared_error_loss()`.

**Success criteria:** `cargo test -p alloygbm-engine` passes with zero behavioral changes.

#### Step 1.3: Guard Leaf Refinement for Non-MSE Objectives

**File:** `crates/engine/src/lib.rs`, `refine_regression_leaf_values()` (line 2339)

This function computes residuals as `target - baseline - ensemble_prediction`, which is specific to MSE. Options:

- **Option A (recommended for now):** Only call `refine_regression_leaf_values()` when the objective is MSE. The call is already gated behind `ALLOYGBM_ENABLE_LEAF_REFINEMENT` env var (experimental), so it's low-risk to skip it for other objectives.
- **Option B (future):** Generalize refinement to use objective-specific gradient recomputation. Significantly more complex.

**How to gate it:** Add an `fn supports_leaf_refinement(&self) -> bool` method to `ObjectiveOps` (default `false`, `SquaredErrorObjective` returns `true`). Check it before calling refinement.

#### Step 1.4: Implement BinaryCrossEntropyObjective

**File:** `crates/engine/src/lib.rs` (new impl block, near `SquaredErrorObjective`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryCrossEntropyObjective;
```

**`initial_prediction`**: Log-odds of the positive class mean. Given targets in {0, 1}:
```
p = weighted_mean(targets)
initial_prediction = ln(p / (1 - p))
```
Clamp `p` to `[1e-7, 1 - 1e-7]` to avoid infinities.

**`compute_gradients`**: For each sample with prediction `f` (raw log-odds):
```
p = sigmoid(f) = 1 / (1 + exp(-f))
grad = (p - target) * weight
hess = p * (1 - p) * weight
```
Clamp `hess` to `[1e-7, ...]` to ensure positivity (the `GradientPair` constructor enforces `hess > 0`).

**`loss`**: Binary cross-entropy (log-loss):
```
p = sigmoid(prediction)
loss = -mean(target * ln(p) + (1 - target) * ln(1 - p))
```
With sample weights: weighted average.

**Validation:** Targets must be exactly 0.0 or 1.0 (or allow any value in [0, 1] for soft labels -- decide with user). Reject targets outside this range in `initial_prediction` and `compute_gradients`.

**Testing strategy:**
- Unit test: `initial_prediction` on balanced dataset returns ~0.0.
- Unit test: `initial_prediction` on 75% positive returns `ln(3)`.
- Unit test: gradients at the optimum (prediction = log-odds of target) are near zero.
- Unit test: loss decreases after one gradient step.
- Integration test: train a binary classifier on a simple linearly-separable dataset, verify >90% accuracy.

#### Step 1.5: Implement MultiClassObjective (if in scope)

**File:** `crates/engine/src/lib.rs`

Multi-class is substantially more complex because it requires K models (one per class) trained jointly. Two implementation approaches:

**Approach A -- K Independent Trees Per Round (simpler, recommended first):**
- Store `num_classes` in the objective.
- `initial_prediction` returns a vector of K log-probabilities (not a single f32). This breaks the current `EngineResult<f32>` return type.
- Each round builds K trees (one per class), each using class-specific gradients.

This requires either:
1. Changing `initial_prediction` to return `Vec<f32>`, which is a large trait change.
2. Running K separate training loops and combining results.

**Approach B -- External Loop Over Classes (simplest):**
- Use one-vs-rest: train K binary classifiers, combine probabilities via softmax.
- No engine changes required -- just Python-level orchestration.
- Weaker than native multi-class but dramatically simpler.

**Recommendation:** Start with binary classification only. Multi-class can be added as Approach B (one-vs-rest in Python) initially, with Approach A as a future optimization.

#### Step 1.6: Implement LambdaMARTObjective (if in scope)

**File:** `crates/engine/src/lib.rs`

LambdaMART requires:
- `group_id` to know which rows belong to the same query.
- Pairwise gradient computation: for each pair of documents in a query, compute the gradient based on their NDCG impact if swapped.
- `initial_prediction`: typically 0.0 (no prior).
- `loss`: NDCG (or 1 - NDCG).

**Key complexity:** Gradient computation is O(n^2) per query group (pairwise), not O(n) per row. The `ObjectiveOps::compute_gradients` signature works (it takes all predictions + targets), but the implementation needs access to `group_id`. Options:
1. Pass `group_id` into the objective at construction time.
2. Add `group_id` as a parameter to `compute_gradients`.

Option 1 is cleaner (store group boundaries in the objective struct).

**Recommendation:** Defer ranking to a separate phase after classification is stable.

---

### Phase 2: Artifact Format Extension

#### Step 2.1: Extend ModelMetadata

**File:** `crates/core/src/lib.rs`, struct `ModelMetadata` (line 396)

Add two fields:

```rust
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub trained_device: Device,
    pub objective: ObjectiveKind,    // NEW
    pub num_classes: Option<u32>,    // NEW (None for regression/binary, Some(K) for multiclass)
}
```

Define the enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveKind {
    SquaredError,
    BinaryCrossEntropy,
    MultiClassCrossEntropy,
    LambdaMART,
}
```

With string conversion methods for JSON serialization (e.g., `"squared_error"`, `"binary_crossentropy"`, `"multiclass_crossentropy"`, `"lambdamart"`).

#### Step 2.2: Update Metadata JSON Serialization

**File:** `crates/core/src/lib.rs`, `serialize_metadata_json()` (line 1109)

Current format:
```json
{"format_version":1,"feature_names":[...],"trained_device":"cpu"}
```

New format:
```json
{"format_version":1,"feature_names":[...],"trained_device":"cpu","objective":"squared_error","num_classes":null}
```

The new fields are appended at the end. `num_classes` is `null` for regression/binary, an integer for multi-class.

#### Step 2.3: Update Metadata JSON Deserialization with Backward Compatibility

**File:** `crates/core/src/lib.rs`, `deserialize_metadata_json()` (line 1125)

The current parser is positional and rigid -- it uses `consume_literal` to expect exact field order and rejects trailing content. The change:

After parsing `trained_device`, attempt to parse `,"objective":` and `,"num_classes":`. If the next character is `}` instead, the artifact is a legacy v1 artifact -- default to `ObjectiveKind::SquaredError` and `num_classes: None`.

```rust
// After parsing trained_device:
let (objective, num_classes, next_index) = if compact[index..].starts_with(",\"objective\":") {
    // Parse new fields
    ...
} else {
    // Legacy artifact -- default to regression
    (ObjectiveKind::SquaredError, None, index)
};
index = next_index;
index = consume_literal(&compact, index, "}")?;
```

**Testing strategy:**
- Roundtrip test: serialize new metadata, deserialize, verify equality.
- Backward compat test: deserialize an old-format JSON string (no objective field), verify defaults.
- Reject test: malformed objective string returns error.

**Success criteria:** All existing artifact deserialization tests pass unchanged (they produce legacy artifacts with no objective field, which now default to `SquaredError`).

---

### Phase 3: Predictor Post-Transforms

#### Step 3.1: Add Transform Awareness to Predictor

**File:** `crates/predictor/src/lib.rs`, struct `Predictor` (line 69)

Add a field:

```rust
pub struct Predictor {
    // ... existing fields ...
    objective: ObjectiveKind,
    num_classes: Option<u32>,
}
```

Populate from `ModelMetadata` during `from_artifact_bytes()`.

#### Step 3.2: Add Raw and Transformed Prediction Methods

The current `predict_row` / `predict_batch` / `predict_batch_dense` methods return raw additive model output. For classification:

- **Binary**: Raw output is log-odds. Apply `sigmoid(x) = 1 / (1 + exp(-x))` to get probability.
- **Multi-class**: Raw outputs are K log-probabilities. Apply softmax to get class probabilities.

Add new methods:

```rust
/// Returns raw model output (no transformation). This is what predict_row currently does.
pub fn predict_row_raw(&self, features: &[f32]) -> PredictorResult<f32>;

/// Returns transformed output (sigmoid for binary, identity for regression).
pub fn predict_row(&self, features: &[f32]) -> PredictorResult<f32>;
```

For regression, `predict_row` and `predict_row_raw` are identical. For binary classification, `predict_row` applies sigmoid. The batch variants follow the same pattern.

**Critical detail:** The training loop must use RAW predictions (log-odds) for gradient computation, not transformed predictions. Only the final user-facing predict applies the transform. The engine's `TrainedModel::predict_row` should remain raw. The transform is a predictor-level concern.

**For multi-class:** The predictor would need to return `Vec<f32>` (K probabilities). This is a bigger API change. If multi-class is deferred, this can wait.

#### Step 3.3: Update Predictor Batch Methods

All batch methods (`predict_batch`, `predict_batch_dense`, `predict_batch_dense_bytes`) call the row-level prediction. Ensure the transform is applied consistently. Consider adding `predict_batch_raw` variants for cases where callers need raw output (e.g., SHAP).

---

### Phase 4: Python Bridge

#### Step 4.1: Add Objective Parameter to Training Functions

**File:** `bindings/python/src/lib.rs`

The core implementation function is `train_regression_artifact_with_summary_dense_impl()` (~line 1332). Currently it hardcodes `&SquaredErrorObjective`.

Add an `objective: &str` parameter (default `"squared_error"`). Map string to objective:

```rust
fn resolve_objective(objective: &str) -> EngineResult<Box<dyn ObjectiveOps>> {
    match objective {
        "squared_error" | "regression" => Ok(Box::new(SquaredErrorObjective)),
        "binary_crossentropy" | "binary" => Ok(Box::new(BinaryCrossEntropyObjective)),
        _ => Err(EngineError::InvalidConfig(format!("unsupported objective: {objective}"))),
    }
}
```

**Note on trait objects vs generics:** The current code uses `O: ObjectiveOps` generics. Using `Box<dyn ObjectiveOps>` would require making `ObjectiveOps` object-safe. Alternatively, match on the string and call the appropriate monomorphized path. The latter avoids dynamic dispatch and is simpler:

```rust
match objective_str {
    "squared_error" => trainer.fit_iterations_with_summary(..., &SquaredErrorObjective, ...),
    "binary_crossentropy" => trainer.fit_iterations_with_summary(..., &BinaryCrossEntropyObjective, ...),
    _ => return Err(...)
}
```

This approach duplicates the call but avoids trait object complications.

#### Step 4.2: Update All 5 Training Pyfunctions

Each of the 5 training functions (`train_regression_artifact`, `train_regression_artifact_dense`, etc.) needs an `objective` parameter with a default of `"squared_error"`. The parameter flows through to the impl function.

**Backward compatibility:** The default value ensures all existing Python code continues to work unchanged.

#### Step 4.3: Update Predictor Bridge for Transforms

The `NativePredictorHandle` prediction methods currently return raw values. After Phase 3, the `Predictor` itself handles transforms. The Python bridge needs no changes for binary classification since the Predictor methods will return transformed values.

However, add a `predict_raw` method on `NativePredictorHandle` for cases where the caller wants raw log-odds (e.g., for custom evaluation).

---

### Phase 5: Python Classifier Class

#### Step 5.1: Create GBMClassifier

**File:** `bindings/python/alloygbm/classifier.py` (new file)

`GBMClassifier` mirrors `GBMRegressor`'s structure but:

- Constructor accepts the same hyperparameters but no `objective` param (it's always binary_crossentropy or multiclass_crossentropy, auto-detected from labels).
- `fit(X, y, ...)`: Validates that `y` contains valid class labels. For binary: exactly {0, 1} or {True, False}. For multi-class: integers 0..K-1 or string labels (mapped to integers).
- `predict(X)`: Returns class labels (integers). Applies sigmoid then thresholds at 0.5 for binary. Applies softmax then argmax for multi-class.
- `predict_proba(X)`: Returns probabilities. For binary: list of P(y=1) values. For multi-class: list of K-element probability vectors.
- `predict_log_proba(X)`: Returns log-probabilities (optional, nice-to-have).
- Stores `classes_: list` attribute after fitting (the set of unique class labels).
- Stores `n_classes_: int` attribute.

**Label encoding:** If the user passes string labels like `["cat", "dog", "fish"]`, `GBMClassifier.fit()` maps them to integers (sorted order) and stores the mapping. `predict()` maps back to the original labels.

**Binary vs multi-class auto-detection:** If `n_classes == 2`, use `BinaryCrossEntropyObjective`. If `n_classes > 2`, use `MultiClassObjective` (or one-vs-rest).

#### Step 5.2: Create GBMRanker (if in scope)

**File:** `bindings/python/alloygbm/ranker.py` (new file)

`GBMRanker` requires:
- `fit(X, y, *, group=None)`: `group` is required, specifies query group boundaries.
- `predict(X)`: Returns relevance scores (raw output, no transform).
- No `predict_proba`.

**Defer this unless ranking is in scope for this phase.**

#### Step 5.3: Add Classification Evaluation Metrics

**File:** `bindings/python/alloygbm/evaluation.py`

Add:

```python
def accuracy(y_true, y_pred) -> float:
    """Fraction of correct predictions."""

def log_loss(y_true, y_prob) -> float:
    """Binary cross-entropy loss. y_prob should be probabilities, not labels."""

def auc_roc(y_true, y_prob) -> float:
    """Area under ROC curve for binary classification."""
```

For multi-class (if in scope):
```python
def multiclass_log_loss(y_true, y_prob) -> float:
    """Multi-class cross-entropy."""
```

For ranking (if in scope):
```python
def ndcg(y_true, y_pred, *, k=None) -> float:
    """Normalized Discounted Cumulative Gain."""
```

**Implementation note:** These are pure Python functions, no Rust needed. Keep them dependency-free (no sklearn, no numpy required), matching the existing evaluation module's style.

#### Step 5.4: Update __init__.py

**File:** `bindings/python/alloygbm/__init__.py`

Add exports:
```python
from .classifier import GBMClassifier
from .ranker import GBMRanker  # if implemented
from .evaluation import accuracy, log_loss, auc_roc  # new metrics
```

---

### Phase 6: Training Summary Updates

The `NativeTrainingSummary` (in `bindings/python/src/lib.rs`) currently exposes `train_rmse` and `validation_rmse`. For classification:

- Binary: track `train_logloss` / `validation_logloss` instead of RMSE.
- The loss values per round are already computed (they come from `objective.loss()`). The naming just needs to change.

**Approach:** Rename `train_rmse` / `validation_rmse` to generic `train_loss` / `validation_loss` in the summary struct. Or add parallel fields. The Python `evals_result_` dict would use `"logloss"` as the metric key instead of `"rmse"`.

**Backward compatibility concern:** If external code reads `summary.train_rmse`, renaming breaks it. Options:
1. Keep `train_rmse` and add `train_loss` alongside (redundant but safe).
2. Just rename since the package is pre-1.0.
3. Make the metric name dynamic based on objective.

Recommendation: option 3 -- expose `train_loss` and `validation_loss` as the generic fields, plus a `loss_metric_name: str` field ("rmse", "logloss", etc.) so the Python side knows what it's looking at.

---

## Testing Strategy

### Unit Tests (Rust)

1. **BinaryCrossEntropyObjective**:
   - `initial_prediction` on balanced data ≈ 0.0.
   - `initial_prediction` on skewed data = correct log-odds.
   - `compute_gradients` at the optimum ≈ zero.
   - `loss` decreases monotonically over training iterations.
   - Rejects targets outside [0, 1].
   - Weighted gradients match expected values.

2. **Training loop with binary objective**:
   - Single-round training produces a valid model.
   - Multi-round training reduces loss.
   - Early stopping works with logloss.
   - Artifact serialization roundtrip preserves objective metadata.

3. **Predictor with transforms**:
   - Binary predictor output is in (0, 1).
   - `sigmoid(0) = 0.5`.
   - Raw vs transformed predictions are consistent.

### Integration Tests (Python)

1. **GBMClassifier on synthetic data**:
   - Linearly separable 2D data: accuracy > 95%.
   - `predict_proba` values sum to 1.0 per row.
   - `predict` returns integer class labels.
   - Fitted attributes (`classes_`, `n_classes_`, `best_iteration_`) are populated.

2. **Backward compatibility**:
   - Old regression artifacts load correctly in new code.
   - `GBMRegressor` behavior is completely unchanged.

3. **Evaluation metrics**:
   - `log_loss` on perfect predictions ≈ 0.
   - `accuracy` on known data matches expected.
   - `auc_roc` on perfectly separated data = 1.0.

### Benchmark Tests

- Compare AlloyGBM binary classification performance (accuracy, logloss, training speed) against LightGBM and XGBoost on a standard dataset (e.g., sklearn's breast_cancer or a synthetic dataset).
- This is important for validating that the gradient/hessian implementation is correct -- wrong gradients will produce models that are far worse than competitors.

---

## Success Criteria

### Binary Classification (Minimum Viable)

- [ ] `GBMClassifier` can fit binary targets {0, 1} and produce a model.
- [ ] `predict()` returns class labels.
- [ ] `predict_proba()` returns probabilities in (0, 1) that sum correctly.
- [ ] Training loss (logloss) decreases monotonically or stabilizes.
- [ ] Early stopping works with validation logloss.
- [ ] Model artifacts serialize/deserialize correctly with objective metadata.
- [ ] Old regression artifacts continue to load and work.
- [ ] `log_loss()` and `accuracy()` evaluation metrics are available.
- [ ] All existing regression tests pass unchanged.

### Multi-Class Classification (if in scope)

- [ ] `GBMClassifier` auto-detects K > 2 classes and trains accordingly.
- [ ] `predict_proba()` returns K-element vectors that sum to 1.0 per row.
- [ ] Supports both integer and string class labels.

### Ranking (if in scope)

- [ ] `GBMRanker` accepts group parameter and trains with LambdaMART.
- [ ] `predict()` returns relevance scores.
- [ ] NDCG evaluation metric is available.

---

## Risk Areas and Troubleshooting

### Numerical Stability in Sigmoid/Log-Loss

The sigmoid function `1 / (1 + exp(-x))` overflows for large negative `x` and underflows for large positive `x`. Use the stable formulation:
```rust
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}
```

Similarly, log-loss requires `ln(p)` where `p` could be 0. Clamp predictions to `[1e-7, 1 - 1e-7]` before taking the log.

### Hessian Positivity

The `GradientPair::new()` constructor requires `hess > 0.0`. For binary cross-entropy, `hess = p * (1 - p)` which is always in `(0, 0.25]` for `p in (0, 1)`. But floating-point precision at extreme predictions could produce zero. Clamp: `hess = max(p * (1 - p), 1e-7)`.

### Training Loop Loss Monitoring

After replacing `squared_error_loss` with `objective.loss()`, verify that the early stopping and loss-improvement logic still works correctly. Different objectives have very different loss scales (MSE can be in the thousands, logloss is typically < 1.0). The `min_loss_improvement` and `max_consecutive_weak_improvements` thresholds in the auto training policy may need objective-aware tuning.

### Auto Training Policy

The auto policy (`auto_iteration_controls` at engine line 1034) uses `target_variance` and dataset size to set controls. This logic is regression-specific. For classification:
- `target_variance` of {0, 1} targets is always p*(1-p), which is at most 0.25. This will interact poorly with the thresholds.
- Solution: either bypass auto policy for classification (use manual), or add objective-aware policy branches.

### Feature Name Pass-Through

Currently feature names are auto-generated as `f0, f1, ...` in `to_artifact_bytes()`. This is Limitation #12 in the limitations doc and is orthogonal to this work, but classification users will want meaningful feature names in importance outputs. Consider addressing Limitation #12 as a prerequisite or concurrent fix.

---

## File-by-File Change Summary

| File | Lines Changed (est.) | Nature |
|------|---------------------|--------|
| `crates/core/src/lib.rs` | ~80 | Add ObjectiveKind enum, extend ModelMetadata, update serde |
| `crates/engine/src/lib.rs` | ~200 | Add loss() to ObjectiveOps, implement BinaryCrossEntropyObjective, replace hardcoded loss calls, guard refinement |
| `crates/predictor/src/lib.rs` | ~60 | Add objective awareness, sigmoid transform |
| `bindings/python/src/lib.rs` | ~80 | Add objective parameter to training functions |
| `bindings/python/alloygbm/classifier.py` | ~400 | New file: GBMClassifier |
| `bindings/python/alloygbm/evaluation.py` | ~80 | New metrics: accuracy, log_loss, auc_roc |
| `bindings/python/alloygbm/__init__.py` | ~5 | Export new classes/metrics |
| Tests (various) | ~300 | New unit and integration tests |

**Total estimated: ~1200 lines of new/changed code.**

---

## Suggested Implementation Order

1. **Step 1.1-1.2**: Generalize loss in ObjectiveOps and training loop. Verify all existing tests pass.
2. **Step 1.3**: Guard leaf refinement. Verify existing tests.
3. **Step 2.1-2.3**: Extend artifact format with backward compat. Verify old artifacts still load.
4. **Step 1.4**: Implement BinaryCrossEntropyObjective with unit tests.
5. **Step 3.1-3.3**: Add predictor transforms. Unit test sigmoid output.
6. **Step 4.1-4.3**: Wire objective through Python bridge.
7. **Step 5.1**: Build GBMClassifier. Integration test on synthetic data.
8. **Step 5.3-5.4**: Add evaluation metrics and exports.
9. **Step 6**: Update training summary for generic loss metric.
10. **Benchmark**: Compare against XGBoost/LightGBM on a standard binary classification dataset.

Each step should be independently testable and committable.
