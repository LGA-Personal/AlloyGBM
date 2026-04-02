# Plan: User-Provided Feature Names

## Status: Not Started

## Summary

When the model artifact is serialized, feature names are auto-generated as `f0`, `f1`, ... (`to_artifact_bytes()` in engine/src/lib.rs:584). The user's real feature names from a DataFrame are not preserved through training. This affects SHAP explanations, feature importance output, and model interpretability.

---

## Questions to Resolve Before Starting

1. **Where to capture names**: From `X.columns` in `fit()` (DataFrame), or as an explicit `feature_names` parameter? Recommendation: auto-detect from DataFrame columns, with optional explicit override via `feature_name` parameter (like LightGBM).

2. **Artifact format**: Should feature names be stored in the binary artifact? They're already in the JSON metadata section (`ModelMetadata.feature_names`), but currently auto-generated. Recommendation: store user-provided names in the existing metadata field.

---

## Implementation

### Phase 1: Capture Feature Names in Python

**`bindings/python/alloygbm/regressor.py`**

#### Step 1.1: Extract names from DataFrame

In `fit()`, after validating X:
```python
if hasattr(X, 'columns'):
    self._feature_names = [str(c) for c in X.columns]
else:
    self._feature_names = [f"f{i}" for i in range(n_features)]
```

#### Step 1.2: Optional explicit parameter

Add `feature_name: list[str] | None = None` to `fit()`:
```python
def fit(self, X, y, *, feature_name: list[str] | None = None, ...):
    if feature_name is not None:
        if len(feature_name) != n_features:
            raise ValueError("feature_name length must match number of features")
        self._feature_names = [str(n) for n in feature_name]
```

#### Step 1.3: Use names in outputs

- `feature_importances_` property: return dict or DataFrame with feature names as keys
- `shap_values()`: return with feature names if available
- `__repr__` or `summary()`: show feature names

### Phase 2: Pass Names Through to Artifact

**`bindings/python/src/lib.rs`**

Add `feature_names: Option<Vec<String>>` parameter to training bridge functions. Pass through to engine.

**`crates/engine/src/lib.rs`**

In `TrainedModel::to_artifact_bytes()` (line ~584), use provided names instead of generating `f0, f1, ...`:
```rust
let feature_names = self.feature_names.clone().unwrap_or_else(|| {
    (0..feature_count).map(|i| format!("f{i}")).collect()
});
```

Add `feature_names: Option<Vec<String>>` field to `TrainedModel`.

**`crates/core/src/lib.rs`**

`ModelMetadata.feature_names` already exists and is serialized in JSON. No format change needed -- just populate with real names instead of auto-generated ones.

### Phase 3: Expose on Predictor (Read Path)

When loading a model (predictor), feature names are available in the deserialized `ModelMetadata`. Expose them through `NativePredictorHandle` so Python can read them back:

```python
model._feature_names = predictor.feature_names  # from artifact metadata
```

This matters for save/load (Limitation #5) -- after loading a saved model, feature names should be restored.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Python capture | 1 (`regressor.py`) | ~25-35 | Very Low |
| Phase 2: Artifact storage | 3 (bridge, engine, core) | ~30-50 | Low |
| Phase 3: Read path | 2 (bridge, regressor.py) | ~15-20 | Very Low |

Total: ~70-105 lines across 4 files.

---

## Testing Strategy

1. **DataFrame input**: `fit(df, y)` captures column names
2. **Explicit names**: `fit(X, y, feature_name=['a', 'b', 'c'])` overrides
3. **Numpy input**: Falls back to `f0, f1, ...`
4. **Artifact roundtrip**: Names survive serialization/deserialization
5. **SHAP output**: `shap_values()` result uses real feature names
6. **Feature importance**: `feature_importances_` uses real feature names

---

## Non-Goals

- **Feature name validation during predict**: Checking that predict-time DataFrame columns match training-time names. Nice-to-have but not essential.
- **Feature name-based parameter specification**: Using feature names instead of indices for `categorical_feature_indices`, `monotone_constraints`, etc. Possible future enhancement.
