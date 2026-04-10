# Feature Importances And SHAP

AlloyGBM exposes SHAP-based explanation methods from the Python API, backed by
a native Rust TreeSHAP implementation.

## Local Explanations

Use `shap_values(...)` to get per-row, per-feature attributions:

```python
from alloygbm import GBMRegressor

model = GBMRegressor(deterministic=True, seed=7)
model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])

values = model.shap_values([[1.5], [2.5]])
print(values)
```

If you also want the model expected value:

```python
expected_value, values = model.shap_values(
    [[1.5], [2.5]],
    include_expected_value=True,
)
```

## Global Importance

Use `feature_importances(...)` to aggregate SHAP importances across rows:

```python
importance = model.feature_importances([[0.5], [1.5], [2.5]])
print(importance)
```

The current supported method is:

- `method="shap"`

## What To Expect

- `shap_values(...)` returns one attribution per feature for each input row.
- `feature_importances(...)` returns `(feature_name, importance)` tuples.
- Feature names are captured from training data column names when available
  (e.g. pandas DataFrames), or auto-generated as `f0`, `f1`, etc.

## TreeSHAP Implementation

AlloyGBM uses the polynomial-time TreeSHAP algorithm for computing exact
Shapley values. This means:

- There is no practical limit on the number of features.
- Computation scales with tree complexity, not exponentially with feature count.
- Results are exact (not approximate).

The previous brute-force SHAP method (limited to 25 features) has been replaced
by TreeSHAP in `v0.2.0`.

## Supported Estimators

SHAP explanations work with all three estimators:

- `GBMRegressor.shap_values(...)`
- `GBMClassifier.shap_values(...)`
- `GBMRanker.shap_values(...)`
