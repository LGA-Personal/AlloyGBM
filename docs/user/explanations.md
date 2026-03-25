# Feature Importances And SHAP

AlloyGBM exposes SHAP-based explanation methods from the Python API.

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
- Feature names currently default to generated names such as `f0`, `f1`, and so on.

## Current Scope

AlloyGBM currently exposes SHAP for regression artifacts. This is part of the
core public API and is backed by native Rust code, but the surrounding
explainability surface is still intentionally narrow.
