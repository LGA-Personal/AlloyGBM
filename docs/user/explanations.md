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

## Leaf Model Compatibility

`leaf_model="constant"` artifacts produce exact SHAP attributions satisfying
`Σ shap_values + expected_value == predict(x)`.

As of v0.7.1, `shap_values(...)` and `feature_importances(...)` also accept
`leaf_model="linear"` artifacts and return an *interventional* decomposition:
the path-based TreeSHAP / brute-force machinery attributes each leaf's
"constant part" (`intercept + Σ wⱼ · μⱼ_global`), then per-leaf row
deviations `wⱼ · (xⱼ − μⱼ_global)` are credited directly to the regressor
features. Global per-feature means `μⱼ_global` are captured at fit time and
persisted in a new `FeatureBaseline` artifact section, so SHAP is
self-contained — the original training data is not required at explain time.

Exact additivity holds when SHAP's internal path walker reaches the same leaf
as the predictor. Today SHAP compares raw feature values against stump
`threshold_bin` indices cast to `f32`, while the predictor converts bin
indices to float thresholds at load time using per-feature min/max. For
scalar leaves this divergence is masked. For linear leaves the leaf value
depends on `xⱼ`, so on continuous-feature artifacts the SHAP path and the
predictor path can disagree slightly. The strict additivity check is
relaxed for linear-leaf models; users get best-effort SHAP values.
Tightening path-walk alignment is queued for v0.7.2. See
[../limitations.md](../limitations.md) for the full caveat.
