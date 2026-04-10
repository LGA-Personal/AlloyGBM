# Plan: Native sklearn Compatibility

## Status: Not Started

## Summary

`GBMRegressor` already has `get_params()`, `set_params()`, `fit()`, and `predict()` -- the core sklearn estimator interface. However, it does not inherit from `sklearn.base.BaseEstimator` or `RegressorMixin`, which means it can't be used with `cross_val_score`, `GridSearchCV`, `Pipeline`, or other sklearn utilities. It's also missing `score()` and `__sklearn_tags__`.

This plan covers adding full sklearn compatibility with minimal coupling to sklearn as a dependency.

---

## Questions to Resolve Before Starting

1. **sklearn as a dependency**: Should sklearn be a required dependency or optional? Recommendation: optional. Import sklearn only when needed (e.g., in `score()`). The base class mixin approach can be done conditionally.

2. **Multiple inheritance approach**: Options:
   - **Option A**: Directly inherit from `sklearn.base.BaseEstimator, RegressorMixin` -- cleanest, but creates a hard dependency
   - **Option B**: Conditionally inherit (try to import sklearn, fall back to plain class) -- no hard dependency but complex metaclass logic
   - **Option C**: Implement the sklearn protocol without inheriting -- define `get_params`, `set_params`, `score`, `__sklearn_tags__` etc. manually. Works with `check_estimator` if all the right methods/attributes exist.
   Recommendation: **Option A** with sklearn as an optional dependency. If sklearn is not installed, `GBMRegressor` works standalone. If installed, it additionally inherits from the mixins. This is what LightGBM does.

3. **`check_estimator` compliance**: Should AlloyGBM pass sklearn's `check_estimator()` test suite? This is strict and requires specific behaviors (e.g., `fit` must accept sparse matrices, `predict` must handle 1D input gracefully, etc.). Recommendation: aim for compliance but don't contort the API for edge cases that don't matter in practice.

4. **Future `GBMClassifier`**: The sklearn plan should be designed so that a future `GBMClassifier` can also inherit from `ClassifierMixin`. Keep the pattern generalizable.

---

## Phase 1: Conditional sklearn Base Class Inheritance

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 1.1: Conditional import and class construction

```python
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

if _SKLEARN_AVAILABLE:
    class _GBMRegressorBase(BaseEstimator, RegressorMixin):
        pass
else:
    class _GBMRegressorBase:
        pass

class GBMRegressor(_GBMRegressorBase):
    # ... existing implementation ...
```

#### Step 1.2: `get_params()` compatibility

sklearn's `BaseEstimator.get_params()` introspects `__init__` signature parameters. AlloyGBM's `get_params()` is manually maintained. Two options:
- **Keep manual `get_params()`**: Override `BaseEstimator.get_params()`. More explicit, no risk of sklearn introspecting wrong.
- **Use sklearn's introspection**: Remove custom `get_params()`, rely on sklearn's. Requires `__init__` parameter names to exactly match stored attribute names (they already do).

Recommendation: Keep the existing manual `get_params()` and `set_params()` to avoid subtle breakage. sklearn's `BaseEstimator` will use the overridden methods.

### Success Criteria

- `isinstance(model, BaseEstimator)` returns `True` when sklearn is installed
- `isinstance(model, RegressorMixin)` returns `True` when sklearn is installed
- Without sklearn installed, `GBMRegressor` still works as before

---

## Phase 2: Add `score()` Method

### Implementation

`RegressorMixin` provides a default `score()` that computes R² using `sklearn.metrics.r2_score`. If we inherit from `RegressorMixin`, we get this for free. If not (sklearn not installed), provide a standalone implementation:

```python
def score(self, X, y, sample_weight=None):
    """Return R² score for the given test data and labels."""
    from sklearn.metrics import r2_score  # lazy import
    return r2_score(y, self.predict(X), sample_weight=sample_weight)
```

If sklearn isn't installed, raise `ImportError` with a helpful message.

Alternative: implement R² without sklearn (it's just `1 - SS_res / SS_tot`). This avoids any sklearn dependency for `score()`.

Recommendation: Implement R² directly (no sklearn import needed for `score()`), since AlloyGBM already has `r2_score` in `evaluation.py`.

```python
def score(self, X, y, sample_weight=None):
    """Return R² score for the given test data."""
    from alloygbm.evaluation import r2_score
    predictions = self.predict(X)
    return r2_score(y, predictions)
```

Note: the current `r2_score` in `evaluation.py` doesn't accept `sample_weight`. Adding `sample_weight` support to `r2_score` would be a small enhancement for full `score()` compatibility.

### Success Criteria

- `model.score(X_test, y_test)` returns a float R² value
- Works with and without sklearn installed

---

## Phase 3: `__sklearn_tags__` Support

### Overview

sklearn 1.6+ uses `__sklearn_tags__` (replacing the older `_more_tags()`) to communicate estimator capabilities. This tells sklearn utilities what the estimator supports.

### Implementation

```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__() if hasattr(super(), '__sklearn_tags__') else {}
    # Override specific tags
    tags.update({
        "non_deterministic": not self.deterministic,
        "requires_y": True,
        "no_validation": False,  # we do our own validation
        "poor_score": False,
        "X_types": ["2darray"],  # we accept numpy arrays and DataFrames
        "allow_nan": False,  # NaN not supported (until Limitation #4 is resolved)
    })
    return tags
```

For older sklearn versions, also implement `_more_tags()`:
```python
def _more_tags(self):
    return {
        "allow_nan": False,
        "requires_y": True,
    }
```

### Success Criteria

- `sklearn.utils.estimator_checks.check_estimator(GBMRegressor())` passes (or has only expected/documented failures)

---

## Phase 4: Pipeline and Cross-Validation Compatibility

### Already Working (No Changes Needed)

With phases 1-3 done, these should work out of the box:
- `sklearn.model_selection.cross_val_score(model, X, y)`
- `sklearn.model_selection.GridSearchCV(model, param_grid)`
- `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('gbm', GBMRegressor())])`

### Potential Issues

1. **`fit()` signature**: sklearn expects `fit(X, y)`. AlloyGBM's `fit()` already accepts this. The extra keyword arguments (`categorical_feature_values`, `eval_set`, `time_index`) are fine -- sklearn ignores unknown kwargs in most contexts.

2. **`predict()` input types**: sklearn may pass numpy arrays, sparse matrices, or DataFrames. AlloyGBM's `predict()` already handles numpy arrays and DataFrames. Sparse matrix support is out of scope (would require significant changes).

3. **Cloning**: `sklearn.base.clone()` calls `get_params()` then constructs a new instance with those params. This should work since `get_params()` returns all constructor parameters.

### Testing

- `cross_val_score(GBMRegressor(n_estimators=10), X, y, cv=3)` returns 3 scores
- `GridSearchCV(GBMRegressor(), {'max_depth': [3, 6]}).fit(X, y)` finds best params
- `clone(fitted_model)` returns an unfitted copy with same params

---

## Estimated Complexity

| Phase | Lines Changed | Risk |
|-------|--------------|------|
| Phase 1: Base class | ~15-20 | Low |
| Phase 2: `score()` | ~10-15 | Very Low |
| Phase 3: `__sklearn_tags__` | ~20-30 | Low |
| Phase 4: Testing | ~50-80 (test code) | Low |

Total: ~50-70 lines of production code, ~50-80 lines of test code. Single file change (`regressor.py`), plus optional test file.

---

## Risk Areas

### sklearn Version Compatibility

The `__sklearn_tags__` API changed in sklearn 1.6. Older versions use `_more_tags()`. Implement both for broad compatibility. Use `hasattr` checks.

### `check_estimator` Strictness

sklearn's `check_estimator` runs many checks, some of which may be hard to satisfy:
- **"estimators_fit_returns_self"**: `fit()` must return `self`. AlloyGBM's `fit()` already returns `self`.
- **"check_estimators_pickle"**: Estimator must be picklable. This overlaps with Limitation #5 (Model Save/Load). Consider implementing `__getstate__`/`__setstate__` as part of this plan or deferring to the persistence plan.
- **"check_estimators_unfitted"**: Calling `predict()` before `fit()` should raise `NotFittedError`. AlloyGBM currently raises a generic error. Should raise `sklearn.exceptions.NotFittedError` (or a custom exception that sklearn recognizes).

### Optional Dependency Management

Need to handle the case where sklearn is installed at class definition time but might not be at runtime (unlikely but possible with editable installs). Use lazy imports where possible.

---

## Non-Goals

- **Sparse matrix support**: Would require fundamental changes to the training pipeline
- **`partial_fit()` / warm-starting via sklearn interface**: Covered in Limitation #14
- **`GBMClassifier` sklearn integration**: Covered in Limitation #1 plan, but the infrastructure built here will be reused
- **Full `check_estimator` compliance**: Aim for it, but don't compromise the API for edge-case checks
