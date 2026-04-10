# Plan: Model Save/Load (Persistence)

## Status: Not Started

## Summary

`GBMRegressor` has no `save_model()` / `load_model()` methods. The fitted state lives in memory as `_artifact_bytes` plus quantization metadata (`_continuous_feature_mins`, `_continuous_feature_maxs`, `_continuous_feature_sorted_values`, etc.). There's no pickle/joblib support (`__getstate__`/`__setstate__`) and no `to_file()` / `from_file()` convenience. The raw artifact bytes exist internally but can't be extracted or restored programmatically.

---

## Questions to Resolve Before Starting

1. **File format**: Should the saved model be:
   - **Option A**: The raw binary artifact bytes (compact, fast) plus a JSON sidecar for Python-side metadata
   - **Option B**: A single file containing both (e.g., zip/tar with artifact.bin + metadata.json)
   - **Option C**: Pickle-compatible `__getstate__`/`__setstate__` (Python ecosystem standard)
   Recommendation: All three. Option A as the low-level API (`save_model`/`load_model`), Option C for pickle/joblib integration. Option B is unnecessary if pickle works.

2. **What metadata needs saving**: Beyond `_artifact_bytes`, the fitted state includes:
   - `_continuous_feature_mins: list[float]`
   - `_continuous_feature_maxs: list[float]`
   - `_continuous_feature_sorted_values: list[list[float]]`
   - `_n_features_in: int`
   - `_uses_continuous_binning: bool`
   - `_float_thresholds_converted: bool`
   - `_is_fitted: bool`
   - `evals_result_: dict` (training history)
   - All constructor params (learning_rate, max_depth, etc.)
   - Future: `_feature_names` (from Feature Names plan)

3. **Artifact-only save/load**: Should there be a lightweight path that saves/loads just the artifact bytes (for deployment where you don't need to retrain)? Recommendation: yes. `save_artifact(path)` / `load_artifact(path)` for the raw bytes, plus `predict_from_artifact` (which already exists as a static method).

4. **Cross-version compatibility**: Should models saved with version X load in version Y? Recommendation: yes, with best-effort. The artifact binary format already has a version field. Python metadata should also include a version tag.

---

## Architecture Overview

### Current Fitted State

After `fit()`, `GBMRegressor` holds:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_artifact_bytes` | `bytes` | Serialized model (trees + metadata) |
| `_native_predictor_handle` | `NativePredictorHandle` | Rust-side predictor (not serializable) |
| `_continuous_feature_mins` | `list[float]` | Binning thresholds for quantized predict |
| `_continuous_feature_maxs` | `list[float]` | Binning thresholds for quantized predict |
| `_continuous_feature_sorted_values` | `list[list[float]]` | Per-feature sorted unique values |
| `_n_features_in` | `int` | Number of features seen during fit |
| `_uses_continuous_binning` | `bool` | Whether continuous binning was used |
| `_float_thresholds_converted` | `bool` | Whether predictor has float thresholds |
| `_is_fitted` | `bool` | Fit status flag |
| `evals_result_` | `dict` | Training loss history |

Plus all constructor params (needed for `get_params()`).

The `_native_predictor_handle` is a PyO3 object wrapping a Rust `Predictor` -- it's reconstructed from `_artifact_bytes`, so it doesn't need to be serialized.

---

## Phase 1: Pickle Support (`__getstate__` / `__setstate__`)

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 1.1: Implement `__getstate__`

```python
def __getstate__(self):
    state = self.__dict__.copy()
    # Remove non-serializable Rust objects
    state.pop('_native_predictor_handle', None)
    return state
```

The `_native_predictor_handle` is a PyO3 object that can't be pickled. Everything else is plain Python (bytes, lists, floats, bools, dicts).

#### Step 1.2: Implement `__setstate__`

```python
def __setstate__(self, state):
    self.__dict__.update(state)
    self._native_predictor_handle = None
    self._float_thresholds_converted = False
    # Predictor will be lazily reconstructed on next predict() call
```

The predictor handle is reconstructed from `_artifact_bytes` when `predict()` is first called. The current `predict()` code already handles `_native_predictor_handle is None` by calling the artifact-based prediction path. Verify this fallback works correctly.

#### Step 1.3: Verify lazy predictor reconstruction

In `predict()`, if `_native_predictor_handle` is None after deserialization, the code should reconstruct it:
```python
if self._native_predictor_handle is None:
    self._native_predictor_handle = _create_native_predictor(self._artifact_bytes)
```

Check if this path already exists. If not, add it.

### Success Criteria

- `pickle.dumps(model)` / `pickle.loads(model)` roundtrips correctly
- `joblib.dump(model, path)` / `joblib.load(path)` works
- Predictions from loaded model match original model exactly
- `get_params()` returns correct values after loading

### Complexity: Very Low (~20-30 lines)

---

## Phase 2: Explicit `save_model()` / `load_model()`

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 2.1: `save_model(path)`

```python
def save_model(self, path: str) -> None:
    """Save the fitted model to a file."""
    if not self._is_fitted:
        raise ValueError("Model must be fitted before saving")
    import json
    metadata = {
        "version": __version__,
        "params": self.get_params(),
        "n_features_in": self._n_features_in,
        "uses_continuous_binning": self._uses_continuous_binning,
        "continuous_feature_mins": self._continuous_feature_mins,
        "continuous_feature_maxs": self._continuous_feature_maxs,
        "continuous_feature_sorted_values": self._continuous_feature_sorted_values,
        "evals_result": getattr(self, 'evals_result_', None),
    }
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_len = len(metadata_json)

    with open(path, 'wb') as f:
        # Header: 4 bytes magic + 4 bytes metadata length
        f.write(b'AGBP')  # AlloyGBM Python model
        f.write(metadata_len.to_bytes(4, 'little'))
        f.write(metadata_json)
        f.write(self._artifact_bytes)
```

#### Step 2.2: `load_model(path)` (classmethod)

```python
@classmethod
def load_model(cls, path: str) -> 'GBMRegressor':
    """Load a model from a file."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'AGBP':
            raise ValueError("Not a valid AlloyGBM model file")
        metadata_len = int.from_bytes(f.read(4), 'little')
        metadata_json = f.read(metadata_len)
        artifact_bytes = f.read()

    metadata = json.loads(metadata_json)
    model = cls(**metadata['params'])
    model._artifact_bytes = artifact_bytes
    model._n_features_in = metadata['n_features_in']
    model._uses_continuous_binning = metadata['uses_continuous_binning']
    model._continuous_feature_mins = metadata.get('continuous_feature_mins')
    model._continuous_feature_maxs = metadata.get('continuous_feature_maxs')
    model._continuous_feature_sorted_values = metadata.get('continuous_feature_sorted_values')
    model.evals_result_ = metadata.get('evals_result')
    model._is_fitted = True
    model._native_predictor_handle = None
    model._float_thresholds_converted = False
    return model
```

### Success Criteria

- `model.save_model('model.agbm')` creates a valid file
- `GBMRegressor.load_model('model.agbm')` restores the model
- Predictions match original
- File is self-contained (no external dependencies for loading)

### Complexity: Low (~60-80 lines)

---

## Phase 3: Artifact-Only Save/Load (Lightweight)

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 3.1: `save_artifact(path)`

```python
def save_artifact(self, path: str) -> None:
    """Save raw model artifact bytes to a file."""
    if not self._is_fitted:
        raise ValueError("Model must be fitted before saving artifact")
    with open(path, 'wb') as f:
        f.write(self._artifact_bytes)
```

#### Step 3.2: `predict_from_artifact` (already exists as static method)

The static method `predict_from_artifact` already exists but may need polish. Verify it works with just artifact bytes and raw float inputs.

#### Step 3.3: Expose `artifact_bytes` property

```python
@property
def artifact_bytes(self) -> bytes:
    """Return the raw model artifact bytes."""
    if not self._is_fitted:
        raise ValueError("Model must be fitted to access artifact bytes")
    return self._artifact_bytes
```

This allows users to manage artifact bytes directly (e.g., store in a database, send over network).

### Complexity: Very Low (~15-20 lines)

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Pickle | 1 (`regressor.py`) | ~20-30 | Very Low |
| Phase 2: save/load | 1 (`regressor.py`) | ~60-80 | Low |
| Phase 3: Artifact | 1 (`regressor.py`) | ~15-20 | Very Low |

Total: ~95-130 lines, all in `regressor.py`. No Rust changes needed.

---

## Risk Areas

### `_native_predictor_handle` Reconstruction

After deserialization, the Rust predictor handle is `None` and must be reconstructed. The current `predict()` flow may not handle this gracefully if it expects the handle to exist. Audit all code paths in `predict()` that use `_native_predictor_handle` and ensure they fall back to artifact-based prediction or lazily reconstruct the handle.

### `_continuous_feature_sorted_values` Size

For datasets with many features and many unique values per feature, `_continuous_feature_sorted_values` can be large (e.g., 100 features * 10000 unique values * 8 bytes = 8 MB). This is serialized as JSON in the metadata, which is verbose for float arrays. Consider:
- Using a binary encoding for this field
- Compressing the metadata section
- Or accepting the size overhead for simplicity

### Version Compatibility

If the `GBMRegressor` constructor signature changes (new parameters added), `load_model` with `**metadata['params']` could fail if the saved params include unknown keys. Use `**{k: v for k, v in metadata['params'].items() if k in cls.__init__.__code__.co_varnames}` or similar filtering.

Simpler approach: catch `TypeError` from unknown params and warn rather than fail.

### Training History

`evals_result_` is a dict of lists. It serializes cleanly to JSON. But if the format changes, old saved models' training history might not match the new format. This is low-risk since training history is informational only.

---

## Testing Strategy

1. **Pickle roundtrip**: `model2 = pickle.loads(pickle.dumps(model1))`, verify `model2.predict(X) == model1.predict(X)`
2. **Joblib roundtrip**: `joblib.dump(model, path); model2 = joblib.load(path)`, same verification
3. **save_model/load_model**: `model.save_model('test.agbm'); model2 = GBMRegressor.load_model('test.agbm')`, same verification
4. **Artifact bytes**: `model.save_artifact('test.bin')`, load bytes, predict via `predict_from_artifact`
5. **Unfitted model**: Verify `save_model` raises error on unfitted model
6. **get_params preservation**: `model2.get_params() == model1.get_params()`
7. **Cross-session**: Save in one Python session, load in another

---

## Non-Goals

- **ONNX/PMML export**: Standard model interchange formats. Separate initiative.
- **Model compression**: Quantizing the model for smaller file size.
- **Encrypted model files**: Security concern, separate from persistence.
- **Cloud storage integration**: S3/GCS save/load. Users can handle this with the bytes API.
