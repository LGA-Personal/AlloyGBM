# AlloyGBM Current Limitations

Last updated for v0.3.2.

## Remaining Limitations

### 1. Metal Backend is Infrastructural (Stage 1)

The `BackendOps` trait is designed for hardware abstraction. Two backends
now exist: the default `CpuBackend` and an Apple-Silicon `MetalBackend`
(macOS only, `device="metal"`). **At v0.3.2 the Metal backend is Stage 1:
only histogram construction runs on the GPU; split finding, row
partitioning, and prediction all still execute on the CPU backend.**

**Recommendation:** leave `device="cpu"` (the default) for every real
workload. The Stage 1 Metal path is slower end-to-end than CPU across
every benchmarked configuration — see the M4 numbers in
[docs/metal-backend/BENCHMARKS.md](metal-backend/BENCHMARKS.md):

- On the shape grid (regression, depth 6, 255 bins, 5 estimators),
  whole-fit wall-clock runs **0.03× to 0.28× CPU speed** across
  (10k, 100k, 1M) rows × (10, 100, 1000) features.
- On the `metal_friendly` scenario (deep trees up to depth 10, up
  to 1024 bins, multiclass with K=10 histograms per round —
  configurations theoretically most favourable to Metal), wall-clock
  still runs **0.06× to 0.09× CPU**. This confirms Stage 1 cannot
  cross break-even on any realistic shape.

This is expected: histogram acceleration alone only pays off once
the histogram phase dominates the inner loop, and every boosting
round currently round-trips through the CPU path for split finding
and partitioning. Dispatch + per-level readback latency dominates
at every shape tested.

The decisive win arrives with Stage 2 (GPU best-split + histogram
subtraction) and Stage 3 (GPU row partitioning + Metal 4 Indirect
Command Buffers), which eliminate the per-level CPU round-trip. Until
those land, `device="metal"` exists to prove the plumbing (bit-exact
histograms vs CPU, warn-and-fallback, capability probe, pipeline
caching) and to unblock the remaining stages — not to deliver
throughput.

**How to detect the backend.** `alloygbm.native_runtime_info()` exposes
three fields for programmatic checks:

- `metal_available: bool` — `True` when a Metal device can be created
  (macOS, `MTLGPUFamilyApple7` or better, default build).
- `metal4_available: bool` — `True` when `MTLGPUFamilyMetal4` is
  supported (Stage 3 fast path; M4 and newer).
- `gpu_family: Optional[str]` — the detected GPU family name, e.g.
  `"Apple7"`, `"Apple8"`, `"Metal4"`, or `None` on non-macOS /
  Metal-disabled builds.

Fitted models record the resolved device in their artifact metadata
(`TrainedModel.trained_device` / `MultiClassTrainedModel.trained_device`)
so the choice round-trips through `save_model`/`load_model`.

**Escape hatch.** Set `ALLOYGBM_METAL_DISABLE=1` in the environment to
force every `device="metal"` / `device="auto"` call to fall back to CPU
with a `RuntimeWarning`. This is how the Metal-init fallback path is
exercised on Metal-capable hardware in the test suite.

Support beyond Apple Silicon (CUDA, ROCm, generic compute) is not
planned.

### 2. No Interaction Constraints

There is no way to constrain which features can interact within the same tree.

### 3. No Dart / GOSS Boosting Modes

Only standard gradient boosting is supported. Dart (dropout) and GOSS
(gradient-based one-side sampling) modes are not available.

### 4. No Multi-Label Ranking

`GBMRanker` supports single-label relevance only.

## Resolved (Previously Limitations)

The following were limitations in prior versions and have been addressed:

- Regression-only (now: classification + ranking)
- Single categorical column only (now: multiple via `categorical_feature_indices`)
- Limited configurability (now: `min_split_gain`, monotone constraints, feature weights, `max_leaves`, leaf-wise growth)
- No NaN support (now: native NaN handling)
- No model persistence (now: pickle, save/load, artifact export)
- No sklearn compatibility (now: `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- No sample weight / group ID from Python (now: fully supported)
- Feature names auto-generated only (now: captured from DataFrames)
- SHAP limited to 20 features (now: TreeSHAP with no practical limit)
- Only RMSE tracked during training (now: objective-aware metric tracking)
- No warm-starting (now: `warm_start=True` for all estimators including multiclass)
- Level-wise growth only (now: leaf-wise available)
- Bins capped at 256 (now: up to 65,535)
- No histogram reuse (now: buffer reuse across rounds)
- Binary classification only (now: multi-class softmax with K > 2 classes)
- No native categorical splits (now: `max_cat_threshold` enables Fisher-sort optimal binary partitions with O(1) bitset prediction)
- No custom objective/metric callbacks (now: `objective` callable and `eval_metric` callable)
- Multiclass warm-start unsupported (now: `warm_start=True` works for multiclass with round-offset continuity)
- Multiclass prediction per-row allocation (now: zero-copy dense slice prediction)
