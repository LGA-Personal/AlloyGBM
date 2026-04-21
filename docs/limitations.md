# AlloyGBM Current Limitations

Last updated for v0.3.2.

## Remaining Limitations

### 1. Metal Backend is Infrastructural (through Stage 2)

The `BackendOps` trait is designed for hardware abstraction. Two backends
now exist: the default `CpuBackend` and an Apple-Silicon `MetalBackend`
(macOS only, `device="metal"`). **As of v0.3.2 the Metal backend is
Stage 2: histogram construction *and* the per-node best-split finder
run on the GPU; row partitioning, histogram subtraction, and prediction
still execute on the CPU backend.** Categorical splits (Fisher-sort
path) also stay on CPU — mixed continuous + categorical fits delegate
the categorical features and combine candidates on the CPU side.

**Recommendation:** leave `device="cpu"` (the default) for every real
workload. Stage 2 did *not* achieve throughput crossover — the Metal
path is still slower end-to-end than CPU across every benchmarked
configuration. See the M4 numbers in
[docs/metal-backend/BENCHMARKS.md](metal-backend/BENCHMARKS.md):

- On the shape grid (regression, depth 6, 255 bins, 5 estimators),
  Stage 2 whole-fit wall-clock runs **0.03× to 0.25× CPU speed**
  across (10k, 100k, 1M) rows × (10, 100, 1000) features.
- On `metal_friendly` (deep trees up to depth 10, up to 1024 bins,
  multiclass with K=10 histograms per round), Stage 2 still runs
  **0.06× to 0.08× CPU**.

The Stage 2 numbers are within run-to-run jitter of the Stage 1
baseline. The interpretation (see BENCHMARKS.md Stage 2 section):
moving `best_split` onto the GPU without also moving the row
partitioner and histogram residency means every per-node call still
memcpys the `HistogramBundle` to the GPU and reads back a candidate.
At depth 10 with 200 features this is on the order of 25 GiB of
memcpy per fit, plus ~5000 dispatches of ~10–50 μs fixed latency.
The CPU time saved on the split-finder compute is absorbed by the
new dispatch-plus-memcpy tax.

The decisive throughput win is now architecturally gated on
**Stage 3 (GPU row partitioning + Metal 4 Indirect Command Buffers)**,
which keeps histograms resident on-device across levels and
collapses the per-level CPU round-trip that Stage 2 on its own
cannot close. Until Stage 3 lands, `device="metal"` exists to prove
the plumbing (structural-plus-ulp-epsilon parity vs CPU on every
objective, warn-and-fallback, capability probe, pipeline caching,
per-call buffer reuse) and to unblock the remaining stages — not to
deliver throughput.

**Numerical parity.** Stage 2's split kernel uses SIMD
`simd_prefix_inclusive_sum` + block-scan for gain accumulation,
which is a tree reduction (not order-identical to the CPU's
strict-ascending serial sweep). Predictions on the 50k × 100 × 255
× 20 golden test still match CPU bit-exactly, but on tiny shapes
(≤1024 rows) near-tied root-split gains can flip under the ulp
drift — producing macroscopic prediction deltas (~0.1) on ≤0.1% of
rows. See `docs/metal-backend/DECISIONS.md` D-013 for the gate
relaxation details.

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
