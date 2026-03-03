# Project Roadmap: GPU-First Financial Gradient Boosting in Rust (CPU → CUDA → Metal/MLX)

**Goal:** Build a *best-in-class* gradient boosting package optimized for **financial tabular time series** and **ranking**, with a **Rust core** and **Python bindings**, delivered in phases:

- **1.x**: CPU-first (correctness + competitive speed)
- **2.x**: CUDA-first (production GPU on NVIDIA)
- **3.x**: Metal/MLX-first (Apple Silicon GPU)

Design principles:

1. **Correctness before performance**: every accelerator backend must match CPU reference within tolerances.
2. **Backend-agnostic core**: algorithms expressed once; backends implement primitives (histograms, scans, reductions).
3. **Finance-native**: time-aware validation, leakage-safe training options, uncertainty outputs, ranking, constraints.
4. **Production ergonomics**: reproducibility, deterministic training modes, robust serialization, stable Python API.
5. **Low-latency inference**: prediction path is a first-class concern from the beginning.
6. **Interpretability**: built-in TreeSHAP support for trust and model debugging.
7. **Panel-aware categorical handling**: finance-relevant categorical features supported early.

---

# Cross-Cutting Engineering (Consolidated) (Added Requirements)

## A. Dedicated Inference Engine (Predictor Crate)

A separate crate:

```
predictor/
```

Goals:

- Microsecond-level latency
- No training dependencies
- Minimal memory footprint
- SIMD-friendly traversal
- Batch + single-row optimized paths
- CPU and GPU inference backends (later)

Design:

- Flat tree representation (array-based nodes)
- Branchless traversal where possible
- Precomputed thresholds and feature indices
- Cache-aligned node storage
- Optional quantized leaf outputs (f16)

Use cases:

- Real-time trading signals
- Portfolio scoring
- Embedded inference
- Latency-sensitive environments

---

## B. Built-in TreeSHAP Feature Importance

Feature importance is mandatory for quant adoption.

Implementation targets:

- Exact TreeSHAP for CPU predictor
- GPU TreeSHAP later (CUDA / Metal optional)
- Global importance + per-sample explanations

Rust advantages:

- Memory safety with recursion-heavy algorithms
- Strong performance via iterative implementations

API:

```
model.shap_values(X)
model.feature_importances(method="shap")
```

Optional:

- Interaction SHAP values
- Fast approximate SHAP mode

---

## C. Early Categorical Feature Support (Finance-Oriented)

Unlike general ML, finance panel data often includes:

- ticker\_id
- sector
- exchange
- country
- strategy group

These are essential.

Early support should include:

### Target Encoding (Primary)

- Mean encoding with smoothing
- Time-aware encoding to avoid leakage
- Per-fold encoding support

### Frequency / Count Encoding

- Useful for sparse identifiers

### Optional Learned Embeddings (Later)

- Small embedding layers trained jointly or precomputed

Design principle: Categorical handling should integrate into binning pipeline.

---

# 0.x.x — Foundations and Reference Implementation (Pre-1.0)

## 0.0.0 — Repo + Interfaces + Data Model

Responsibility boundaries: `core/` defines public types, configs, and serialization contracts; `engine/` contains the training algorithms and tree construction logic; `backend_*` crates implement hardware-specific primitives (CPU/CUDA/Metal) behind a common trait; `predictor/` is a minimal, dependency-light inference path; `shap/` provides explanation algorithms; `categorical/` handles encoding pipelines; and `bindings_python/` exposes the stable Python API.

Crates:

- core/
- engine/
- backend\_cpu/
- predictor/
- shap/
- categorical/
- bindings\_python/

Canonical dataset supports:

- numeric dense
- categorical encoded
- group ids
- timestamps

---

## 0.1.0 — Minimal Histogram GBDT (CPU), Regression Only

**Deliverables**

- Histogram-based CART growth:
  - depth-limited
  - min data in leaf
  - L2 regression objective
- Training loop:
  - shrinkage (learning rate)
  - row subsampling
  - column subsampling
  - early stopping on validation set
- Basic inference:
  - single-row prediction
  - batch prediction

**Performance targets**

- Within **3–5×** LightGBM CPU on dense numeric tasks (early stage, correctness-focused).

---

## 0.2.0 — Python Wrapper (Sklearn-Compatible)

**Deliverables**

- `fit/predict/predict_proba` stubs (proba later)
- `get_params/set_params`, `feature_importances_`
- Accepts `numpy.ndarray` and `pandas.DataFrame`
- Validation helpers:
  - `TimeSeriesSplit`-friendly behavior
  - explicit `eval_set=[(X_val, y_val)]` like LightGBM/XGBoost

**Release discipline**

- Semantic-ish versioning begins (0.x = breaking allowed)
- CI: Linux + macOS, Python wheels (x86\_64 + arm64)

---

## 0.3.0 — Finance-Grade Evaluation + Leakage Guardrails

**Deliverables**

- Built-in evaluation metrics commonly used in quant:
  - regression: RMSE, MAE, R²
  - finance: correlation, rank-IC, ICIR, hit-rate, tail metrics (optional)
- **Time-aware validation helpers**:
  - Purged K-fold and embargo options (for labeling overlap)
  - Group/time index support for panel data (asset\_id × date)

**Core stance**

- You don’t “bake in” purging/embargo into the model; you provide **tooling** that makes correct evaluation the default.

---

## 0.4.0 — Fast CPU Kernels + SIMD

**Deliverables**

- CPU histogram kernel tuned for:
  - binned features
  - dense blocks
  - multi-threading (rayon or custom threadpool)
- SIMD optimization:
  - AVX2 baseline; AVX-512 optional
- Memory:
  - aligned allocations
  - feature blocking / tiling

**Performance targets**

- Within **\~1.5–2×** LightGBM CPU on targeted dense numeric benchmarks.

---

## 0.5.0 — Model IO + Predictor Integration

Predictor crate becomes production-ready.

---

# 1.x.x — CPU-First Production Release + Quant Innovations

## 1.0.0 — CPU Production Baseline

**Scope**

Includes:

- Predictor crate
- TreeSHAP CPU
- Target encoding
- High-quality CPU GBDT for dense numeric data with robust Python API.

**Must-have**

- strong tests + reproducibility
- competitive performance
- stable model format

---

## 1.1.0 — Ranking Core

**Deliverables**

- Group-aware dataset (`group_id` for queries / days / universes)
- Pairwise ranking objective:
  - LambdaRank-style gradients with NDCG weighting
- Metrics:
  - NDCG\@K, MAP\@K
- Engineering constraints:
  - efficient pair sampling per group
  - clipping / stabilization for extreme lambdas
  - optional “top-heavy” focus for portfolio selection

**Quant-focused API**

- `fit_ranker(X, y, group, sample_weight=None)`
- `predict_scores(X)` suitable for sorting assets each rebalance

---

## 1.2.0 — Probabilistic Outputs

**Deliverables**

- Probabilistic prediction interface:
  - `predict_mean(X)`
  - `predict_var(X)` or `predict_dist(X)` returning (μ, σ²)
- Leaf statistics track sufficient moments:
  - mean, variance of residuals / targets (as defined by method)
- Calibration layer:
  - isotonic or simple scaling for variance (initially optional)

**Quant value**

- Position sizing, risk-aware ranking, uncertainty filters, scenario generation hooks.

---

## 1.3.0 — Linear Leaves

**Deliverables**

- Leaf models:
  - constant leaf (default)
  - linear leaf (small ridge regression per leaf)
- “Fast linear leaf” strategies:
  - restricted feature subset per leaf (top-k by correlation)
  - incremental updates while splitting (avoid full refits)
  - ridge regularization + stable solvers

**Quant value**

- Better local extrapolation for momentum-like patterns, smoother surfaces.

---

## 1.4.0 — Ordered Boosting Mode

**Deliverables**

- Training mode: `ordered=true`
- Core idea:
  - for each sample, compute residuals from a model that did not train on that sample
- Practical implementation plan (CPU):
  - maintain multiple permutations (small number, e.g., 4–8)
  - incremental training snapshots
  - compute residuals via permutation-consistent prefix models

**Quant value**

- Reduces target leakage effects in sequential/panel settings, improves robustness under drift.

---

## 1.5.0 — Accelerated Boosting

**Deliverables**

- AGBM training option:
  - momentum ensemble + corrected pseudo-residuals
- Strong safeguards:
  - divergence detection
  - fallback to vanilla boosting for stability

**Quant value**

- Faster convergence on noisy objectives, potentially fewer trees for similar performance.

---

## 1.6.0 — Constraints

**Deliverables**

- Monotonic constraints (per-feature)
- Interaction constraints (allow/forbid feature interactions)
- Feasibility checks for finance:
  - enforce monotonicity where economically required (e.g., some option-like surfaces)

---

## 1.7.0 — Sampling & Scaling

**Deliverables**

- Weighted stratified sampling
- Hard-negative sampling for ranking (SelGB-style)
- “Streaming binning” utilities:
  - bin thresholds computed in chunks
- Optional out-of-core dataset reader (memory-mapped)

---

## 1.8.0 — Online Updates

**Deliverables**

- Rolling-window training utilities:
  - warm-start from previous model
  - partial retraining strategies:
    - add trees on new data
    - drop oldest trees (experimental)
- Time-decay weighting built-in:
  - exponential decay of sample weights by timestamp

**Quant value**

- Realistic production loop for daily/intraday refresh.

---

## 1.9.0 — CPU Maturity Release

**Deliverables**

- Tight benchmarking suite:
  - finance-like tabular
  - ranking by day/universe
  - drift stress tests
- Documentation:
  - “How to avoid leakage in finance”
  - “Probabilistic outputs for risk”
  - “Ranking for portfolio construction”

---

# 2.x.x — CUDA Backend

**Core strategy**

- Keep algorithm in `engine/` unchanged.
- Implement GPU primitives in `backend_cuda/`:
  - histogram build
  - prefix scans
  - reductions
  - best-split selection
  - apply split / partition
- Includes GPU TreeSHAP later in series.

**Kernel plan**

- Binning stays CPU or GPU? (phased)
  - 2.0: **CPU binning**, GPU training consumes `X_binned`
  - 2.1: optional **GPU binning** for large throughput pipelines

**Correctness**

- CPU vs CUDA parity tests:
  - identical splits for deterministic mode where possible
  - tolerance-based equivalence for floating point sums

### 2.1.0 — CUDA End-to-End Pipeline + Mixed Precision

**Deliverables**

- GPU-side histogram accumulation in `u32` then convert to `f32` sums
- Mixed precision:
  - `f16`/`bf16` optional for gradients/hessians where safe
- Memory:
  - pinned host memory for transfers
  - feature blocks copied once, reused across iterations
- Batching strategy for many features:
  - stream features in tiles to fit GPU memory

**Performance targets**

- On dense numeric: competitive with XGBoost GPU on similar hardware for your supported subset.

### 2.2.0 — CUDA Ranking + Probabilistic Outputs

**Deliverables**

- Ranking lambdas computed on GPU or CPU?
  - Start CPU lambdas, GPU tree building
  - Later GPU lambdas for scalability
- Probabilistic outputs fully supported on GPU backend

### 2.3.0 — Distributed GPU (Optional, Only If Needed)

**Deliverables**

- Multi-GPU single-node scaling (NCCL-based) for hist aggregation
- Keep scope tight; only add if you truly need it.

### 2.4.0 — CUDA Maturity Release

**Deliverables**

- Profiling + kernel fusion opportunities
- Robust fallback paths:
  - GPU OOM → feature tiling
  - compute capability detection
- Documentation:
  - best practices for CUDA performance on tabular finance

---

# 3.x.x — Metal / MLX Backend

### 3.0.0 — Metal Compute Backend for Core Primitives

**Reality check**

- MLX is evolving; the lowest-risk plan is:
  - Implement kernels using **Metal compute** first
  - Optionally integrate MLX as a frontend or tensor runtime later

**Deliverables**

- `backend_metal/` implementing:
  - histogram build kernel
  - reductions
  - split selection
  - partition/apply-split
  - Includes Metal inference path and SHAP acceleration.

**Apple-specific advantage**

- Unified memory: minimize explicit copies
- Use “shared” buffers where possible
- Exploit high bandwidth and low transfer overhead

### 3.1.0 — MLX Integration Layer (Optional but Strategic)

**Deliverables**

- Provide an MLX-friendly API path:
  - accept MLX arrays (or convert with minimal copying)
- Potential route:
  - keep training kernels in Metal
  - use MLX for pipeline interoperability (preprocessing / feature transforms)

### 3.2.0 — Metal Performance Pass + Apple-Optimized Tiling

**Deliverables**

- Feature tiling tuned for Apple GPU execution
- Kernel optimizations:
  - minimize branching
  - coalesced memory
  - use threadgroup memory for partial histograms then reduce

**Targets**

- On M-series laptops: meaningful speedups vs CPU baseline (e.g., 3–10× on certain workloads)

### 3.3.0 — Metal/MLX Full Feature Parity (CPU + CUDA)

**Deliverables**

- Ranking support
- Probabilistic outputs
- Constraints
- Ordered boosting mode (if feasible; may remain CPU-only fallback initially)

### 3.4.0 — “Mac Quant” Polished Release

**Deliverables**

- Simple install (`pip install ...`) with universal2 wheels
- Great docs:
  - “Fast boosting on Apple Silicon”
  - finance examples (portfolio ranking, volatility-aware filters)
- Benchmarks published:
  - CPU vs Metal vs CUDA
  - accuracy and calibration comparisons
  - drift robustness tests


---

# Cross-Cutting Engineering: What Must Exist Throughout

## A) Backend Abstraction (Non-Negotiable)

Core trait (conceptually):

- `build_histograms(binned_X, grad, hess, node_row_idx, feature_tiles) -> hist`
- `best_split(hist) -> split`
- `apply_split(...) -> (left_idx, right_idx)`
- `reduce_sums(...)`, `scan_prefix(...)`

Design goal: algorithm changes (ranking, probabilistic, ordered boosting) should not require rewriting GPU kernels—only how gradients/hessians are formed and how leaf values are computed.

---

## B) Determinism Modes

- `deterministic=true`: stable split tie-breaking, reproducible sampling, fixed parallel ordering where possible
- `fast=true`: allows non-deterministic reductions for speed on GPU

Finance often cares about reproducibility—offer both.

---

## C) Benchmark Suite (Finance-Native)

Include:

- dense panel regression (asset×time)
- ranking by day/universe (NDCG@K + top-k precision)
- drift stress tests (train early years, test late years)
- latency test for inference batching

Track:

- wall-clock train time
- throughput (rows/s, features/s)
- peak memory
- calibration quality (for probabilistic)

---

## D) Packaging and Distribution

- Rust core compiled into Python wheels via `maturin`
- Separate optional extras:
  - `package[cuda]`
  - `package[metal]` (or auto-detect on macOS)
- Clear device selection API:
  - `device="cpu" | "cuda" | "metal"`

---

## E) Practical API Design (Python)

Keep it boring and familiar:

- `GBMRegressor`
- `GBMRanker`
- `predict`, `predict_mean`, `predict_std`
- `save_model`, `load_model`
- callbacks/logging similar to LightGBM

Add finance convenience, but don’t force it:

- `time_index=...`
- `group=...`
- `purge_gap=...`, `embargo=...` in evaluation tools

---

# Suggested “Killer Feature” Bundles by Major Version

## 1.x (CPU): “Finance-Grade Booster”

- leakage-aware evaluation tooling
- ranking
- probabilistic mean/variance
- optional linear leaves
- optional ordered boosting mode

## 2.x (CUDA): “Fast GPU Booster for Production”

- GPU histogram training
- mixed precision
- ranking + probabilistic
- robust memory tiling

## 3.x (Metal/MLX): “Mac Quant Booster”

- Metal backend + unified memory advantages
- MLX interoperability
- one-command install, excellent docs, strong benchmarks

---

# Release Gates (When You’re Allowed to Ship Each Major)

## 1.0.0 gate

- passes correctness suite
- stable model format
- within ~1.5–2× CPU performance of LightGBM on target workloads

## 2.0.0 gate

- CUDA backend matches CPU within tolerances
- beats CPU baseline materially on NVIDIA GPUs
- clean fallback behavior

## 3.0.0 gate

- Metal backend matches CPU within tolerances
- meaningful speedup on M-series
- packaging works smoothly on macOS arm64

---

# Final Note: What to Not Do Early

Avoid until after 1.0 unless absolutely needed:

- sparse matrix support
- categorical handling (CatBoost-like)
- distributed multi-node training
- exotic generative boosting (flows / energy-based)

They’re cool, but they can wait.

---

# Release Philosophy

Major versions correspond to backend maturity:

- 1.x CPU
- 2.x CUDA
- 3.x Metal

---

# What Is Explicitly Prioritized (Updated)

Now included early:

- SHAP interpretability
- Low-latency inference
- Categorical features

Still deferred:

- Distributed multi-node
- Generative boosting
- Exotic architectures

---

# Strategic Positioning

End-state vision:

> The first GPU-native, finance-specialized gradient boosting system with strong interpretability and production latency guarantees.

Potential differentiators:

- Probabilistic outputs
- Ranking for portfolios
- Apple GPU acceleration
- Rust safety + speed
- Microsecond inference path

---

If extended fully, this system could represent a new class of boosting framework optimized specifically for financial modeling and quantitative research workflows.

