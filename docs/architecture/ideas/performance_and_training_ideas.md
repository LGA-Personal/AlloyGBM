# AlloyGBM Idea Backlog: Fundamental vs Variant

## Purpose
Capture improvement ideas for CPU competitiveness and continuous-feature quality, and label each as either:

- **Fundamental**: core architecture/training-path change that materially redefines baseline behavior.
- **Variant**: optional strategy/mode that can be toggled or configured without redefining the core pipeline.

This is an idea catalog only (no implementation commitment in this document).

## Idea Catalog

| Area | Idea | Type | Why It Matters | Configurability |
| --- | --- | --- | --- | --- |
| Continuous features | Train-time quantile cut construction per feature with persisted cut points (train + predict parity) | Fundamental | Aligns split search with LightGBM/XGBoost-style histogram training; large speed/accuracy leverage. | Bin budget (for example 64/128/256/512), min samples per bin. |
| Continuous features | Weighted quantile sketch (streaming/distributed-friendly cut construction) | Fundamental | Scales quantile quality with large data and weighted rows while controlling memory. | Sketch epsilon, max sketch size. |
| Continuous features | Adaptive bin budget by feature (more bins for high-entropy/high-gain features) | Variant | Avoids over-allocating bins to low-value features while preserving fidelity where needed. | Per-feature max bins, entropy/gain thresholds. |
| Split search | Exact greedy split search as an optional reference mode | Variant | Useful as a correctness/quality oracle on smaller datasets; not ideal as default due to cost. | `split_method=exact|hist`. |
| Split search | Hybrid split mode (exact on top-k features, hist on the rest) | Variant | Can recover quality in sensitive features without full exact-search cost. | `exact_top_k_features`, feature ranking method. |
| Base learner | Piece-wise linear trees (linear models in leaves) with half-additive fitting | Fundamental | Improves local trend modeling and extrapolation versus constant leaves, especially for time-series-style targets. | `leaf_model=constant|linear`, ridge/L2, max regressors, half-additive enable flag. |
| Histogram engine | PL-adapted histograms (store additional sufficient statistics for linear-leaf fitting) | Fundamental | Enables scalable linear-leaf training without repeated dense-data rescans; required for production PL-tree throughput. | Stats granularity and compression strategy, fallback to dense scan for validation mode. |
| Histogram engine | Parent/child histogram reuse and subtraction optimizations | Fundamental | Cuts repeated work during tree growth; common high-impact optimization in GBDT engines. | Node-size threshold for reuse/subtraction path. |
| Sampling | Gradient-based row sampling (GOSS/MVS-style) | Variant | Better gradient signal per compute unit than uniform subsampling on noisy tasks. | Sampling fraction, high-gradient retention ratio. |
| Feature handling | Exclusive Feature Bundling (EFB) for sparse/high-dimensional features | Variant | Reduces effective feature count and histogram cost in sparse settings. | Bundle conflict threshold, max bundle size. |
| Tree growth policy | Depth-wise vs leaf-wise growth as selectable policy | Variant | Different workloads prefer different growth shape; can improve speed or accuracy depending on regime. | `growth_policy=depthwise|leafwise`, leaf constraints. |
| Apple Silicon systems | Dense matrix layout and histogram buffers optimized for cache locality (SoA/blocked layouts, compact bins) | Fundamental | On M-series, memory traffic and cache behavior dominate many kernels. | Block/tile size, bin dtype (`u8`/`u16`). |
| Apple Silicon systems | Thread-local histograms + reduction with false-sharing avoidance | Fundamental | Improves scaling and reduces coherence traffic in multi-threaded histogram build. | Threads, shard size, reduction strategy. |
| Apple Silicon systems | Dynamic thread scheduling tuned for bandwidth saturation | Variant | Prevents oversubscription on bandwidth-bound kernels. | Max threads by dataset/kernel profile. |
| SIMD | Portable SIMD-friendly kernel loops (auto-vectorization first) | Variant | Keeps code portable and maintainable while unlocking compiler vectorization wins. | Compile flags/feature gates per target. |
| SIMD | ARM NEON intrinsics in proven hotspots | Variant | Gives explicit control where auto-vectorization underperforms. | Target-feature gating, kernel-specific toggles. |
| SIMD infra | SIMD abstraction layer (for example internal trait layer) | Variant | Can reduce duplication if multiple explicit SIMD paths are maintained. | Backend/kernel registration policy. |
| Memory system | Manual prefetch hints in specific kernels | Variant | Can help on select streaming patterns, but often neutral/negative if misused. | Prefetch distance, kernel-level enable flag. |
| Task-specific modes | Retrieval-oriented discretization/index mode for nearest-history retrieval workflows | Fundamental (for retrieval mode) | Required if Alloy adds retrieval-native GBDT workflows; distinct data-path assumptions. | Index granularity, retrieval depth/window. |
| Task-specific modes | Supervised discretization for explainability/rule extraction (for example MDL-style intervals) | Variant | Useful for interpretability pipelines; not required for base trainer. | Discretizer choice, max intervals, stopping criteria. |
| Benchmark governance | Tiered benchmark policy (must-pass core scenarios + informational extended scenarios) | Variant | Prevents optimization churn while keeping release gating clear and auditable. | Thresholds by scenario/profile/hardware class. |
| Tuning workflow | Offline autotuning of core hyperparameters for benchmark profiles | Variant | Improves competitiveness without hardcoding one-size-fits-all defaults. | Search budget, objective weighting (RMSE vs time). |

## Suggested Priority Order (Near-Term)

1. Quantile cut construction with persisted cut points.
2. Histogram reuse/subtraction plus cache-aware memory layout improvements.
3. Thread-local histogram + reduction scaling on Apple Silicon.
4. Add optional exact reference mode for regression-quality verification.
5. Layer in variants (adaptive bins, GOSS/MVS, growth policy choices) behind explicit config.

## Notes on Interpretation

- A **Fundamental** idea is not automatically "better" in every metric; it means it changes the baseline architecture.
- A **Variant** idea may outperform the default on some scenarios and underperform on others; variants should remain benchmark-driven and configurable.

## Experiment Notes (2026-03-03)

- Variant tested: `continuous_binning_strategy=rank` against `linear` baseline across full benchmark matrices.
- Outcome: keep as configurable variant; do not promote to default in current state.
- Reason: rank-mode speed regressions were significant on the full matrix (especially `histogram_stress`) despite mixed/near-parity accuracy.
- Evidence: see `docs/architecture/benchmarks/regression_report.md`.
- Variant tested: `continuous_binning_strategy=quantile` with `continuous_binning_max_bins` sweep (`64/128/256`).
- Outcome: keep as configurable experimental variant; do not promote to default in current state.
- Reason: ultra-profile speed gains were observed, but full-matrix quality regressed materially on `histogram_stress` (RMSE increase).
- Fundamental tested: parallel histogram building across feature tiles (CPU backend).
- Outcome: keep and proceed.
- Reason: training fit-time improved materially in A/B checks (largest gain on `histogram_stress`) with no observed accuracy drift.
- Variant tested: `leaf_model=linear` (stage-1 piece-wise linear leaf terms on split feature).
- Outcome: keep as configurable experimental variant; do not promote to default in current state.
- Reason: benchmark deltas were mixed with severe regression clusters on `panel_time_series` and `histogram_stress`; competitiveness against LightGBM/XGBoost did not improve.

## Suggested Staging For PL Trees

1. Add a low-risk variant path first: `leaf_model=linear` with strict regressor cap and strong regularization.
2. Implement ancestor-feature-only regressor selection to bound leaf-model dimensionality.
3. Add half-additive fitting (collapse prior regressors into one synthetic feature) for fixed-cost updates.
4. Extend histogram payload with PL-relevant statistics and benchmark memory/runtime deltas before defaulting anywhere.
