# AlloyGBM Phase 1 Program Plan (0.0.0 to 1.0.0)

## Summary
- Scope: `0.0.0 -> 1.0.0` (CPU production baseline).
- `0.0.0` is contracts-first: interfaces, model format contract, and binding scaffolding before full algorithmic depth.
- Primary user: Python-first quant workflows, with Rust as the engine core.
- Defaults: deterministic training, Linux+macOS CPU support, Python `3.10-3.13`, versioned binary model format + JSON metadata.
- Ranking is architecture-ready but delivery starts in `1.1.0`.

## Core Interfaces and Boundaries
- Crates: `core`, `engine`, `backend_cpu`, `predictor`, `shap`, `categorical`, `bindings_python`.
- `core`: shared configs/types, dataset schema, feature metadata, model metadata/versioning.
- `engine`: backend-agnostic training loop and objective wiring.
- `backend_cpu`: histogram/split primitives for CPU execution.
- `predictor`: low-latency inference path with flat tree layout.
- `shap`: TreeSHAP APIs (global + per-row).
- `categorical`: target encoding and frequency/count encoding with leakage-safe behavior.
- `bindings_python`: `alloygbm.GBMRegressor` API surface.

## Milestones

### 0.0.0 - Repo + Interfaces + Data Model
- Build workspace + crate scaffolding.
- Define shared types and trait contracts.
- Define model format v1 draft and metadata schema.
- Add Python binding skeleton and CI build plumbing.

Exit criteria:
- All crates compile.
- Model metadata/version roundtrip tests pass.
- Python wheels build/import on Linux and macOS.

### 0.1.0 - Minimal Histogram GBDT (CPU, Regression)
- Implement regression objective, histogram tree growth, and early stopping.
- Support row/column subsampling and learning-rate shrinkage.
- Deliver row/batch prediction path.

Exit criteria:
- Correctness fixtures pass.
- Baseline regression quality beats naive predictor.
- Performance target: within `3-5x` LightGBM CPU on selected dense tasks.

### 0.2.0 - Python Wrapper (Sklearn-Compatible Core)
- Complete `fit`, `predict`, `get_params`, `set_params`.
- Support NumPy, pandas, and Polars inputs.
- Finalize packaging with maturin wheels.

### 0.3.0 - Finance Evaluation + Leakage Guardrails
- Metrics: RMSE, MAE, R2, correlation, rank-IC, hit-rate.
- Add Purged K-Fold and embargo split tooling with time/group awareness.

### 0.4.0 - CPU Kernel Optimization + SIMD
- Optimize hist kernels, threading, memory access.
- Add AVX2 path and scalar fallback.
- Performance target: `~1.5-2x` LightGBM CPU on target dense workloads.

### 0.5.0 - Model IO + Predictor Integration
- Make predictor crate the canonical inference path.
- Freeze model format v1 compatibility policy ahead of 1.0.

### 0.6.0 - Categorical Support v1
- Leakage-safe target encoding + frequency/count encoding.
- Integrate categorical pipeline into training/inference metadata.

### 0.7.0 - TreeSHAP CPU
- Exact TreeSHAP for regression trees.
- Global and per-row explanation APIs in Rust and Python.

### 0.8.0 - Release Candidate Hardening
- Expand tests, docs, and benchmark reproducibility artifacts.
- Finalize migration notes and compatibility checks.

### 0.9.0 - Debugging, Benchmark Improvement, and Documentation
- Benchmark Expansion: Expand benchmark to include both shallow and deep runs
- Debugging: Fix any bugs found during testing.
- Benchmark Improvement: Improve benchmark performance (accuracy and speed) to compete with lightgbm & xgboost.
- Documentation: Improve documentation and add tutorials.

### 1.0.0 - CPU Production Baseline
- Stable CPU release with predictor, SHAP, categorical support, and robust Python API.
- Release gates: correctness suite green, stable format, performance target met.

## Testing and Release Gates
- Deterministic mode must reproduce model bytes under fixed seeds.
- Serialization roundtrip and compatibility tests must pass.
- Public benchmark suite: synthetic panel + UCI + LETOR.
- Wheel install/import/tests must pass on Linux + macOS across Python `3.10-3.13`.

## Assumptions Locked
- Python package name: `alloygbm`.
- License: dual MIT/Apache-2.0.
- Device scope for this phase: CPU only.
- Ranking user-facing API starts in `1.1.0`.
