# AlloyGBM - Claude Code Project Guide

## What This Is

AlloyGBM is a Rust-first Gradient Boosted Decision Tree (GBDT) library with Python bindings via PyO3. It supports regression, binary classification, and learning-to-rank. Published on PyPI as `alloygbm`.

## Project Structure

```
AlloyGBM/
  Cargo.toml              # Workspace root (7 crates, edition 2024, Rust 1.92.0)
  crates/
    core/src/lib.rs        # Data structures: TrainParams, BinnedMatrix, ModelMetadata, artifact serde, NaN handling
    engine/src/lib.rs      # Training loop, ObjectiveOps trait (8 objectives), Trainer, IterationControls
    backend_cpu/src/lib.rs # Histogram kernels, split finding, NaN-aware partitioning (Rayon parallelism)
    predictor/src/lib.rs   # Prediction from trained artifacts (post-transforms: identity, sigmoid)
    shap/src/lib.rs        # TreeSHAP (polynomial-time) + legacy brute-force Shapley values
    categorical/src/lib.rs # Target encoding, frequency encoding (multi-column support)
  bindings/python/
    src/lib.rs             # PyO3 bridge: training pyfunctions for all objectives, NativePredictorHandle
    alloygbm/
      __init__.py          # Public API: GBMRegressor, GBMClassifier, GBMRanker, metrics, validation
      regressor.py         # GBMRegressor (sklearn-compatible, ~3400 lines)
      classifier.py        # GBMClassifier (binary cross-entropy, predict_proba, ClassifierMixin)
      ranker.py            # GBMRanker (5 ranking objectives, group-sorted training)
      evaluation.py        # Metrics: rmse, mae, r2_score, accuracy, log_loss, ndcg, etc.
      validation.py        # Purged time-series and panel cross-validation splits
  docs/
    limitations.md         # Current limitation analysis (v0.2.0)
    plans/                 # Implementation plans (historical, archived copy in docs/archive/v0.1_plans/)
  benchmarks/              # Cross-library comparison (regression, classification, ranking)
```

## Build & Test Commands

```bash
# Rust
cargo check --workspace
cargo test --workspace
cargo clippy --workspace

# Python (requires maturin + virtual env)
maturin develop --release      # Build and install Python extension
.venv/bin/python -m pytest bindings/python/tests/ -q   # Run Python tests

# Quick smoke tests
.venv/bin/python -c "from alloygbm import GBMRegressor; m = GBMRegressor(n_estimators=3); m.fit([[1],[2],[3]], [1,2,3]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMClassifier; m = GBMClassifier(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,0,1,1]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMRanker; m = GBMRanker(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,1,0,1], group=[0,0,1,1]); print(m.predict([[2]]))"
```

## Critical Conventions

- **`unsafe_code = "forbid"`** -- no unsafe Rust anywhere in the workspace
- **Edition 2024** with Rust 1.92.0 minimum
- **Newton-Raphson leaf values**: `leaf = -lr * grad_sum / (hess_sum + lambda + eps)` -- general-purpose for any objective
- **Hand-rolled JSON serde** for `ModelMetadata` in `core/src/lib.rs` -- positional parser, very brittle. Adding fields requires careful ordering.
- **`BinnedMatrix`** uses adaptive `Vec<u8>` or `Vec<u16>` -- up to 65,535 bins, column-major duplicate for cache-friendly histograms
- **Artifact format**: Binary with magic bytes `AGBM`, versioned sections (Trees, PredictorLayout, CategoricalState), JSON metadata header. Includes objective type for post-transform dispatch.

## Key Architectural Patterns

- **ObjectiveOps trait** (`engine/src/lib.rs`): Generic trait with `initial_prediction`, `compute_gradients`, `compute_gradients_into`. Implementations: SquaredError, BinaryCrossEntropy, RankPairwise, RankNdcg, RankXendcg, QueryRmse, YetiRank.
- **BackendOps trait** (`engine/src/lib.rs`): Abstraction over hardware. Only `CpuBackend` exists.
- **Training policy**: Auto mode with dataset-aware heuristics for `min_split_gain`, `min_rows_per_leaf`, regularization. Manual mode uses raw user params.
- **Tree growth**: Level-wise (default) or leaf-wise (best-first) via `tree_growth` parameter.
- **Histogram subtraction trick**: Used for child nodes within a level (smaller child built from scratch, larger = parent - smaller). Histogram buffers are reused across rounds.
- **NaN handling**: Missing values get a dedicated bin; split finding learns the optimal direction for NaN.
- **Model persistence**: Pickle support via `__getstate__`/`__setstate__`, `save_model`/`load_model`, and raw `model_bytes()` export.

## When Implementing Changes

1. **Run `cargo test --workspace` and `.venv/bin/python -m pytest bindings/python/tests/ -q` before and after** -- the existing test suite must not regress
2. **Commit granularly** -- one commit per logical change, not one giant commit
3. **When adding fields to structs** (TrainParams, IterationControls, etc.) -- add at the end, add a default, add validation
4. **When adding Python parameters** -- update `__init__`, `get_params()`, `set_params()`, `__repr__`, and `_params_order` together
5. **When adding a new objective** -- implement `ObjectiveOps`, add a variant to the objective dispatch in `engine`, update the predictor post-transform table, and add Python-side estimator support
