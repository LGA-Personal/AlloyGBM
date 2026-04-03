# AlloyGBM - Claude Code Project Guide

## What This Is

AlloyGBM is a Rust-first Gradient Boosted Decision Tree (GBDT) library with Python bindings via PyO3. It's a published package on PyPI.

## Project Structure

```
AlloyGBM/
  Cargo.toml              # Workspace root (7 crates, edition 2024, Rust 1.92.0)
  crates/
    core/src/lib.rs        # Data structures: TrainParams, BinnedMatrix, ModelMetadata, artifact serde
    engine/src/lib.rs      # Training loop, ObjectiveOps trait, Trainer, IterationControls
    backend_cpu/src/lib.rs # Histogram kernels, split finding, partitioning (Rayon parallelism)
    predictor/src/lib.rs   # Prediction from trained artifacts (float threshold + bin-level)
    shap/src/lib.rs        # Exact Shapley values (brute-force, capped at 20 features)
    categorical/src/lib.rs # Target encoding, frequency encoding
  bindings/python/
    src/lib.rs             # PyO3 bridge: 5 training pyfunctions, NativePredictorHandle
    alloygbm/
      regressor.py         # GBMRegressor (sklearn-style interface, ~2400 lines)
      evaluation.py        # Metrics: rmse, mae, r2_score, pearson_correlation, etc.
      validation.py        # Purged time-series and panel cross-validation splits
  docs/
    limitations.md         # Comprehensive limitation analysis (16 items)
    plans/                 # Implementation plans for addressing each limitation
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

# Quick smoke test
.venv/bin/python -c "from alloygbm import GBMRegressor; m = GBMRegressor(n_estimators=3); m.fit([[1],[2],[3]], [1,2,3]); print(m.predict([[2]]))"
```

## Critical Conventions

- **`unsafe_code = "forbid"`** -- no unsafe Rust anywhere in the workspace
- **Edition 2024** with Rust 1.92.0 minimum
- **All training paths converge** to `train_regression_artifact_with_summary_dense_impl()` in the Python bridge
- **Newton-Raphson leaf values**: `leaf = -lr * grad_sum / (hess_sum + lambda + eps)` -- already general-purpose for any objective
- **Hand-rolled JSON serde** for `ModelMetadata` in `core/src/lib.rs` -- positional parser, very brittle. Adding fields requires careful ordering.
- **`BinnedMatrix` uses `Vec<u8>`** -- 256 max bins, column-major duplicate for cache-friendly histograms
- **Feature names are auto-generated** as `f0, f1, ...` in `to_artifact_bytes()`
- **`sample_weights: None` and `group_id: None`** are hardcoded in the Python bridge (the Rust engine supports both)

## Key Architectural Patterns

- **ObjectiveOps trait** (`engine/src/lib.rs:108`): Generic trait with `initial_prediction`, `compute_gradients`, `compute_gradients_into`. Only `SquaredErrorObjective` exists today.
- **BackendOps trait** (`engine/src/lib.rs`): Abstraction over hardware. Only `CpuBackend` exists.
- **Training policy**: Auto mode with dataset-aware heuristics for `min_split_gain`, `min_rows_per_leaf`, regularization. Manual mode uses raw user params.
- **Artifact format**: Binary with magic bytes `AGBM`, versioned sections (Trees, PredictorLayout, CategoricalState), JSON metadata header.
- **Histogram subtraction trick**: Used for child nodes within a level (smaller child built from scratch, larger = parent - smaller).

## Improvement Plans

See `docs/plans/00_conflict_resolution_guide.md` for the master implementation order and conflict resolution strategy. Individual plans are in `docs/plans/`:

| Priority | Plan | File |
|----------|------|------|
| Tier 0 | Model Persistence | `model_persistence.md` |
| Tier 0 | Feature Names | `feature_names.md` |
| Tier 0 | sklearn Compatibility | `sklearn_compatibility.md` |
| Tier 0 | SHAP quick fix | `shap_feature_limit.md` |
| Tier 1 | Sample Weight Support | `sample_weight_support.md` |
| Tier 1 | Group ID Support | `group_id_support.md` |
| Tier 1 | NaN Support | `nan_support.md` |
| Tier 1 | min_split_gain exposure | `expanded_configurability.md` (Feature A) |
| Tier 2 | Classification & Ranking | `classification_and_ranking.md` |
| Tier 2 | Training Metric Tracking | `training_metric_tracking.md` |
| Tier 3 | Configurability (B-G) | `expanded_configurability.md` |
| Tier 3 | Bin Cap Increase | `bin_cap_increase.md` |
| Tier 3 | Leaf-Wise Growth | `leaf_wise_growth.md` |
| Tier 3 | Histogram Caching | `histogram_caching.md` |
| Tier 4 | Multi-Categorical | `multiple_categorical_columns.md` |
| Tier 4 | Warm-Starting | `warm_starting.md` |
| Tier 4 | SHAP TreeSHAP | `shap_feature_limit.md` |

## When Implementing a Plan

1. **Read the plan document first** -- it has questions for the user, phased steps, success criteria, and risk areas
2. **Check `00_conflict_resolution_guide.md`** -- look up your plan in the quick reference table at the bottom to see what conflicts to watch for and what must land first
3. **Run `cargo test --workspace` and `.venv/bin/python -m pytest bindings/python/tests/ -q` before and after** -- the existing test suite must not regress
4. **Commit granularly** -- one commit per phase within a plan, not one giant commit
5. **When adding fields to structs** (TrainParams, IterationControls, etc.) -- add at the end, add a default, add validation
6. **When adding Python parameters** -- update `__init__`, `get_params()`, `set_params()`, `__repr__`, and `_params_order` together
