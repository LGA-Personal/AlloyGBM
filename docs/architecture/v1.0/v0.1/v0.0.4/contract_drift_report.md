# v0.0.4 Contract Drift Report

## Scope
- Layer: `docs/architecture/v1.0/v0.1/v0.0.4`
- Plan reference: `docs/architecture/v1.0/v0.1/v0.0.4/plan.md`
- Check date: 2026-02-23

## Planned Contract Baseline (v0.0.4)
- Iterative multi-round training in `engine`.
- Trained-model representation with row/batch prediction helpers in `engine`.
- Initial model artifact binary writer/reader contracts in `core`.
- Engine export/import via core artifact contracts.
- Keep scope minimal (stump-level, deterministic, no predictor integration).

## Findings

### 1) Naming/path drift
- Severity: Low
- Status: Open
- Planned: Layer/docs should reflect current implementation slice.
- Observed:
  - `bindings/python/alloygbm/__init__.py` docstring still says `v0.0.3`.
  - `bindings/python/alloygbm/regressor.py` docstring still says `v0.0.3`.
  - `bindings/python/tests/test_regressor_contract.py` module docstring still says `v0.0.3`.
- Impact: No runtime or contract breakage, but versioned naming context is stale and can mislead future layer audits.
- Recommendation: Update docstrings to current layer-neutral wording (or `v0.0.4` if explicitly versioned).

## Category Check Summary

### dependency-direction drift
- Result: No drift detected.
- Evidence:
  - `crates/engine/Cargo.toml` depends on `alloygbm-core` only.
  - `crates/backend_cpu/Cargo.toml` depends on `alloygbm-core` + `alloygbm-engine`, matching trait-implementation direction.
  - `crates/predictor/Cargo.toml`, `crates/shap/Cargo.toml`, and `crates/categorical/Cargo.toml` depend on `alloygbm-core` only.
  - No reverse dependency from `core` or `engine` to `backend_cpu` found in source.

### public API drift
- Result: No drift detected.
- Evidence:
  - `crates/engine/src/lib.rs` exposes iterative API via `Trainer::fit_iterations(...)`.
  - `crates/engine/src/lib.rs` exposes trained-model prediction via `TrainedModel::predict_row(...)` and `TrainedModel::predict_batch(...)`.
  - `crates/engine/src/lib.rs` retains one-round path (`fit_one_round`) and keeps v0.0.4 scope bounded (stump-oriented model representation).

### artifact contract drift
- Result: No drift detected.
- Evidence:
  - `crates/core/src/lib.rs` defines artifact contracts and helpers:
    - `serialize_model_artifact_v1(...)`
    - `deserialize_model_artifact_v1(...)`
    - `ModelBinaryHeader` / `ModelSectionDescriptor` / `ModelIoContractV1`
  - `crates/engine/src/lib.rs` uses core artifact contracts for export/import:
    - `TrainedModel::to_artifact_bytes(...)`
    - `TrainedModel::from_artifact_bytes(...)`
  - Section kind usage aligns with minimal v0.0.4 payload (`ModelSectionKind::Trees`).

## Conclusion
- Contract alignment for `v0.0.4` is intact across dependency direction, public APIs, and artifact contracts.
- One low-severity naming/path drift exists in Python docstrings and should be cleaned up for documentation consistency.
