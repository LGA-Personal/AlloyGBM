# v0.1 Contract Drift Report

## Scope
- Layer: `docs/architecture/v1.0/v0.1`
- Plan reference: `docs/architecture/v1.0/v0.1/plan.md`
- Ancestor reference: `docs/architecture/v1.0/plan.md`
- Check date (UTC): 2026-02-25

## Planned Contract Baseline (v0.1)
- Workspace boundaries include `core`, `engine`, `backend_cpu`, `predictor`, `shap`, `categorical`, and `bindings_python`.
- `engine` defines backend primitive traits and trainer interfaces.
- `core` provides model-format v1 contracts and metadata schema for artifact IO.
- Python layer exposes importable `alloygbm.GBMRegressor` with parameter validation.
- Dependency direction remains backend-agnostic (avoid CPU-specific leakage into `engine`).

## Findings

### 1) Naming/path drift
- Severity: Low
- Status: Open
- Planned: docs consistently refer to Python bindings crate as `bindings_python`.
- Observed:
  - plans/roadmap use `bindings_python` naming (`docs/architecture/v1.0/plan.md`, `docs/architecture/v1.0/v0.1/plan.md`, `docs/architecture/gpu_financial_gbm_roadmap.md`).
  - repository path and build manifest use `bindings/python` (`bindings/python/Cargo.toml`, `.github/workflows/ci.yml` manifest-path).
- Impact: no runtime or contract behavior break, but nomenclature mismatch can create audit friction when mapping docs to filesystem paths.
- Recommendation: normalize docs to mention canonical filesystem path (`bindings/python`) and optionally note alias (`bindings_python`) once.

## Category Check Summary

### dependency-direction drift
- Result: No drift detected.
- Evidence:
  - `crates/engine/Cargo.toml` depends on `alloygbm-core` only.
  - `crates/backend_cpu/Cargo.toml` depends on `alloygbm-core` + `alloygbm-engine` and implements `BackendOps`.
  - `crates/predictor/Cargo.toml`, `crates/shap/Cargo.toml`, and `crates/categorical/Cargo.toml` depend on `alloygbm-core` only.
  - no reverse dependency from `engine` to `backend_cpu` detected.

### public API drift
- Result: No drift detected.
- Evidence:
  - `crates/engine/src/lib.rs` exposes backend/objective contracts (`BackendOps`, `ObjectiveOps`) and trainer interfaces (`fit_one_round`, `fit_iterations` family).
  - Python package exports `GBMRegressor` and native runtime entrypoint (`bindings/python/alloygbm/__init__.py`).
  - `GBMRegressor` constructor and parameter validation are callable (`bindings/python/alloygbm/regressor.py`), aligned with `v0.1` acceptance intent.

### artifact contract drift
- Result: No drift detected.
- Evidence:
  - `crates/core/src/lib.rs` defines model artifact contract types and IO helpers (`ModelIoContractV1`, `serialize_model_artifact_v1`, `deserialize_model_artifact_v1`).
  - `crates/engine/src/lib.rs` uses core artifact contracts for model export/import (`to_artifact_bytes`, `from_artifact_bytes*`).
  - metadata/version roundtrip contract tests are present in core and exercised by workspace tests.

## Conclusion
- `v0.1` implementation remains aligned with planned contract boundaries and dependency direction.
- One low-severity naming/path drift remains open (`bindings_python` vs `bindings/python`) and should be normalized in documentation for consistency.
