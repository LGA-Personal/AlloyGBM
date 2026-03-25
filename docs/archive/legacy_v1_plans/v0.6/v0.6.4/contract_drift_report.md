# AlloyGBM v0.6.4 Contract Drift Report

## Layer
- Path: `docs/architecture/v1.0/v0.6/v0.6.4`
- Date: 2026-03-02

## Contract Sources Reviewed
- Target plan: [docs/architecture/v1.0/v0.6/v0.6.4/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.6/v0.6.4/plan.md)
- Related interfaces/code:
  - [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs)
  - [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py)
  - [bindings/python/Cargo.toml](/Users/lashby/Projects/AlloyGBM/bindings/python/Cargo.toml)
  - [crates/engine/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/engine/src/lib.rs)

## Planned vs Observed Contract Summary
- Planned:
  - Extend native `train_regression_artifact` with optional categorical + `time_index` inputs.
  - Build `CategoricalTargetEncodingSpec` + `TargetEncoderConfig` in binding.
  - Route categorical path through `Trainer::fit_iterations_with_single_target_encoded_feature`.
  - Additive Python estimator params and fit-time validation only; preserve numeric behavior.
  - No engine contract changes expected in this slice.
- Observed:
  - Native binding signature is extended with optional categorical/time fields and consistency validation.
  - Binding constructs `CategoricalTargetEncodingSpec` + `TargetEncoderConfig` and routes to engine wrapper.
  - `GBMRegressor` gained additive categorical params and fit-time validations while keeping numeric path intact.
  - Engine public contract remains unchanged in this slice (existing wrapper consumed as planned).

## Drift Findings

### 1) Dependency-Direction Drift
- Status: NONE
- Evidence:
  - `bindings/python` depends on `alloygbm-engine` and `alloygbm-categorical` to build planned bridge types.
  - No reverse dependency from engine/core back into binding introduced.

### 2) Public API Drift
- Status: NONE
- Evidence:
  - `GBMRegressor` changes are additive (`categorical_*` constructor params and optional fit kwargs).
  - Numeric-only usage shape remains valid (`fit(X, y)` and `predict(X)` unchanged for existing call sites).

### 3) Artifact Contract Drift
- Status: NONE
- Evidence:
  - `v0.6.4` scope does not alter core artifact schema/version.
  - Bridge consumes existing engine categorical-state behavior; no new artifact format fields were introduced in this layer.

### 4) Naming/Path Drift
- Status: NONE
- Evidence:
  - Implementation files match plan-declared interface locations.
  - Layer artifacts exist at expected target path under `docs/architecture/v1.0/v0.6/v0.6.4/`.

## Severity Summary
- High: 0
- Medium: 0
- Low: 0

## Approved Intentional Drift
- None.

## Conclusion
- No contract drift detected for `v0.6.4` against the declared plan interfaces and boundaries.
- Layer implementation remains aligned with planned dependency direction, API scope, and artifact-contract expectations.
