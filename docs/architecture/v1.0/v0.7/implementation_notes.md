# AlloyGBM v0.7 Implementation Notes

## Summary of What Was Built
- Delivered `v0.7` TreeSHAP milestone through child slices `v0.7.1` to `v0.7.4`:
  - `v0.7.1`: established artifact-backed SHAP contract, deterministic validation, additivity harness, and global-importance baseline.
  - `v0.7.2`: replaced baseline contribution assignment with exact Shapley traversal math over current tree payload semantics.
  - `v0.7.3`: hardened compatibility/parity coverage (legacy/strict/malformed artifacts, predictor parity, deterministic tie-ordering).
  - `v0.7.4`: exposed Python SHAP bridge APIs and `GBMRegressor` SHAP methods with runtime/contract coverage.

## Key Implementation Outcomes by Area
- Rust SHAP runtime (`crates/shap`):
  - artifact-backed explanation entrypoint with expected value + per-row contribution matrix,
  - exact Shapley subset-weighted contributions with deterministic additivity checks,
  - deterministic global-importance aggregation from mean absolute SHAP contributions,
  - compatibility handling for strict dual-section and legacy trees-only artifacts.
- Python bridge (`bindings/python/src/lib.rs`):
  - native SHAP entrypoints:
    - `shap_explain_rows`
    - `shap_global_importance`
  - deterministic error mapping from Rust SHAP errors to Python exception types.
- Python regressor surface (`bindings/python/alloygbm/regressor.py`):
  - `GBMRegressor.shap_values(X, include_expected_value=False)`,
  - `GBMRegressor.feature_importances(X, method="shap")`.

## Non-Intuitive Decisions
- Decision: retain legacy `shap_values_stub` / `global_importance_stub` compatibility shims while shifting to artifact-backed APIs.
- Reason: avoid abrupt breakage for earlier call paths while migrating to deterministic runtime behavior.
- Impact: newer APIs are authoritative, while compatibility helpers remain stable.

- Decision: exact Shapley computation bounded by split-feature guardrail (`MAX_EXACT_SPLIT_FEATURES = 20`).
- Reason: exact subset enumeration is exponential in split-feature count.
- Impact: wide split-feature models return deterministic contract violations rather than silent approximation.

- Decision: Python `feature_importances` requires `X` and only supports `method="shap"` in this milestone.
- Reason: SHAP global importance is data-distribution dependent and should be explicit about evaluation rows.
- Impact: deterministic, explicit behavior with fast failure for unsupported methods.

## Plan Contradictions and Why
- No parent-plan contradictions were introduced.
- Parent scope and ordering (`v0.7.1` -> `v0.7.4` -> closeout) remained intact.

## Boundary/Interface Changes vs Plan
- Implemented planned interface additions:
  - Rust SHAP artifact-backed interfaces,
  - Python-native SHAP bridge APIs,
  - `GBMRegressor` SHAP methods.
- Preserved out-of-scope boundaries:
  - no GPU/Metal SHAP work,
  - no model format changes,
  - no ranking/classification SHAP semantics.

## Residual Risks
- Exact SHAP complexity remains exponential in split-feature count for models at/above guardrail thresholds.
- SHAP expected-value/global-importance outputs are row-set dependent by design and must be interpreted with consistent evaluation rows.

## Follow-Up Actions
- Advance to top-level `v0.8` planning with `v0.7` accepted as verified.
- Retain SHAP parity/additivity/compatibility tests as required non-regression gates during `v0.8` hardening.
