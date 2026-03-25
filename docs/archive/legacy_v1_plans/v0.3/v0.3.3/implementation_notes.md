# AlloyGBM v0.3.3 Implementation Notes

## Summary of What Was Built
- Added leakage-aware split helper module [validation.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/validation.py):
  - `purged_time_series_splits(time_index, n_splits, purge_gap, embargo)`
  - `purged_panel_splits(time_index, group_index, n_splits, purge_gap, embargo)`
- Implemented deterministic contiguous fold construction over sorted unique time periods with:
  - purge window exclusion before each test fold,
  - embargo window exclusion after each test fold,
  - explicit invalid-configuration errors when training rows are eliminated.
- Exported split helpers from package API in [__init__.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/__init__.py).
- Added dedicated split-helper tests in [test_validation_splits.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_validation_splits.py):
  - deterministic outputs,
  - no-overlap and purge/embargo invariants,
  - panel time-bucket behavior,
  - invalid parameter/data-path assertions.
- Extended runtime package checks in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py) to assert split helper exports and basic callability from installed wheel.
- Added layer plan and verification artifacts:
  - `docs/architecture/v1.0/v0.3/v0.3.3/plan.md`
  - `docs/architecture/v1.0/v0.3/v0.3.3/verification_report.md`

## Non-Intuitive Decisions
- Decision: define panel splitting as time-bucketed splitting across all groups, with `group_index` used for shape validation and panel intent (not as the primary split axis).
- Reason: parent `v0.3` scope requires time-aware panel-safe evaluation tooling, and time-first splitting best enforces purge/embargo leakage boundaries.
- Impact: split outputs are deterministic and leakage-oriented; group-aware balancing/stratification remains future scope.

- Decision: fail fast when purge/embargo settings produce empty training folds.
- Reason: silently returning invalid folds would allow leakage tooling misuse and ambiguous downstream behavior.
- Impact: users receive explicit `ValueError` guidance for incompatible split configuration.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Added new public APIs exactly scoped to `v0.3.3`:
  - `purged_time_series_splits`
  - `purged_panel_splits`
- Kept `GBMRegressor` behavior unchanged.
- Kept `evaluation.py` metric semantics unchanged.
- No native Rust bridge changes or training-time leakage enforcement changes were introduced.

## Known Gaps Deferred to Next Layer
- Optional `v0.3.4` polish/cleanup slice is still open if parent `v0.3` needs additional API/docs hardening.
- Parent `v0.3` rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.3/verification_report.md`

## Follow-Up Actions
- Update `docs/architecture/state/layer_index.yaml` to mark `docs/architecture/v1.0/v0.3/v0.3.3` as `verified`.
- Decide whether to open `v0.3.4` (if residual scope remains) or close parent `v0.3` with rollup artifacts and verification evidence.
