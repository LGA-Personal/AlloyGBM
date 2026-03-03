# AlloyGBM v0.8 Migration and Compatibility Narrative (`v0.8.4`)

## Purpose
- Finalize migration-facing guidance for `v0.8` hardening closeout.
- Convert compatibility expectations into executable checks tied to release evidence.
- Provide a checklist that can be reused during parent `v0.8` and milestone `1.0.0` go/no-go review.

## Migration Impact Summary
- Model artifact format:
  - No format version bump is introduced in `v0.8` (`v1` contract remains in effect).
  - Strict mode still requires exactly one `Trees` section and one `PredictorLayout` section.
  - Legacy-compatible mode still allows strict dual-section artifacts and legacy trees-only artifacts.
- Python API:
  - No breaking signature changes are introduced for `GBMRegressor` fit/predict/SHAP methods in this slice.
  - `predict_from_artifact` continues to accept bytes-like payloads (`bytes`, `bytearray`, `memoryview`).
- Runtime behavior:
  - `v0.7` SHAP behavior and predictor parity commitments remain non-regression baselines.
  - Categorical-aware artifact compatibility behavior remains additive and backward-compatible.

## Compatibility Policy (Locked)
1. Strict artifact compatibility remains the canonical path for regressor inference (`predict`) and canonical bridge checks.
2. Legacy trees-only artifacts remain supported in compatibility mode where documented.
3. Optional sections (for example categorical state) remain additive and must not break strict required-section classification.
4. Compatibility errors remain deterministic and actionable (mode/section mismatch reporting).

## Operator Checklist
1. Confirm formatting/lint/test/doc/python gate health:
   - `cargo fmt -- --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
   - `cargo doc --workspace --no-deps`
   - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
2. Confirm strict/legacy classification invariants in core:
   - `cargo test -p alloygbm-core required_section_compatibility_report_classifies_strict_and_legacy_layouts`
   - `cargo test -p alloygbm-core strict_compatibility_allows_optional_categorical_state_section`
3. Confirm predictor compatibility behavior:
   - `cargo test -p alloygbm-predictor predictor_accepts_legacy_trees_only_artifact`
4. Confirm Python bridge/regressor compatibility behavior:
   - `cargo test -p alloygbm-python canonical_bridge_rejects_legacy_trees_only_artifacts`
   - `python3 -m unittest bindings/python/tests/test_regressor_contract.py`
5. Confirm benchmark reproducibility evidence exists from `v0.8.3`:
   - `docs/architecture/v1.0/v0.8/v0.8.3/benchmark_run_summary.md`
   - `benchmarks/results/model_comparison_latest.md`

## Traceability Matrix
| Compatibility Dimension | Contract Expectation | Evidence Source | Verification Command |
| --- | --- | --- | --- |
| Required-section classification | strict vs legacy trees-only remains stable | `crates/core/src/lib.rs` tests | `cargo test -p alloygbm-core required_section_compatibility_report_classifies_strict_and_legacy_layouts` |
| Optional categorical section compatibility | strict mode remains compatible with optional categorical state | `crates/core/src/lib.rs` tests | `cargo test -p alloygbm-core strict_compatibility_allows_optional_categorical_state_section` |
| Predictor legacy support | predictor loads legacy trees-only artifacts in compatibility mode | `crates/predictor/src/lib.rs` tests | `cargo test -p alloygbm-predictor predictor_accepts_legacy_trees_only_artifact` |
| Canonical Python bridge strictness | canonical bridge rejects legacy trees-only artifacts | `bindings/python/src/lib.rs` tests | `cargo test -p alloygbm-python canonical_bridge_rejects_legacy_trees_only_artifacts` |
| Python artifact payload compatibility | `predict_from_artifact` accepts bytes-like payloads | `bindings/python/tests/test_regressor_contract.py` | `python3 -m unittest bindings/python/tests/test_regressor_contract.py` |
| SHAP + estimator non-regression | SHAP/predict contract remains stable | `v0.7` child/parent verification reports | full gate reruns in `v0.8.4` |
| Benchmark reproducibility continuity | benchmark workflow remains documented and reproducible | `v0.8.3` benchmark summary + results | `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 80` (from `v0.8.3` evidence) |

## Residual Risks
- Legacy compatibility remains intentionally scoped to documented paths; unsupported malformed layouts should continue to fail fast.
- Benchmark thresholds are not enforced in CI yet; reproducibility evidence exists, but pass/fail policy remains a parent `v0.8`/later decision.

## Ready-for-Parent Conditions
- `v0.8.4` artifacts complete (`plan`, `implementation_notes`, `verification_report`).
- Compatibility checklist commands are green and linked in verification evidence.
- Layer index advances to parent `docs/architecture/v1.0/v0.8` for rollup closeout.
