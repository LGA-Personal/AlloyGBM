# AlloyGBM v0.9.2 Plan (Benchmark Matrix Expansion and Financial Scenario Addition)

## Summary
- Goal: execute `v0.9.2` by expanding benchmark coverage from single-setting runs to a reproducible hyperparameter profile matrix (shallow/intermediate/deep/optional ultra-deep) and adding a finance-relevant low-SNR scenario based on UCI Dow Jones Index data.
- Success criteria:
  - benchmark runner supports multi-profile execution with persisted per-profile outputs,
  - benchmark evidence includes low-tree/high-learning-rate and high-tree/low-learning-rate comparisons for AlloyGBM, LightGBM, and XGBoost,
  - a new `dow_jones_financial` scenario is available with leakage-aware preprocessing and deterministic preparation behavior,
  - `v0.9.1` deferred item `BG-902` is closed with command-backed evidence.
- Audience: engineers preparing `v0.9.3` temporal integrity hardening, `v0.9.4` runtime provenance fixes, and later continuous-feature/competitiveness/closeout slices.

## Scope
### In Scope
- Create `v0.9.2` layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.9.2/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.9.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.9.2/verification_report.md`
- Expand benchmark profile coverage in `benchmarks/run_model_comparison.py`:
  - support profile-matrix execution in a single run,
  - preserve backward compatibility for current single-profile CLI usage.
- Add finance-focused scenario:
  - `benchmarks/dow_jones_financial/manifest.yaml`
  - `benchmarks/dow_jones_financial/prepare.py`
  - source URL: `https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip`
- Publish profile-run outputs in `benchmarks/results/` with profile-aware records and summary tables.
- Update benchmark documentation:
  - `benchmarks/README.md` usage examples for profile matrix and new scenario.
- Update layer state:
  - mark `docs/architecture/v1.0/v0.9/v0.9.2` as `verified`,
  - advance `active_target` and `suggested_next_layer` to `docs/architecture/v1.0/v0.9/v0.9.3`.

### Out of Scope
- Algorithmic training/inference optimizations intended for `v0.9.3`.
- CI hard-fail benchmark threshold enforcement finalization intended for `v0.9.7+` (policy draft updates allowed, full enforcement deferred).
- Ranking/GPU/new-objective roadmap features.
- Breaking changes to `GBMRegressor` public API or model format.

## Interfaces and Types
- New benchmark scenario contract:
  - `benchmarks/dow_jones_financial/manifest.yaml` uses existing manifest schema (`name`, `version`, `kind`, `source`, `prepared`).
  - `prepare.py` writes `prepared.csv` under `benchmarks/data/dow_jones_financial/prepared/`.
  - output includes:
    - `group_id` (ticker symbol),
    - `timestamp` (parsed market date),
    - numeric feature columns from current-week information only,
    - target column default: `target_percent_change_next_weeks_price`.
- Leakage-safe preprocessing defaults:
  - default target source column: `percent_change_next_weeks_price`,
  - fallback target if missing: `next_weeks_close`,
  - exclude leakage-prone columns from features (for example `next_weeks_open`, `next_weeks_close`, and direct target aliases).
- `run_model_comparison.py` CLI contract changes:
  - add profile-matrix options (for example `--profile-grid` and/or repeated `--profile`),
  - keep current single-profile flags (`--learning-rate`, `--max-depth`, `--rounds`) functional as compatibility mode.
- Output schema extensions:
  - add profile metadata columns (`profile_name`, `profile_index`, `run_index`, `seed`, `learning_rate`, `max_depth`, `rounds`),
  - retain existing metric columns (`fit_seconds`, `predict_seconds`, `rmse`, `mae`, `r2`, `status`, `error`).

Backward-compatibility expectations:
- Existing invocation `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 80` continues to work.
- Existing `model_comparison_latest.{csv,json,md}` outputs remain present.
- New profile-summary artifacts are additive.

## Implementation Sequence
1. Define `v0.9.2` benchmark profile matrix and encode it in a reusable config/loader path.
2. Implement `dow_jones_financial` manifest and preparation script with deterministic parsing and leakage-safe feature filtering.
3. Extend `run_model_comparison.py` to execute profile matrix across all requested models/scenarios and emit profile-aware raw results.
4. Add profile-level aggregation (median fit/predict/quality metrics) and publish summary artifacts.
5. Update `benchmarks/README.md` with profile-matrix commands and Dow Jones scenario usage.
6. Execute verification commands, author `implementation_notes.md` and `verification_report.md`, and advance `layer_index.yaml` to `v0.9.3`.

## Test Cases and Scenarios
- Unit/script behavior checks:
  - `python3 -m py_compile benchmarks/dow_jones_financial/prepare.py benchmarks/run_model_comparison.py`
  - `python3 benchmarks/dow_jones_financial/prepare.py --help`
  - `python3 benchmarks/run_model_comparison.py --help`
- Dataset preparation checks:
  - `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial`
  - assert prepared output exists and target column is present.
- Profile matrix integration checks:
  - run shallow/high-lr profile set and deep/low-lr profile set with all three models,
  - run optional ultra profile command path (`10000` rounds at very low learning rate) on constrained scenario set.
- Suggested default profile matrix (decision for this layer):
  - `shallow_high_lr`: `rounds=200`, `learning_rate=0.20`, `max_depth=4`
  - `mid_balanced`: `rounds=1200`, `learning_rate=0.05`, `max_depth=6`
  - `deep_low_lr`: `rounds=5000`, `learning_rate=0.01`, `max_depth=8`
  - `ultra_low_lr` (optional): `rounds=10000`, `learning_rate=0.005`, `max_depth=8`
- Reproducibility defaults:
  - seeds per profile run: `[7, 17, 29]` with median reporting in summary.
- Non-regression command gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.2/plan.md` is present and decision-complete.
2. `benchmarks/dow_jones_financial/manifest.yaml` and `prepare.py` exist and produce a deterministic prepared dataset.
3. Dow Jones scenario preprocessing is leakage-aware with explicit target and excluded feature rules documented in code/comments and notes.
4. `run_model_comparison.py` supports benchmark profile-matrix execution while preserving single-profile backward compatibility.
5. Benchmark outputs include profile metadata and per-profile records for all requested models.
6. At least three default profiles (shallow/mid/deep) run successfully across AlloyGBM, LightGBM, and XGBoost.
7. Optional ultra profile (`10000` rounds, very low learning rate) is executable and documented with runtime caveats.
8. Updated benchmark summary artifact for `v0.9.2` reports profile-level best-quality and best-speed outcomes by scenario.
9. `docs/architecture/v1.0/v0.9/v0.9.2/implementation_notes.md` is present.
10. `docs/architecture/v1.0/v0.9/v0.9.2/verification_report.md` is present with criterion-to-evidence mapping.
11. Standard Rust/Python gate commands pass (`fmt`, `clippy`, `test`, `doc`, Python unittest discovery).
12. `docs/architecture/state/layer_index.yaml` marks `v0.9.2` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.3`.

## Risks and Mitigations
- Risk: deep/ultra profiles cause excessive runtime and noisy comparisons.
  - Mitigation: define default profile set with capped repetitions, treat ultra profile as optional, and report medians across seeds.
- Risk: Dow Jones schema irregularities (percent strings, missing fields) break preparation.
  - Mitigation: robust numeric normalization, explicit missing-value handling, and deterministic row filtering with clear counters.
- Risk: accidental target leakage from next-week columns inflates metrics.
  - Mitigation: hard-code excluded future columns and verify resulting feature set in preparation logs/tests.
- Risk: matrix expansion reduces comparability if model-specific knobs diverge.
  - Mitigation: enforce shared core profile definitions across all models for `v0.9.2`; defer model-specific tuning to `v0.9.3`.

## Assumptions and Defaults
- `v0.9.2` prioritizes benchmark coverage expansion and evidence quality, not algorithmic optimization.
- Default financial target for the new scenario is `percent_change_next_weeks_price` (fallback `next_weeks_close` when unavailable).
- Profile matrix defaults are as listed in this plan; ultra profile is opt-in.
- Runner remains CPU-only and deterministic-mode friendly.
