# AlloyGBM v0.9.2 Benchmark Run Summary

## Scope
- Layer: `docs/architecture/v1.0/v0.9/v0.9.2`
- Date: 2026-03-03
- Purpose: capture profile-matrix benchmark execution evidence, including financial low-SNR scenario coverage and optional ultra-depth path.

## Executed Commands
1. `python3 -m py_compile benchmarks/dow_jones_financial/prepare.py benchmarks/run_model_comparison.py`
2. `python3 benchmarks/dow_jones_financial/prepare.py --help`
3. `python3 benchmarks/run_model_comparison.py --help`
4. `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial`
5. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7`
6. `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`

## Dataset Preparation Outcome (`dow_jones_financial`)
- Output dataset: `benchmarks/data/dow_jones_financial/prepared/prepared.csv`
- Preparation counters from command output:
  - `kept_rows=720`
  - `dropped_rows=30`
  - `fallback_targets=0`

## Output Artifacts
Default profile grid run (`shallow_high_lr`, `mid_balanced`, `deep_low_lr`):
- `benchmarks/results/model_comparison_20260303T083035Z.csv`
- `benchmarks/results/model_comparison_20260303T083035Z.json`
- `benchmarks/results/model_comparison_20260303T083035Z.md`
- `benchmarks/results/model_comparison_profile_summary_20260303T083035Z.csv`
- `benchmarks/results/model_comparison_profile_summary_20260303T083035Z.json`

Ultra profile run (`default_ultra` on constrained scenarios):
- `benchmarks/results/model_comparison_20260303T083104Z.csv`
- `benchmarks/results/model_comparison_20260303T083104Z.json`
- `benchmarks/results/model_comparison_20260303T083104Z.md`
- `benchmarks/results/model_comparison_profile_summary_20260303T083104Z.csv`
- `benchmarks/results/model_comparison_profile_summary_20260303T083104Z.json`

## Profile-Level Outcomes

### Best RMSE By Scenario (Default Grid Run)
- `dense_numeric`: `xgboost` on `deep_low_lr` (`rmse_median=0.522196`)
- `panel_time_series`: `lightgbm` on `deep_low_lr` (`rmse_median=1.769793`)
- `histogram_stress`: `xgboost` on `deep_low_lr` (`rmse_median=0.354513`)
- `dow_jones_financial`: `alloygbm` on `deep_low_lr` (`rmse_median=3.300862`)

### Fastest Fit By Scenario (Default Grid Run)
- `dense_numeric`: `alloygbm` on `mid_balanced` (`fit_seconds_median=0.000701`)
- `panel_time_series`: `alloygbm` on `shallow_high_lr` (`fit_seconds_median=0.003300`)
- `histogram_stress`: `alloygbm` on `deep_low_lr` (`fit_seconds_median=0.062718`)
- `dow_jones_financial`: `alloygbm` on `mid_balanced` (`fit_seconds_median=0.000374`)

### Ultra Profile Evidence (`default_ultra`)
- Command executed successfully for `dense_numeric` and `dow_jones_financial`, including `ultra_low_lr` (`rounds=10000`, `learning_rate=0.005`).
- Best RMSE in constrained ultra run:
  - `dense_numeric`: `xgboost` on `ultra_low_lr` (`rmse_median=0.521456`)
  - `dow_jones_financial`: `alloygbm` on `deep_low_lr` (`rmse_median=3.300862`)
- Fastest fit in constrained ultra run:
  - `dense_numeric`: `alloygbm` on `deep_low_lr` (`fit_seconds_median=0.000706`)
  - `dow_jones_financial`: `alloygbm` on `shallow_high_lr` (`fit_seconds_median=0.000324`)

## Notes
- All scenario/profile/model rows reported `PASS` in both matrix executions.
- `v0.9.2` captures benchmark breadth and reproducible artifact generation; CI threshold enforcement remains deferred to later `v0.9` layers.
