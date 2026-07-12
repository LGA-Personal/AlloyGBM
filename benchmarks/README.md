# Benchmark Dataset Workspace

This directory organizes benchmark dataset preparation and cross-library model comparison for AlloyGBM.

## Scenario Overview

| Scenario | Task | Source | Rows | Features | Notes |
|---|---|---|---|---|---|
| `california_housing` | regression | sklearn | 20640 | 8 | Geography-based price prediction |
| `bike_sharing` | regression | UCI | ~17389 | 11 | Time-series with temporal split |
| `dense_numeric` | regression | UCI (Wine Quality) | 1599 | 11 | Dense continuous, no categoricals |
| `panel_time_series` | regression | UCI (Air Quality) | ~9471 | 11 | Panel with next-step target |
| `histogram_stress` | regression | synthetic | 50000 | 32 | Skewed + quantized histogram pressure |
| `dow_jones_financial` | regression | UCI | ~750 | 10 | Low-SNR financial, temporal split |
| `abalone_regression` | regression | UCI | 4177 | 8 | Age prediction, 1 ordinal feature |
| `synthetic_categorical` | regression | synthetic | 10000 | 15 | Categorical interaction target |
| `breast_cancer` | classification | sklearn | 569 | 30 | Binary, Wisconsin diagnostic |
| `adult_income` | classification | UCI | ~30000 | 13 | Binary income >50K, mixed features |
| `synthetic_classification` | classification | synthetic | 50000 | 32 | Binary, weighted linear + nonlinear |
| `wine_multiclass` | multiclass | sklearn | 178 | 13 | 3-class cultivar identification |
| `digits_multiclass` | multiclass | sklearn | 1797 | 64 | 10-class digit recognition |
| `synthetic_multiclass` | multiclass | synthetic | 10000 | 20 | 5-class cluster-based boundaries |
| `synthetic_ranking` | ranking | synthetic | 5000 | 16 | 200 queries × 25 docs, 5-level relevance |
| `california_ranking` | ranking | sklearn | ~20595 | 8 | California Housing: ~44 geographic queries × ~468 docs, 5-level relevance |

## Layout

Each scenario is a directory containing:

- `manifest.yaml` — metadata (name, task type, source, target column, optional group column)
- `prepare.py` — standalone script that downloads (if needed) and writes `prepared.csv`

Generated data is written under `benchmarks/data/` (git-ignored).

## Usage

### Prepare individual scenarios

```bash
# sklearn scenarios (no download needed)
python3 benchmarks/breast_cancer/prepare.py
python3 benchmarks/wine_multiclass/prepare.py
python3 benchmarks/digits_multiclass/prepare.py

# UCI download scenarios
python3 benchmarks/adult_income/prepare.py
python3 benchmarks/abalone_regression/prepare.py
python3 benchmarks/dense_numeric/prepare.py
python3 benchmarks/bike_sharing/prepare.py
python3 benchmarks/panel_time_series/prepare.py --max-rows 150000
python3 benchmarks/dow_jones_financial/prepare.py --force-download

# Synthetic scenarios
python3 benchmarks/synthetic_classification/prepare.py
python3 benchmarks/synthetic_multiclass/prepare.py
python3 benchmarks/synthetic_categorical/prepare.py
python3 benchmarks/synthetic_ranking/prepare.py
python3 benchmarks/california_ranking/prepare.py
python3 benchmarks/histogram_stress/prepare.py --rows 100000 --features 48
```

### Cross-library model comparison

The runner registers the following model arms by default per task type:

- `alloygbm` (auto training mode)
- `alloygbm_dro` (`leaf_solver="dro"`)
- `alloygbm_factor_neutral` (`neutralization="per_round_gradient"` with synthetic factor exposures unless real exposures are provided)
- `alloygbm_factor_neutral_dro` (factor-neutral + DRO leaves)
- `alloygbm_morph` (`training_mode="morph"`, constant LR)
- `alloygbm_morph_cosine` (`training_mode="morph"`, `lr_schedule="warmup_cosine"`)
- `alloygbm_linear` (`leaf_model="linear"`, auto training mode)
- `alloygbm_morph_linear` (`leaf_model="linear"` + `training_mode="morph"`)
- `lightgbm`, `xgboost`, `catboost`

The two `*_linear` arms apply `lambda_l2=0.01` by default
(tunable via `--alloy-linear-lambda-l2`), as recommended for weight stability
under the closed-form ridge solve.

Use `--models` to filter which arms run. Example: just MorphBoost vs peers:

```bash
python3 benchmarks/run_model_comparison.py \
  --models alloygbm alloygbm_morph alloygbm_morph_cosine lightgbm xgboost catboost \
  --force-prepare
```

Just PL-trees vs peers:

```bash
python3 benchmarks/run_model_comparison.py \
  --models alloygbm alloygbm_linear lightgbm xgboost catboost \
  --force-prepare
```

Run all scenarios with default profiles and a single seed:

```bash
python3 benchmarks/run_model_comparison.py --force-prepare
```

Profile matrix (shallow / mid / deep) with 3 seeds:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default \
  --profile-seeds 7,17,29
```

Focused multiclass run (demonstrates AlloyGBM's softmax classification):

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios wine_multiclass digits_multiclass synthetic_multiclass \
  --profile-grid default \
  --profile-seeds 7,17,29
```

Classification head-to-head (binary + multiclass):

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios breast_cancer adult_income wine_multiclass digits_multiclass synthetic_multiclass \
  --profile-grid default \
  --profile-seeds 7,17,29
```

Ranking focused run:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios synthetic_ranking california_ranking \
  --profile-grid default \
  --profile-seeds 7,17,29
```

Focused real UCI regression set:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios california_housing bike_sharing dense_numeric panel_time_series \
              dow_jones_financial abalone_regression \
  --profile-grid default \
  --profile-seeds 7
```

Continuous-feature binning strategy A/B:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default \
  --profile-seeds 7,17,29 \
  --alloy-continuous-binning-strategy quantile \
  --alloy-continuous-binning-max-bins 256
```

Supported values: `linear` (default), `rank`, `quantile`.

Ultra profile (10000 rounds) on constrained scenarios:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default_ultra \
  --profile-seeds 7 \
  --scenarios dense_numeric dow_jones_financial
```

### Focused harnesses

Additional lighter-weight scripts target specific features:

- `benchmarks/morph_report.py` — quick MorphBoost-vs-peers comparison on a
  curated set of sklearn-based datasets. Defaults to `--quick` (60 rounds);
  drop the flag for 300-round comparisons.
- `benchmarks/morph_ablation.py` — toggles MorphBoost components individually
  (warmup, balance penalty, lr_schedule) on synthetic data to attribute
  per-component impact.
- `benchmarks/numerai_benchmark.py` — Numerai-tournament-style residualized
  regression at scale, evaluating numerai_corr, Sharpe, and MMC. Includes
  the `alloygbm_morph` and `alloygbm_morph_cosine` arms.
- `benchmarks/pl_trees_benchmark.py` — piecewise-linear-leaf
  convergence-curve and λ-sweep analysis across regression, classification,
  and ranking scenarios. Report at `docs/benchmarks/pl_trees_v1.md`.
- `benchmarks/dro_robustness.py` — deterministic clean-holdout comparison of
  standard and DRO leaves after clean versus outlier-contaminated training.
  It includes scalar and joint shared-tree paths. Report at
  `docs/benchmarks/dro_robustness_v1.md`.

```bash
# Quick MorphBoost comparison report
python3 benchmarks/morph_report.py

# MorphBoost component ablation
python3 benchmarks/morph_ablation.py

# DRO clean-holdout robustness report (two-seed smoke profile)
python3 benchmarks/dro_robustness.py --quick

# Numerai benchmark (slow; downloads data on first run)
python3 benchmarks/numerai_benchmark.py --feature-set small \
  --rounds 1200 --learning-rate 0.05 --max-depth 6 --col-subsample 0.3
```

## Outputs

Results are written to `benchmarks/results/`:

- `model_comparison_latest.csv` — per-record raw results
- `model_comparison_latest.json` — raw results + run metadata
- `model_comparison_latest.md` — formatted report with per-task-type tables
- `model_comparison_profile_summary_latest.csv` — aggregated by (scenario, profile, model)

## Runtime Contract Validation

The runner validates the loaded `alloygbm` runtime before any benchmarks run:

- `GBMRegressor` must expose `n_estimators`, `learning_rate`, `max_depth`, `row_subsample`, `col_subsample`.
- The native extension must expose `train_regression_artifact`.

If the check fails, benchmarks stop early with a descriptive error instead of silently benchmarking a stale build.

## Per-record Timing

Each record captures:

| Field | Meaning |
|---|---|
| `input_adaptation_seconds` | Python-side data conversion to AlloyGBM format |
| `native_bridge_prepare_seconds` | Rust bridge preparation before training |
| `native_train_seconds` | Rust training loop |
| `fit_seconds` | Total `model.fit()` wall time |
| `predict_seconds` | Total `model.predict()` wall time |

The split between `native_bridge_prepare_seconds` and `native_train_seconds` isolates AlloyGBM-specific overhead from the core gradient-boosting loop.

## Split Strategies

| Task type | Split strategy |
|---|---|
| `regression` | Random split |
| `classification` | Stratified on class label |
| `multiclass_classification` | Stratified on class label |
| `ranking` | Group-aware (whole queries stay together) |
| Time-series scenarios | Timestamp-boundary split (no timestamp appears in both train and test) |

## Temporal Leakage Safeguards

- `panel_time_series`: uses a next-timestep target (`target_co_gt`) rather than same-timestep duplication.
- `dow_jones_financial`: excludes forward-looking `next_weeks_*` fields from features; only the target carries future information.
- All time-series scenarios: `run_model_comparison.py` enforces timestamp-boundary splits.

## Adding a New Scenario

1. Create `benchmarks/<scenario_name>/manifest.yaml` following the schema in any existing manifest.
2. Create `benchmarks/<scenario_name>/prepare.py` following the pattern in `breast_cancer/prepare.py` (sklearn) or `dense_numeric/prepare.py` (UCI download).
3. Add `"<scenario_name>"` to `AVAILABLE_SCENARIOS` in `run_model_comparison.py`.
4. Run `python3 benchmarks/run_model_comparison.py --force-prepare --scenarios <scenario_name>` to verify end-to-end.
