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
  per-component impact. `--gate` also compares calibrated MorphBoost with the
  matching `auto` run and fails on non-finite output or a material regression.
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
- `benchmarks/objective_benchmark.py` — deterministic held-out validation for
  large-query LambdaMART truncation and skewed-count Poisson/Gamma/Tweedie
  objectives. Report at `docs/benchmarks/objective_benchmark_v1.md`.
- `benchmarks/review_guardrails.py` — deterministic July-review evidence for
  quantile split selection, GOSS rates, and DART dropout profiles. The full
  report is `docs/benchmarks/review_guardrails_v1.md`; `--quick --gate` is the
  CI-sized contract run. DART timings are descriptive, and its 1.50x quality
  gate applies only to explicit default-like profiles (`drop_rate <= 0.10`).
- `benchmarks/monotone_constraints_benchmark.py` — deterministic scalar
  monotone-constraint acceptance evidence across regression and binary
  objectives. The full report is `docs/benchmarks/monotone_constraints_v1.md`;
  finite numeric sweeps and quality gates are required, while timing is
  descriptive.
- `benchmarks/multiclass_parallelism_benchmark.py` — deterministic `n_jobs`
  and multiclass class-tree scaling evidence across tall/narrow, medium/wide,
  and small matrices, 3 and 12 classes, and both growth strategies. The full
  report is `docs/benchmarks/multiclass_parallelism_v1.md`; exact serial versus
  parallel artifact and prediction hashes are required.
- `benchmarks/allocation_reuse_benchmark.py` — isolated paired subprocess
  evidence for allocation reuse across tall/deep, wide/deep, short/wide, and
  shallow/tall regression matrices with missing values. Quick mode compares
  the installed candidate runtime with itself. Full mode accepts separate
  Python executables and worktrees for a same-host baseline/candidate run.
- `benchmarks/architectural_backlog/` — isolated baseline/candidate harness for
  SoA histograms, node-level parallelism, duplicate bin storage, compact
  predictor nodes, EFB, and approximate quantile sketches. Methodology and
  baseline report are in `docs/benchmarks/architectural_backlog_v1.md`.

```bash
# Quick MorphBoost comparison report
python3 benchmarks/morph_report.py

# MorphBoost component ablation
python3 benchmarks/morph_ablation.py

# Fast calibrated-MorphBoost A/B regression gate
python3 benchmarks/morph_ablation.py --quick --gate

# DRO clean-holdout robustness report (two-seed smoke profile)
python3 benchmarks/dro_robustness.py --quick

# Large-query LambdaMART and skewed-count GLM validation
python3 benchmarks/objective_benchmark.py --gate

# July-review evidence capture (full) and CI-sized contract run
python3 benchmarks/review_guardrails.py --gate \
  --output docs/benchmarks/review_guardrails_v1.md
python3 benchmarks/review_guardrails.py --quick --gate

# Scalar monotone-constraint evidence capture (full) and CI-sized contract run
python3 benchmarks/monotone_constraints_benchmark.py --quick --gate
python3 benchmarks/monotone_constraints_benchmark.py \
  --gate \
  --output docs/benchmarks/monotone_constraints_v1.md

# Allocation-reuse contract and compact candidate self-consistency gate
python3 -m pytest benchmarks/tests/test_allocation_reuse_benchmark.py -q
python3 benchmarks/allocation_reuse_benchmark.py --quick --gate

# Full same-host A/B after running maturin develop --release in each environment
/path/to/candidate/.venv/bin/python \
  /path/to/candidate/benchmarks/allocation_reuse_benchmark.py \
  --full --gate \
  --baseline-python /path/to/baseline/.venv/bin/python \
  --baseline-workdir /path/to/baseline \
  --candidate-python /path/to/candidate/.venv/bin/python \
  --candidate-workdir /path/to/candidate \
  --output-json benchmarks/results/allocation_reuse.json \
  --output-markdown benchmarks/results/allocation_reuse.md

# Fast smoke run for all six deferred architecture projects
python3 -m benchmarks.architectural_backlog.run \
  --profile quick --mode baseline --gate

# Full baseline capture (three isolated repetitions per case)
python3 -m benchmarks.architectural_backlog.run \
  --profile full --mode baseline \
  --output benchmarks/results/architectural_backlog_baseline.json --gate

# Same-host candidate comparison from an implementation branch
python3 -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate \
  --baseline benchmarks/results/architectural_backlog_baseline.json \
  --output benchmarks/results/architectural_backlog_candidate.json --gate

# Numerai benchmark (slow; downloads data on first run)
python3 benchmarks/numerai_benchmark.py --feature-set small \
  --rounds 1200 --learning-rate 0.05 --max-depth 6 --col-subsample 0.3
```

The allocation-reuse workers run with isolated import paths and record the
source commit from each declared worktree together with the loaded package and
native-extension paths and extension digest. Warmups run in separate
subprocesses and are excluded from the recorded repetitions. Artifact,
prediction, and RMSE equivalence is exact. Full-mode performance gates compare
per-case medians and then aggregate geometric timing and RSS ratios: aggregate
native slowdown must stay within 3%, aggregate incremental RSS growth within
5%, and at least one deep-pressure case must improve in median time or RSS.
Quick mode intentionally gates equivalence only because its single repetition
is too small for defensible performance claims.

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
