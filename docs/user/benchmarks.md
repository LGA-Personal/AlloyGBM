# Benchmarks

This page summarizes how AlloyGBM is benchmarked and what the current results
say.

## Methodology

The comparative benchmark runner lives in `benchmarks/run_model_comparison.py`.

The suite compares AlloyGBM against:

- XGBoost
- LightGBM
- CatBoost

It also includes two MorphBoost variants of AlloyGBM as separate arms:

- `alloygbm_morph` -- `training_mode="morph"` with the default constant LR schedule
- `alloygbm_morph_cosine` -- `training_mode="morph"` with `lr_schedule="warmup_cosine"`

A focused MorphBoost-vs-peers comparison script is also provided at
`benchmarks/morph_report.py`, with a Numerai-specific harness at
`benchmarks/numerai_benchmark.py`.

Benchmarks span three task types:

### Regression

- `dense_numeric`
- `california_housing`
- `bike_sharing`
- `panel_time_series`
- `dow_jones_financial`

### Classification

- `breast_cancer`
- `synthetic_classification`

### Ranking

- `synthetic_ranking`

Profiles are evaluated across shallow, mid, and deep configurations to show how
each library behaves under different learning-rate / depth / round budgets.

## Current Results

### Regression

- AlloyGBM is strongest on `panel_time_series`.
- AlloyGBM is strong on `dow_jones_financial`, especially under the deeper
  low-learning-rate profile.
- AlloyGBM is competitive but not leading on `dense_numeric`.
- AlloyGBM currently trails on `california_housing` and `bike_sharing`.
- AlloyGBM is typically the fastest trainer on most scenario/profile rows.

### Classification

- AlloyGBM is competitive with established libraries on accuracy, log-loss, and
  AUC metrics across `breast_cancer` and `synthetic_classification`.

### Ranking

- AlloyGBM competes on `synthetic_ranking` using its native LambdaMART
  implementation, evaluated via NDCG@5, NDCG@10, and full NDCG.

### MorphBoost Variants

- On Numerai-style residualized regression at scale (~2.7M rows, 42 features,
  5000 rounds), AlloyGBM's MorphBoost variants lead all peer libraries on
  validation MMC (Meta-Model Contribution) and Sharpe, while Numerai-corr
  trails the peers by a small margin (~0.0006-0.0009).
- `alloygbm_morph` is typically the fastest of the three AlloyGBM variants
  on this workload due to faster convergence under the EMA-shaped gain.
- See `benchmarks/numerai_benchmark.py` for the reproducer.

## Metrics By Task Type

| Task Type | Metrics |
| --- | --- |
| Regression | RMSE, MAE, R2 |
| Classification | Accuracy, Log-Loss, AUC |
| Ranking | NDCG@5, NDCG@10, NDCG |

## Stage Timing Output

The benchmark runner breaks AlloyGBM fit time into:

- `input_adaptation_seconds`
- `native_bridge_prepare_seconds`
- `native_train_seconds`
- `fit_seconds`
- `predict_seconds`

Use those columns to distinguish preprocessing-heavy regressions from actual
trainer regressions.

## How To Run Them

Basic run (all scenarios):

```bash
python3 benchmarks/run_model_comparison.py --force-prepare
```

Regression only:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios california_housing bike_sharing dense_numeric panel_time_series dow_jones_financial \
  --profile-grid default \
  --profile-seeds 7
```

Classification only:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios breast_cancer synthetic_classification \
  --profile-grid default \
  --profile-seeds 7
```

Ranking only:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios synthetic_ranking \
  --profile-grid default \
  --profile-seeds 7
```

See the full runner guide in [benchmarks/README.md](../../benchmarks/README.md).

## How To Interpret The Results

Use the benchmark suite to answer two different questions:

- Where is AlloyGBM already clearly strong?
- Where does it still lag established libraries?

That second question matters. The current suite is intentionally honest about
weak spots, especially on broader real-world datasets.
