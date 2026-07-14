# Benchmarks

This page summarizes how AlloyGBM is benchmarked and what the current results
say.

## Methodology

The comparative benchmark runner lives in `benchmarks/run_model_comparison.py`.

The suite compares AlloyGBM against:

- XGBoost
- LightGBM
- CatBoost

It also includes additional AlloyGBM variants as separate arms:

- `alloygbm_dro` -- `leaf_solver="dro"` with robust scalar leaves
- `alloygbm_morph` -- `training_mode="morph"` with the default constant LR schedule
- `alloygbm_morph_cosine` -- `training_mode="morph"` with `lr_schedule="warmup_cosine"`
- `alloygbm_linear` -- `leaf_model="linear"` (piecewise-linear leaves) with auto training mode
- `alloygbm_morph_linear` -- `leaf_model="linear"` combined with `training_mode="morph"`

A focused MorphBoost-vs-peers comparison script is also provided at
`benchmarks/morph_report.py`, with a Numerai-specific harness at
`benchmarks/numerai_benchmark.py`. A dedicated PL-trees benchmark with
convergence-curve and λ-sweep analysis lives at `benchmarks/pl_trees_benchmark.py`;
results are reported in `docs/benchmarks/pl_trees_v1.md`.
A deterministic large-query LambdaMART and skewed-count GLM harness lives at
`benchmarks/objective_benchmark.py`; its current results are recorded in
`docs/benchmarks/objective_benchmark_v1.md`.

The comparative runner also emits a temporal/panel stability table for scenarios
whose names include `time`, `temporal`, or `panel`. It reports mean score,
worst score, and score standard deviation across repeated runs; this is the
primary comparison surface for `alloygbm_dro`.

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

### Piecewise-Linear Leaf Variants

- `leaf_model="linear"` shows ~10× faster convergence on linearly-structured
  data (fewer rounds to reach the same RMSE).
- +3.5% RMSE improvement on California Housing and +1.75pp accuracy on
  Breast Cancer vs constant-leaf baselines.
- 2–8× per-round training overhead from the closed-form Cholesky solve.
- See `docs/benchmarks/pl_trees_v1.md` for the full report.

### DRO Leaf Variant

- `leaf_solver="dro"` is expected to trade a modest training-time overhead for
  lower sensitivity to noisy within-leaf gradient dispersion.
- Inference speed matches standard constant leaves because DRO values are stored
  directly in the artifact.
- Treat success as improved temporal/panel stability, especially worst-run or
  worst-era score, not necessarily better in-sample convergence.

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
