# Benchmarks

This page summarizes how AlloyGBM is benchmarked and what the current results
say.

## Methodology

The comparative benchmark runner lives in `benchmarks/run_model_comparison.py`,
with a large-scale, thread-sweeping companion at
`benchmarks/scale_comparison.py`.

### Fairness controls

Benchmark numbers are only meaningful if every library is given the same work
and the same resources. The harness enforces this explicitly, and each run
records the settings it used in its JSON output under `params.fairness` and
`params.environment`.

**Equal compute budget.** `--threads N` is applied to every library through its
own knob — AlloyGBM/LightGBM/XGBoost `n_jobs`, CatBoost `thread_count` — and the
OpenMP/BLAS environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) are pinned
to the same value so no library's runtime quietly takes more. `--threads 0`
means all logical CPUs.

> Before v1.0.0 this was wrong in AlloyGBM's favour: the peer libraries were
> hard-coded to a single thread while AlloyGBM was left unconstrained and used
> every core. Any speed comparison published from a pre-1.0 run is inflated by
> roughly AlloyGBM's parallel speedup and should not be trusted.

**Equal hyperparameters.** All libraries share `n_estimators`, `learning_rate`,
`max_depth`, the seed, and row/column subsampling (0.8/0.8). Two
library-specific corrections are needed for that equality to be real:

- LightGBM silently ignores `subsample` unless `subsample_freq >= 1`. Without
  the fix it trained on 100% of rows while the others used 80%.
- LightGBM grows leaf-wise and caps trees at `num_leaves=31` by default, so a
  shared `max_depth=6` did not give it the same capacity as the depth-wise
  peers. The harness sets `num_leaves = 2 ** max_depth`.

**Equal data.** Every library receives the same arrays from the same
train/test split; data preparation and metric computation sit outside the
timed region.

### Choosing a thread budget

The curated scenario suite is run **single-threaded**. That is deliberate: at
these dataset sizes (142 to 40,000 rows) forcing all 10 cores measures
thread-spawn overhead rather than throughput. On the reference host LightGBM
and XGBoost are *slower* multi-threaded than single-threaded below roughly
40,000 rows — LightGBM by up to 20x on the smallest scenarios. Reporting a
multi-threaded win at that scale would flatter AlloyGBM for a reason that has
nothing to do with its algorithms.

Single-threaded results are therefore the honest per-core comparison, and are
also the most reproducible across machines. The realistic multi-threaded case
is covered separately by `benchmarks/scale_comparison.py` on 200k- and
1M-row datasets, where parallelism genuinely pays for every library.

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
The experimental PL split-shortlist harness lives at
`benchmarks/pl_topk_performance.py`; its five-seed compatibility, quality,
memory, and runtime evidence is recorded in
[pl_topk_pr136.md](../benchmarks/pl_topk_pr136.md).
The exhaustive SIMD scanner, nine-shape quality matrix, rejected calibration
trials, and secondary-cost decisions for PR #132 are recorded in
[morphboost_pr132.md](../benchmarks/morphboost_pr132.md).
A dedicated PR #135 report records the exhaustive numeric DRO scanner, its
scalar fallbacks, seven-run 16/64/255-bin timings, the nine-case paired fit
matrix, exact quality-equivalence checks, and rejected optimization trials in
[dro_simd_pr135.md](../benchmarks/dro_simd_pr135.md). The report preserves the
64-bin scanner miss and the accepted fit-time fallback rather than treating the
same-host result as a universal performance claim.
A deterministic large-query LambdaMART and skewed-count GLM harness lives at
`benchmarks/objective_benchmark.py`; its current results are recorded in
`docs/benchmarks/objective_benchmark_v1.md`.
A deterministic July-review harness lives at `benchmarks/review_guardrails.py`;
its committed evidence is [review_guardrails_v1.md](../benchmarks/review_guardrails_v1.md).
It records smoothed-pinball split-selection, GOSS rate-sweep, and DART dropout
profile evidence. DART timing is descriptive; the 1.50x RMSE gate applies only
to explicit default-like profiles with `drop_rate <= 0.10`, while the aggressive
stress profile remains reported and contract-checked.

The PR #137 DART policy harness at `benchmarks/dart_policy_calibration.py`
evaluates explicit caps `2`, `5`, `10`, `20`, and `50` across ten fixed
fixtures and five seeds. Its [calibration report](../benchmarks/dart_policy_calibration_pr137.md)
records the selected default, all gate results, compatibility hashes, and
rejected candidates. The machine comparator owns the fixed matrix/capture
contract and hash parity; warm-start and `n_jobs` are separate passing
regression sentinels, not JSON compatibility records. Timing remains a
same-host policy-selection gate rather than a universal performance claim.

A deterministic scalar monotone-constraint harness lives at
`benchmarks/monotone_constraints_benchmark.py`; its committed evidence is
[monotone_constraints_v1.md](../benchmarks/monotone_constraints_v1.md). It
checks finite numeric sweeps and held-out quality for regression and binary
models; fit timing is descriptive only.

A deterministic fit-thread and multiclass class-tree scaling harness lives at
`benchmarks/multiclass_parallelism_benchmark.py`; its committed evidence is
[multiclass_parallelism_v1.md](../benchmarks/multiclass_parallelism_v1.md).
Serial and parallel arms must produce exact artifact and prediction hashes,
complete the same rounds, emit finite probabilities, and beat the class-prior
log-loss baseline across tall/narrow, medium/wide, and small workloads.

The allocation-reuse harness at `benchmarks/allocation_reuse_benchmark.py`
compares separately built, manifest-attested native runtimes across tall/deep,
wide/deep, short/wide, and shallow/tall matrices under both growth strategies.
Its [committed evidence](../benchmarks/allocation_reuse_v1.md) requires exact
artifact and prediction digests and gates aggregate native fit time and
incremental RSS.

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

Full v1.0.0 tables, including the large-scale multi-threaded runs, are in
[docs/benchmarks/v1.0.0_comparison.md](../benchmarks/v1.0.0_comparison.md).
Across the 15 curated scenarios AlloyGBM wins 5 and places top-two on 11.

### Regression

- AlloyGBM is strongest on `histogram_stress` by a wide margin (0.1803 RMSE
  against 0.3600-0.3649 for all three peers).
- AlloyGBM leads on `bike_sharing` (68.72 RMSE vs LightGBM 70.00,
  XGBoost 70.95, CatBoost 75.80) and `dow_jones_financial` (3.3944 vs
  CatBoost 3.4517, XGBoost 3.5226, LightGBM 3.5559).
- AlloyGBM is second on `abalone_regression` and `dense_numeric`.
- AlloyGBM trails on `california_housing` (0.4933 vs LightGBM 0.4670) and is
  last of four on `panel_time_series` (32.51 vs CatBoost 31.17).

> Earlier releases claimed `panel_time_series` as AlloyGBM's strongest
> scenario. That claim came from a run in which LightGBM's bagging was
> silently disabled and its trees were capped at half the peers' capacity;
> once both were corrected the ordering changed. It is recorded here rather
> than quietly dropped.

### Classification

- AlloyGBM is second on `synthetic_classification` (0.9842 vs CatBoost 0.9858)
  and third on `adult_income` (0.8676 vs LightGBM 0.8697) — all four libraries
  within 0.8 percentage points.
- AlloyGBM trails on the small `breast_cancer` set (0.9474 vs LightGBM 0.9737;
  455 training rows, so single-digit prediction flips move the metric).
- Multiclass: second on `digits_multiclass` and `synthetic_multiclass`, tied
  first on `wine_multiclass`.

### Ranking

- AlloyGBM leads `synthetic_ranking` (NDCG@10 1.0000) and is second on
  `california_ranking` (0.7674), ahead of CatBoost (0.7461) and LightGBM
  (0.7430), behind XGBoost (0.7727).
- The `lambdarank_normalize=True` default introduced in v1.0.0 is what moved
  `california_ranking` from 0.6547 to 0.7674. On that dataset (median 120,
  max 3,307 documents per query) raising `lambdarank_truncation_level` above
  the default 30 gains a further ~0.04 NDCG@10 at ~1.5x the fit time.

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
- Active scalar numeric DRO split selection is exhaustive and safe-SIMD; native
  categorical, factor-penalized, Morph+DRO, and joint-output paths retain their
  documented scalar or standard-gain behavior. See the PR #135 report for
  measured scanner and fit-time evidence.

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

Review-evidence capture:

```bash
python3 benchmarks/review_guardrails.py --gate \
  --output docs/benchmarks/review_guardrails_v1.md
```

Scalar monotone-constraint acceptance:

```bash
python3 benchmarks/monotone_constraints_benchmark.py --quick --gate
python3 benchmarks/monotone_constraints_benchmark.py \
  --gate \
  --output docs/benchmarks/monotone_constraints_v1.md
```

See the full runner guide in [benchmarks/README.md](../../benchmarks/README.md).

## How To Interpret The Results

Use the benchmark suite to answer two different questions:

- Where is AlloyGBM already clearly strong?
- Where does it still lag established libraries?

That second question matters. The current suite is intentionally honest about
weak spots, especially on broader real-world datasets.
