# GreedyLogSum Quantile Borders v1

- Question: does CatBoost's `GreedyLogSum` border selection beat the
  LightGBM-style greedy borders that became the default in PR #146, overall
  or on particular feature shapes? If it did, the follow-up would be a
  per-feature choice of border method.
- Build: this branch, on main after PR #146 (`8558bf4`). Both arms use the
  same build. The candidate sets `ALLOYGBM_EXPERIMENT_GREEDY_LOG_SUM_BINS=1`.
- Environment: Linux x86_64 container, Python 3.13, scikit-learn 1.9.1.
- Raw records: [`data/greedy_log_sum_v1/`](data/greedy_log_sum_v1/).

## What GreedyLogSum does

It starts from one bin and repeatedly splits the bin whose split most
increases the sum of log bin weights. Each split sits at the bin's weighted
median, rounded to a distinct-value boundary, and stops when the budget is
spent or no bin holds two distinct values. The port follows CatBoost's
weighted variant (`library/cpp/grid_creator/binarization.cpp`), with cuts at
midpoints like the greedy default. CatBoost's `1e-8` inside the log is
dropped, so cuts are exactly invariant to rescaling the weights.

When a feature's distinct values fit the budget, both methods give every
value its own bin, so they differ only on features with more distinct values
than bins.

## Result: no improvement

### Default-quality suite (19 datasets, 5 seeds)

```text
python benchmarks/default_quality.py compare greedy_alloy.json greedy_log_sum_alloy.json \
    --baseline alloy --candidate alloy
geomean loss ratio 0.9998, Wilcoxon p=0.71, improved 8, regressed 6
```

| Dataset | Loss ratio (GreedyLogSum / greedy) | Seed SE (log) |
|---|---:|---:|
| `linear_interaction@scale0.001` | 0.991 | 0.004 |
| `breast_cancer` | 0.992 | 0.038 |
| `friedman1` (and both rescaled variants) | 0.996 to 0.999 | 0.003 to 0.006 |
| `linear_interaction` (and `@scale1000`) | 0.997 | 0.007 |
| `diabetes` (all variants), `digits`, `wine` | 1.000 (identical cuts) | 0.000 |
| `make_classification`, `make_multiclass`, `imbalanced_binary` | 1.000 to 1.001 | 0.004 to 0.008 |
| `hastie_10_2` | 1.003 | 0.008 |
| `skewed_discrete` | 1.004 | 0.010 |
| `small_wide_noisy` | 1.008 | 0.009 |
| `zero_inflated` | 1.010 | 0.009 |

Every dataset is within about one seed standard error of 1.0.

### Targeted feature shapes (4 fixtures, 10 seeds)

[`benchmarks/greedy_log_sum_shapes.py`](../../benchmarks/greedy_log_sum_shapes.py)
builds one feature per fixture chosen to separate the two methods, 20,000
rows each:

| Fixture | Loss ratio (GreedyLogSum / greedy) | Seed SE (log) | GreedyLogSum wins |
|---|---:|---:|---:|
| `spike` (5% mass point inside a normal) | 0.9997 | 0.0011 | 6 / 10 |
| `rounded_600` (about 600 distinct values) | 1.0025 | 0.0016 | 3 / 10 |
| `mass0_lognormal` (30% zeros, heavy tail) | 1.0034 | 0.0010 | 1 / 10 |
| `zipf_flat` (Zipf 1.1, 5,000 distinct) | 0.9994 | 0.0005 | 7 / 10 |

The largest difference is GreedyLogSum being 0.3% worse on
`mass0_lognormal`. No fixture shows a gain worth a per-feature selector.

## Conclusion

On every shape tried, GreedyLogSum and greedy borders give practically the
same loss. That matches what the borders look like: both isolate heavy
values and spread the rest nearly evenly. The large binning gain came from
leaving equal-frequency cuts (PR #146). Border selection is now an unlikely
source of CatBoost's remaining lead at defaults. A per-feature choice
of border method is not worth building on this evidence. A per-feature bin
count remains untested.

## Reproduce

```bash
python benchmarks/default_quality.py run --arms alloy --seeds 5 --output greedy_alloy.json
ALLOYGBM_EXPERIMENT_GREEDY_LOG_SUM_BINS=1 python benchmarks/default_quality.py run \
    --arms alloy --seeds 5 --output greedy_log_sum_alloy.json
python benchmarks/default_quality.py compare greedy_alloy.json greedy_log_sum_alloy.json \
    --baseline alloy --candidate alloy
python benchmarks/greedy_log_sum_shapes.py shapes_greedy.json
ALLOYGBM_EXPERIMENT_GREEDY_LOG_SUM_BINS=1 python benchmarks/greedy_log_sum_shapes.py \
    shapes_greedy_log_sum.json
```
