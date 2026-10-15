# Multiclass gap to CatBoost (v1)

**Question.** In the matched-capacity OpenML run on main 629f69f (depth 6, lr 0.05,
up to 5,000 rounds, early stopping after 50 rounds on a shared 15% validation split),
Alloy trailed CatBoost by about 12% in log loss on CC18 multiclass (3/13/17 win/tie/loss).
Binary was 3% behind and regression was even. What closes the multiclass gap in the
default (auto-policy) path?

**Answer.** Three auto-policy changes for multiclass softmax:

| Change | Before | After |
|---|---|---|
| Leaf L2 (`AUTO_LAMBDA_L2_MULTICLASS`) | 1.0 | 3.0 |
| Row subsample (`AUTO_ROW_SUBSAMPLE_MULTICLASS`) | 1.0 below 2,048 rows, 0.9 below 16k, then 0.8 | 0.8 at every size |
| Split-score noise (`AUTO_SPLIT_NOISE_MULTICLASS`, new) | none | 1.0 |

Split-score noise is new (`SplitNoise`, `BackendOps::best_split_with_noise`). It plays the
role of CatBoost's `random_strength`. When a node picks its split, each feature's best
candidate competes on `weighted_gain * max(0, 1 + strength * z)`, where `z` is a
deterministic draw with unit variance keyed by (seed, node, feature). Each class tree gets
its own key. The chosen split keeps its true gain, so `min_split_gain`, leaf values and
artifacts are unchanged. It applies only under the auto policy, only to multiclass, and only
on the standard split path (not MorphBoost or piecewise-linear shortlists).
`ALLOYGBM_EXPERIMENT_SPLIT_NOISE_STRENGTH` overrides the strength for any objective
(`0` turns it off). Manual policy and the other objectives are unchanged.

## Method

`benchmarks/multiclass_dev_suite.py` builds a dev suite of 36 PMLB multiclass datasets
plus 7 stratified 1,000-row subsamples. None of them duplicates a CC18 task, because CC18
is the held-out test set. It uses 5-fold CV (×2 repeats below 1,000 rows), the `matched`
regime above, and 4 cores in a Linux container. The metric is test log loss, with the
shifted geometric-mean ratio `(a+0.01)/(b+0.01)` across datasets. W/L counts paired
differences larger than 2 SE and 0.5%. CatBoost 1.2.10 and Alloy main 629f69f were used.

## CatBoost ablation (matched regime, 10 small datasets, each row = CatBoost minus one default)

| CatBoost change | Loss vs CatBoost defaults | W/L vs defaults |
|---|---:|---:|
| `random_strength` 1 → 0 | 1.066 | 1/4 |
| `bootstrap_type` Bayesian → No | 1.040 | 1/3 |
| `score_function` Cosine → L2 | 1.017 | 1/3 |
| `leaf_estimation_method` Newton → Gradient | 1.014 | 0/2 |
| `grow_policy` SymmetricTree → Depthwise | 1.011 | 2/2 |
| `l2_leaf_reg` 3 → 1 | 1.005 | 1/1 |

Split-score noise is the largest single contributor to CatBoost's lead. Its stochastic row
weighting is next.

## Alloy knobs (matched regime; loss vs Alloy main, and vs CatBoost)

| Alloy config | Datasets | Loss vs main | Loss vs CatBoost | W/L vs main |
|---|---:|---:|---:|---:|
| main (L2 1, no subsample below 2k rows) | 38 | 1.000 | 1.095 | |
| Hessian × K/(K−1) (LightGBM) | 38 | 1.005 | | 1/4 |
| Hessian × 2 (XGBoost) | 38 | 1.013 | | 3/13 |
| lambda_l2 3 | 38 | 0.985 | | 18/4 |
| row_subsample 0.8 | 38 | 0.971 | | 15/4 |
| per-row Bayesian bootstrap weights (prototype) | 38 | 0.982 | 1.076 | 15/6 |
| L2 3 + subsample 0.8 | 38 | 0.960 | 1.052 | 22/5 |
| L2 3 + subsample 0.7 | 38 | 0.957 | 1.048 | 22/4 |
| L2 5 + subsample 0.8 | 38 | 0.957 | 1.048 | 22/5 |
| L2 3 + subsample 0.8 + noise 0.3 (prototype) | 38 | 0.923 | 1.011 | 24/5 |

A CatBoost-style shared tree per round (one tree for all classes, K-vector leaves, through the
joint multi-output builder) did worse than K trees on the first datasets tried (iris, ecoli)
and was too slow to run further. It was dropped.

## Strength sweep (this PR's implementation, matched regime, 22 of 38 datasets)

The container restarted twice during the sweep, so it covers the 22 datasets that finished.

| Noise strength (with L2 3 + subsample 0.8) | Loss vs main | Loss vs CatBoost | W/L vs main | W/L vs CatBoost |
|---|---:|---:|---:|---:|
| 0 | 0.952 | 1.019 | 13/2 | 5/8 |
| 0.5 | 0.932 | 0.997 | 14/1 | 6/5 |
| **1.0 (shipped)** | **0.926** | **0.991** | **13/1** | **6/6** |
| 2.0 | 0.925 | 0.990 | 15/1 | 7/5 |

Strengths from 0.5 to 2 are all at CatBoost parity on the dev suite. 1.0 matches
CatBoost's default. The noise adds no histogram or scan work: it only changes how a node
compares the per-feature best candidates.

## Not yet measured

- OpenML CC18 multiclass confirmation (held-out test set). To be run on the Mac with PR head
  vs main, defaults and `early_stopping` regimes.
