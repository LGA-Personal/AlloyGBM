# Default Quality v1: Objective-Aware, Scale-Free Auto Policy

- Harness: `benchmarks/default_quality.py` (every library at its own defaults,
  5 seeds, 75/25 split, test loss: log loss for classification, RMSE for
  regression).
- Base build: `a35bc42` (main after PR #143). Candidate: this branch.
- Environment: Linux x86_64 container, 2 threads, Python 3.13, scikit-learn
  1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
- Raw records: [`data/default_quality_v1/`](data/default_quality_v1/).

## What changed

Under `training_policy="auto"`:

| Objective family | Before | After |
|---|---|---|
| Binary classification | no leaf regularization | `min_child_hessian = 1.0` (tapered below 64 rows) |
| Multiclass softmax | no leaf regularization | leaf `lambda_l2 = 1.0` |
| Regression | small-wide L2 = 2 only when target variance > 4 | small-wide L2 = 2 by shape alone |
| Ranking | unchanged | unchanged |
| All | absolute `min_split_gain` floor | no implicit floor |

A nonzero `lambda_l1`, `lambda_l2` or `min_child_hessian` turns the auto
values off. `training_policy="manual"` is unchanged.

The two removed conditions both depended on the target's units: gain is
measured in squared target units, so a fixed floor (and a fixed variance
threshold) changes the model when `y` is rescaled. L2 and
`min_child_hessian` are in Hessian units, which do not change with the
target scale for squared error, and are bounded for log loss.

## Gate: candidate vs base (AlloyGBM defaults only)

```text
python benchmarks/default_quality.py compare base_alloy.json candidate_alloy.json \
    --baseline alloy --candidate alloy --gate
geomean loss ratio 0.8157, Wilcoxon p=0.0034, improved 10, regressed 2 -> Gate: passed
```

| Dataset | Loss ratio (cand / base) | Seed SE (log) |
|---|---:|---:|
| `breast_cancer` | 0.210 | 0.162 |
| `wine` | 0.276 | 0.410 |
| `linear_interaction@scale0.001` | 0.451 | 0.012 |
| `digits` | 0.899 | 0.072 |
| `make_multiclass` | 0.955 | 0.017 |
| `diabetes` (and both rescaled variants) | 0.978 | 0.025 |
| `imbalanced_binary` | 0.984 | 0.006 |
| `hastie_10_2` | 0.995 | 0.003 |
| `friedman1`, `linear_interaction`, `skewed_discrete`, `zero_inflated` (+ variants) | 1.000 | 0.000 |
| `small_wide_noisy` | 1.004 | 0.010 |
| `make_classification` | 1.009 | 0.007 |

Neither regression is significant against seed noise.

## Where defaults now stand

Loss normalized to the best library per dataset (1.0 = best), geometric mean
over the 19 datasets. Competitor runs are shared; the "before" AlloyGBM
column scores the base build against them:

| AlloyGBM auto, before | AlloyGBM auto, after | AlloyGBM manual | LightGBM | XGBoost | CatBoost |
|---:|---:|---:|---:|---:|---:|
| 1.514 | **1.236** | 1.544 | 1.222 | 1.318 | 1.001 |

AlloyGBM's defaults now beat XGBoost's and are within about 1% of LightGBM's.
CatBoost's defaults (ordered boosting, 1000 rounds with an automatic learning
rate, symmetric trees) remain clearly best; closing that gap is the job of the
later auto-rounds/learning-rate work.

Largest remaining gaps at defaults: `wine` 1.82, `zero_inflated` 1.69,
`linear_interaction@scale0.001` 1.51, `skewed_discrete` 1.46, `digits` 1.39,
`hastie_10_2` 1.31.

## Why these values

Each value was chosen from explicit-parameter sweeps on the base build
(`sweep_*.json`), loss relative to the base default:

| Dataset | L2=1 | L2=1, mch=1 | L2=3, mch=1 | mch=1 | mch=0.5 |
|---|---:|---:|---:|---:|---:|
| `breast_cancer` (binary) | 0.228 | 0.205 | 0.202 | 0.213 | 0.234 |
| `hastie_10_2` (binary) | 1.057 | 1.058 | 1.096 | **0.995** | 1.002 |
| `imbalanced_binary` (binary) | 0.969 | 0.958 | 0.940 | 0.988 | 0.996 |
| `make_classification` (binary) | 0.999 | 0.998 | 0.996 | 1.009 | 1.004 |
| `digits` (multiclass) | **0.881** | 1.002 | 1.064 | | |
| `wine` (multiclass) | **0.281** | 0.340 | 0.321 | | |
| `make_multiclass` (multiclass) | 0.951 | 0.946 | 0.947 | | |

- Binary: `min_child_hessian = 1` alone keeps nearly all of the
  `breast_cancer` gain and is the only arm that does not hurt `hastie_10_2`.
  An earlier candidate with L2 = 1 for every objective failed the gate on
  `hastie_10_2` (+5.8%).
- Multiclass: a Hessian floor blocks many softmax splits (per-class
  Hessians are small) and hurts `digits`; L2 = 1 helps every multiclass
  fixture.
- Regression: L2 = 1 moved most fixtures by under 1% in both directions and
  hurt `linear_interaction` by about 2%, so regression gets no new default.
- The binary floor tapers linearly below 64 rows, because a binary row
  carries at most 0.25 Hessian; a full floor would leave toy datasets with no
  legal split.

## Known remaining scale dependence

`linear_interaction@scale0.001` still scores 1.51 vs 1.40 at unit scale
(`friedman1@scale0.001` shows the same effect, smaller). The likely cause, not
yet confirmed, is the absolute `max(1.0)` floor in the split tie-break
tolerance (`gain_materially_exceeds`), which treats every gain below about
1e-6 as a tie when targets are tiny. It is tracked as a follow-up because
changing it alters tie-breaking at every scale and needs its own gate run.

## Reproduce

```bash
# base build in a separate venv, candidate in .venv
python benchmarks/default_quality.py run --arms alloy --seeds 5 --threads 2 --output base_alloy.json
python benchmarks/default_quality.py run --arms alloy --seeds 5 --threads 2 --output candidate_alloy.json
python benchmarks/default_quality.py compare base_alloy.json candidate_alloy.json \
    --baseline alloy --candidate alloy --gate
python benchmarks/default_quality.py run --arms alloy alloy-manual lightgbm xgboost catboost \
    --seeds 5 --threads 2 --output candidate_all_libraries.json
```
