# Auto Rounds v1: Round Count and Learning Rate by Early Stopping

- Harness: `benchmarks/default_quality.py` (every library at its own defaults,
  5 seeds, 75/25 split, test log loss for classification, RMSE for
  regression) and `benchmarks/auto_rounds_variants.py` (the design sweep).
- Base: `main` at `fb60583` (after PR #147). Candidate: this branch.
- Environment: Linux x86_64 container, 4 threads, Python 3.13, scikit-learn
  1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
- Raw records: [`data/auto_rounds_v1/`](data/auto_rounds_v1/).

## What changed

`n_estimators` and `learning_rate` now default to `"auto"` on
`GBMRegressor`, `GBMClassifier` and `GBMRanker`. Under
`training_policy="auto"`:

| Situation | Rounds | Learning rate | Refit |
|---|---|---|---|
| No `eval_set` | early stopping on a seeded 10% holdout (stratified by class, whole query groups for rankers), cap 2,000, patience 50, 0.1% relative tolerance | 0.05 | yes, on every row, rounds × 1/0.9 |
| `eval_set` given | early stopping on `eval_set`, cap 2,000, patience 50 | 0.05 | no |
| Integer `n_estimators` | exactly as requested | 0.1 unless set | no |
| Manual policy, warm start / `init_model`, DART, MorphBoost, LR schedules, custom objectives; without `eval_set` also neutralization, `time_index`, < 50 rows | 100 (the old default) | 0.1 (continuations reuse the earlier fit's rate) | no |

`early_stopping_rounds` sets the patience, an explicit `learning_rate`
replaces 0.05, and `auto_rounds_` records the mode and choices.
`MultiLabelGBMRanker` keeps 100 rounds at 0.1 in both modes.

Why: "many small steps, stopped by validation" is how strong GBDT models are
trained in practice, and the default-quality suite showed the fixed 100
rounds at 0.1 underfitting most datasets (`hastie_10_2` wants about 1,500
rounds at 0.05). CatBoost wins at defaults partly for the same reason: 1,000
rounds with a learning rate it picks from the data.

## Gate: auto vs the old defaults, same build

```text
python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --output final_depth6.json
python benchmarks/default_quality.py compare final_depth6.json --baseline alloy-fixed --candidate alloy --gate
alloy vs alloy-fixed: geomean loss ratio 0.9184, Wilcoxon p=3.8e-05, improved 17, regressed 2 -> Gate: passed
```

| Dataset | Loss ratio (auto / fixed) | Seed SE (log) |
|---|---:|---:|
| `hastie_10_2` | 0.547 | 0.027 |
| `zero_inflated` | 0.812 | 0.005 |
| `digits` | 0.871 | 0.033 |
| `linear_interaction` (and ×1000) | 0.896 | 0.005 |
| `skewed_discrete` | 0.904 | 0.011 |
| `breast_cancer` | 0.939 | 0.017 |
| `imbalanced_binary` | 0.943 | 0.008 |
| `linear_interaction@scale0.001` | 0.954 | 0.004 |
| `diabetes` (and both rescaled variants) | 0.956 to 0.961 | 0.010 |
| `friedman1` (and both rescaled variants) | 0.981 to 0.989 | 0.004 |
| `make_multiclass` | 0.985 | 0.006 |
| `make_classification` | 0.997 | 0.005 |
| `wine` | 1.004 | 0.250 |
| `small_wide_noisy` | 1.011 | 0.016 |

Neither regression is significant against seed noise (`wine` has 133
training rows, so its 10% holdout is 13 rows).

## Where defaults now stand, by depth

Loss normalized to the best library per dataset (1.0 = best), geometric mean
over 19 datasets, 5 seeds. "Depth 6" is every library at its own defaults
(AlloyGBM, XGBoost and CatBoost use depth 6; LightGBM uses 31 leaves). The
depth 4 and 8 runs pass `--max-depth`, which also sets LightGBM's
`num_leaves` to `2**depth` and CatBoost's `depth`.

| Depth | AlloyGBM auto | AlloyGBM fixed (old) | LightGBM | XGBoost | CatBoost |
|---|---:|---:|---:|---:|---:|
| 4 | **1.129** | 1.307 | 1.366 | 1.320 | 1.036 |
| 6 (defaults) | **1.142** | 1.236 | 1.243 | 1.341 | 1.018 |
| 8 | **1.142** | 1.211 | 1.231 | 1.344 | 1.013 |

| Gate at depth | Loss ratio | Wilcoxon p | Improved / regressed |
|---|---:|---:|---|
| 4 | 0.859 | 2.7e-05 | 17 / 2 |
| 6 | 0.918 | 3.8e-05 | 17 / 2 |
| 8 | 0.938 | 3.8e-05 | 17 / 2 |

AlloyGBM's defaults now beat LightGBM's and XGBoost's at every depth tried
(they were tied with LightGBM before). CatBoost is still clearly best; its
remaining edge is concentrated in `zero_inflated`, `linear_interaction`,
`digits` and `wine`, where its symmetric trees, L2 = 3 and 254 GreedyLogSum
borders matter more than the round count. `hastie_10_2` is the one dataset
where AlloyGBM now beats CatBoost by a wide margin (1.00 vs 1.42).

## Cost

Total fit seconds over the 95 fits at depth 6, 4 threads:

| AlloyGBM auto | AlloyGBM fixed | LightGBM | XGBoost | CatBoost |
|---:|---:|---:|---:|---:|
| 199.3 | 26.3 | 11.7 | 11.9 | 146.2 |

The holdout pass plus the refit cost about 7.6 times the old defaults on
these small datasets, and about 1.4 times CatBoost. Most of it is
multiclass: `digits` (10 classes, about 680 refit rounds) is 50 of the 199
seconds, and `wine` runs to the 2,000-round cap because its 13-row holdout
log loss keeps falling by more than 0.1% per 50 rounds as probabilities
sharpen. Users who want the old cost pass `n_estimators=100,
learning_rate=0.1`; passing an `eval_set` skips the refit.

Two follow-ups would cut this without changing quality: a relative
validation tolerance inside the engine's early stopping (today it is applied
afterwards, so the holdout pass still runs to the absolute-tolerance stop;
the variant sweep estimates about 17% less total time), and AlloyGBM's
per-round speed on small data (the fixed arm is 2.2 times LightGBM at the
same 100 rounds).

## How the design was chosen

`benchmarks/auto_rounds_variants.py` emulates each design with explicit
parameters on the same suite (5 seeds). Normalized against the variants and
the three peers together:

| Variant | Geomean | Fit seconds |
|---|---:|---:|
| Old defaults (100 rounds, 0.1) | 1.236 | 25.8 |
| Holdout, lr 0.1, cap 2,000, patience 50, keep holdout model | 1.161 | 69.2 |
| Same, refit | 1.149 | 136.7 |
| Holdout, lr 0.05, cap 3,000, patience 100, keep | 1.144 | 151.7 |
| Same, refit | **1.128** | 300.1 |
| Same, 20% holdout | 1.130 | 310.6 |
| Holdout with CatBoost's data-size learning rate, refit | 1.130 | 520.0 |
| CatBoost's learning rate, fixed 1,000 rounds, no holdout | 1.177 | 182.2 |
| lr 0.05, cap 1,000, patience 50, refit | 1.135 | 190.5 |
| lr 0.05, cap 1,000, patience 50, keep | 1.150 | 96.8 |
| lr 0.05, patience 100, 0.1% relative tolerance, refit | 1.132 | 264.4 |
| lr 0.05, patience 100, 0.01% relative tolerance, refit | 1.129 | 296.4 |
| lr 0.1, cap 1,000, patience 50, 0.1% relative tolerance, refit | 1.153 | 116.8 |
| LightGBM / XGBoost / CatBoost defaults | 1.243 / 1.341 / 1.019 | 11.7 / 11.9 / 146.2 |

- **Learning rate 0.05 over 0.1:** about 1.5% better at every setting.
- **Refit over keeping the holdout model:** about 1.5% better, at twice the
  cost. Kept, because accuracy at defaults is the goal; `eval_set` users get
  the cheaper path.
- **CatBoost's data-size learning rate:** no better than 0.05 overall, much
  worse on `hastie_10_2` (it picks very small rates on mid-sized data), and
  the slowest.
- **Cap and patience:** the top lr-0.05 variants are within 0.7% of each
  other, which is inside seed noise. A 1,000-round cap hurt `digits`,
  `hastie_10_2` and `wine`, which want more rounds, so the shipped design
  uses a 2,000 cap with patience 50 and a 0.1% relative tolerance (the
  tolerance stops a long, nearly flat tail from pushing the refit out).

Two smaller choices: the refit scales the holdout's best round by
`1 / (1 - 0.1)`, the usual correction for training on more rows per round;
and the relative tolerance is applied to the objective's own validation loss
(log loss, MSE, ...), so it does not depend on the target's units.

## Known limitations

- The early-stopping decision inherits the known small-target-scale effect
  from `default_quality_v1.md`: `linear_interaction@scale0.001` chooses fewer
  rounds than the unit-scale data (1.46 vs 1.26 normalized). The suspected
  cause is still the absolute floor in the split tie-break tolerance.
- These are small and medium datasets (178 to 20,000 rows). Large datasets
  pay the refit too; the cap bounds a fit at about 4,200 rounds in total
  (2,000 on the holdout plus at most 2,222 in the refit).

## Reproduce

```bash
python benchmarks/auto_rounds_variants.py default,es_refit_lr0.05 5 variants.json
python benchmarks/default_quality.py run --arms lightgbm xgboost catboost --seeds 5 --threads 4 --output peers.json
python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --output final_depth6.json
python benchmarks/default_quality.py compare final_depth6.json --baseline alloy-fixed --candidate alloy --gate
for d in 4 8; do
  python benchmarks/default_quality.py run --arms alloy alloy-fixed lightgbm xgboost catboost \
      --seeds 5 --threads 4 --max-depth $d --output final_depth$d.json
done
```
