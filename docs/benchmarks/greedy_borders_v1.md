# Greedy Quantile Borders v1

- Harness: `benchmarks/default_quality.py` (every library at its own defaults,
  5 seeds, 75/25 split, test loss: log loss for classification, RMSE for
  regression).
- Build: this branch, on main after PR #145 (`aacf83f`). The baseline arm is
  the same build with `ALLOYGBM_EXPERIMENT_EQUAL_FREQUENCY_BINS=1`, which
  restores the previous equal-frequency cuts.
- Environment: Linux x86_64 container, Python 3.13, scikit-learn 1.9.1,
  LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
- Raw records: [`data/greedy_borders_v1/`](data/greedy_borders_v1/).

## What changed

`continuous_binning_strategy="quantile"` (the default) now chooses borders
the way LightGBM's `GreedyFindBin` does (with `min_data_in_bin = 1`):

1. If a feature has no more distinct values than data bins, every distinct
   value gets its own bin.
2. Otherwise any value holding at least an equal share of the rows becomes a
   bin of its own, and the remaining budget is spread over the remaining
   rows. The target bin size is recomputed after each bin closes, and a bin
   closes early (at half the target) when the next value is a heavy one.
3. Each border sits halfway between two neighbouring distinct training
   values, so an unseen value lands in the nearer bin.

Equal-frequency cuts placed a cut every `rows / bins` rows. On a skewed
feature several cuts land on the same heavy value; the duplicates were
dropped and their share of the budget was never reused. On the
`skewed_discrete` fixture's Zipf feature only 48 of 255 data bins were
used; greedy borders use 234, one per distinct value.

Borders are still stored as explicit cut values with the same meaning
(`bin = number of cuts <= value`), so models fitted before this change keep
their cuts and predict bit-for-bit as before. Weighted fits use the same rule
with each value's weight share rescaled to row units. The pre-binned path
(every feature a nonnegative integer) is unchanged.

## Gate: greedy vs equal-frequency (AlloyGBM defaults only)

```text
python benchmarks/default_quality.py compare equal_frequency_alloy.json greedy_alloy.json \
    --baseline alloy --candidate alloy --gate
geomean loss ratio 0.9825, Wilcoxon p=0.52, improved 9, regressed 9 -> Gate: failed (p)
```

| Dataset | Loss ratio (greedy / equal-frequency) | Seed SE (log) |
|---|---:|---:|
| `skewed_discrete` | **0.827** | 0.020 |
| `wine` | 0.860 | 0.137 |
| `zero_inflated` | 0.985 | 0.009 |
| `small_wide_noisy` | 0.987 | 0.007 |
| `make_multiclass` | 0.988 | 0.008 |
| `friedman1` (and both rescaled variants) | 0.992 to 0.995 | 0.005 |
| `make_classification` | 0.996 | 0.004 |
| `digits` (pre-binned integers, untouched) | 1.000 | 0.000 |
| `imbalanced_binary` | 1.000 | 0.008 |
| `diabetes` (and both rescaled variants) | 1.003 to 1.006 | 0.007 |
| `hastie_10_2` | 1.005 | 0.002 |
| `linear_interaction` (and both rescaled variants) | 1.010 to 1.013 | 0.008 |
| `breast_cancer` | 1.012 | 0.032 |

No dataset regresses significantly against seed noise. The Wilcoxon part of
the gate fails because the change is targeted: on smooth continuous features
greedy borders are essentially equal-frequency borders, so most datasets sit
within noise of 1.0. The largest apparent regression, `linear_interaction`,
was re-run over 20 seeds to check it:

| Dataset (20 seeds) | Loss ratio | Seed SE (log) | Greedy wins |
|---|---:|---:|---:|
| `linear_interaction` | 1.004 | 0.004 | 9 / 20 |
| `friedman1` | 1.000 | 0.003 | 8 / 20 |
| `diabetes` | 1.000 | 0.003 | 12 / 20 |

All three are neutral, so the 5-seed 1.1% on `linear_interaction` was noise.

## Where defaults now stand

Loss normalized to the best library per dataset (1.0 = best), geometric mean
over the 19 datasets. All arms share the same splits and seeds:

| AlloyGBM, equal-frequency | AlloyGBM, greedy | LightGBM | XGBoost | CatBoost |
|---:|---:|---:|---:|---:|
| 1.236 | **1.215** | 1.222 | 1.318 | 1.001 |

The equal-frequency arm reproduces PR #145's 1.236 exactly. With greedy
borders AlloyGBM's defaults move ahead of LightGBM's. On `skewed_discrete`
AlloyGBM goes from 1.459 to 1.208, now ahead of LightGBM (1.248) and behind
XGBoost (1.079) and CatBoost (1.000). `zero_inflated` barely moves (1.689 to
1.662) even though its skewed feature goes from 104 bins to 255. Its zeros
already had a bin of their own under equal-frequency cuts, so most of the
remaining gap there is not a binning problem.

## Side effect: exact feature bundling under the default strategy

Bundle discovery treats bin 0 as a feature's empty value. Equal-frequency
cuts could put a one-hot column's zeros above bin 0, so
`feature_bundling="exact"` silently did nothing unless binning was
`"linear"`. Greedy borders give a `{0, 1}` column one bin per value, so a
nonnegative sparse column's zeros now occupy bin 0 and bundling works under
the default `"quantile"` strategy.

## Follow-ups

- Per-feature choice between equal-frequency, greedy and CatBoost's
  GreedyLogSum, chosen by data shape.
- LightGBM also forces a border around zero for features with both signs
  (`FindBinWithZeroAsOneBin`). Not ported: zero already gets its own bin here
  when it is a heavy value, and the main reason for it in LightGBM is sparse
  storage.

## Reproduce

```bash
ALLOYGBM_EXPERIMENT_EQUAL_FREQUENCY_BINS=1 python benchmarks/default_quality.py run \
    --arms alloy --seeds 5 --output equal_frequency_alloy.json
python benchmarks/default_quality.py run --arms alloy --seeds 5 --output greedy_alloy.json
python benchmarks/default_quality.py compare equal_frequency_alloy.json greedy_alloy.json \
    --baseline alloy --candidate alloy --gate
python benchmarks/default_quality.py run --arms lightgbm xgboost catboost \
    --seeds 5 --threads 2 --output peers.json
```
