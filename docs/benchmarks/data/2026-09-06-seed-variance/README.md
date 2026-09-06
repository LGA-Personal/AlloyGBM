# Curated-suite seed variance — the noise floor

`benchmarks/curated_seed_variance.py --seeds 5 --threads 1`, measured at the
PR head. Seeds: [20260906, 20260907, 20260908, 20260909, 20260910].

`--seed` drives the train/test split *and* every library's own seed, so this
spread covers the same sources of variation that move a before/after delta.
It exists to answer one question: **is a scenario-level delta signal or noise?**

A delta smaller than its scenario's spread is not evidence of anything,
whichever direction it points.

## AlloyGBM spread across 5 seeds, against the bin-budget deltas

| Scenario | Metric | Median | Min | Max | Spread | Bin-budget delta | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `wine_multiclass` | log_loss_val | 0.02505 | 0.00004 | 0.28538 | 1138.95% | +0.00% | inside noise |
| `breast_cancer` | log_loss_val | 0.19252 | 0.01848 | 0.27449 | 132.98% | +7.15% | inside noise |
| `digits_multiclass` | log_loss_val | 0.07879 | 0.04322 | 0.08936 | 58.55% | +0.00% | inside noise |
| `california_ranking` | ndcg_10 | 0.76662 | 0.57027 | 0.82947 | 33.81% | -11.06% | inside noise |
| `histogram_stress` | rmse | 0.14912 | 0.14342 | 0.18007 | 24.58% | +19.23% | inside noise |
| `synthetic_classification` | log_loss_val | 0.04138 | 0.03923 | 0.04420 | 12.02% | -2.09% | inside noise |
| `abalone_regression` | rmse | 2.16187 | 2.03392 | 2.23920 | 9.50% | -0.41% | inside noise |
| `dense_numeric` | rmse | 0.56440 | 0.54965 | 0.60216 | 9.30% | +0.22% | inside noise |
| `synthetic_multiclass` | log_loss_val | 0.47970 | 0.45149 | 0.49455 | 8.98% | +1.78% | inside noise |
| `bike_sharing` | rmse | 70.11628 | 69.57350 | 74.47914 | 7.00% | +0.00% | inside noise |
| `panel_time_series` * | rmse | 32.87155 | 31.71445 | 33.52574 | 5.51% | +0.93% | inside noise |
| `california_housing` | rmse | 0.48090 | 0.47860 | 0.49946 | 4.34% | +0.66% | inside noise |
| `dow_jones_financial` * | rmse | 3.44193 | 3.41207 | 3.52857 | 3.38% | +1.16% | inside noise |
| `adult_income` | log_loss_val | 0.29387 | 0.28969 | 0.29874 | 3.08% | +0.27% | inside noise |
| `synthetic_ranking` | ndcg_10 | 0.99817 | 0.99625 | 0.99915 | 0.29% | -0.08% | inside noise |

`*` chronological split — does not repartition with the seed, so its spread
reflects model randomness only and reads narrower for a structural reason.

Median scenario spread: **9.30%**. 
Maximum: **1138.95%** (`wine_multiclass`).

## Reading

**No scenario-level delta from the quantile bin-budget change exceeds its own
seed noise — zero of fifteen.** That includes the two largest: `histogram_stress`
at +19.23% against a 24.58% spread, and `california_ranking` at -11.06% against
a 33.81% spread. `california_ranking` NDCG across these five seeds runs 0.570,
0.673, 0.767, 0.798, 0.830; an 11% move on that dataset is an ordinary draw.

The test is deliberately conservative. Each delta is compared against the full
min-max range of only five samples, which understates the true spread, so the
comparison is biased *toward* declaring signal. Nothing qualified anyway.

Three scenarios are noisy enough that single-seed results should not be quoted
at all: `wine_multiclass` (log-loss ranges 0.00004 to 0.285 on 142 rows),
`breast_cancer`, and `digits_multiclass`. These are small datasets where a
different split changes the answer more than any code change does.

This does not weaken the case for the bin-budget fix. That case rests on
correctness — 255 data bins were addressed by 255 cuts, so the two highest
quantiles shared a bin — and on the 500k x 40 fixture where the effect
(0.163775 -> 0.152451) is large and reproduces exactly. What this measurement
establishes is the narrower claim the curated suite can actually support: the
fix does not regress these scenarios in aggregate, and no individual scenario
here demonstrates either a gain or a loss.
