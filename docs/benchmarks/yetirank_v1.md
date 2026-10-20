> Historical bespoke-benchmark evidence. The harness is [archived](../archive/benchmarks/pre-tabarena-2026-10-20/README.md); these are not standard TabArena results.

# YetiRank v1: the published objective, and what it changes

**Date:** 2026-10-16 · **Base:** main at 6fb1866 · **Script:** `benchmarks/yetirank_comparison.py`

`ranking_objective="yetirank"` is now the published YetiRank (Gulin,
Kuralenok & Pavlov 2011) as CatBoost implements it, checked line by line
against CatBoost's `yetirank_helpers.cpp` and `TPairLogitError`: standard
Logistic noise on the scores, 10 noisy rankings per query per round, pairs
only between neighbours in each noisy ranking, pair weight
`|r_a - r_b| * 0.85^p`, and a pairwise logistic loss on the un-noised
scores. The old objective, LambdaMART NDCG over all pairs on
uniform-noise-smoothed scores, is unchanged under the name
`rank:ndcg_smoothed`.

Result: at the default (auto) policy, the new YetiRank ties or beats the old
objective on three small learning-to-rank sets and loses 0.009 NDCG@10 on
the MSLR-WEB10K slice, while fitting 4x faster there. The MSLR loss is not
specific to YetiRank: every Alloy ranking objective gains about 0.016 there
from leaf L2 regularization, which the auto policy sets to 0 for ranking,
and at `lambda_l2=100` the new YetiRank, the old objective and `rank:ndcg`
all reach 0.497 (CatBoost YetiRank: 0.500).

## Test NDCG@10, default policy

Learning rate 0.1, depth 6, fixed rounds (300 on MSLR, 200 elsewhere),
5 seeds. The last column is new YetiRank minus the old objective, paired by
seed, with its standard error.

| Dataset | Train rows / queries | yetirank (new) | rank:ndcg_smoothed (old yetirank) | rank:ndcg | CatBoost YetiRank | LightGBM lambdarank | new − old |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSLR-WEB10K slice (CatBoost test data) | 70,823 / 600 | 0.4813 | 0.4900 | 0.4809 | 0.4996 | 0.4856 | −0.0087 ± 0.0022 |
| LightGBM `examples/lambdarank` | 3,005 / 201 | 0.7588 | 0.7498 | 0.7505 | 0.7594 | 0.7544 | +0.0090 ± 0.0055 |
| CatBoost `pytest/data/querywise` | 1,500 / 91 | 0.6714 | 0.6745 | 0.6639 | 0.6847 | (needs integer labels) | −0.0031 ± 0.0023 |
| `benchmarks/synthetic_ranking` (80/20 query split) | 4,000 / 160 | 0.9981 | 0.9975 | 0.9982 | 0.9980 | 0.9981 | +0.0006 ± 0.0005 |

Median fit time on MSLR: new YetiRank 13.4 s, old objective 55.0 s,
`rank:ndcg` 12.4 s, CatBoost 41.1 s, LightGBM 8.3 s (4 cores). The old
objective scores every label-differing pair in a query; YetiRank scores
only neighbours.

## Leaf L2 is the bigger lever

| MSLR slice, `lambda_l2` | 0 (auto, 5 seeds) | 10 (3 seeds) | 100 (3 seeds) |
| --- | ---: | ---: | ---: |
| yetirank (new) | 0.4813 | 0.4893 | 0.4972 |
| rank:ndcg_smoothed | 0.4900 | 0.4936 | 0.4972 |
| rank:ndcg | 0.4809 | 0.4891 | 0.4968 |

YetiRank at 300 drops back to 0.4910 and at 1000 to 0.4724. On the three
small sets `lambda_l2=10` is neutral to helpful and 100 starts to hurt (for
example querywise YetiRank: 0.673, 0.689, 0.673 at 0, 10, 100), so no single
constant is right for every size. Changing the ranking auto policy is left
for a separate, suite-wide calibration rather than this change.

## Why Alloy's YetiRank trails CatBoost's at λ = 0

Inside CatBoost, YetiRank beats LambdaMart on the same slice (0.4996 vs
0.4898, 3 seeds), which matches the paper. CatBoost YetiRank with
depthwise trees, no random strength and no bagging still scores 0.4916, so
tree shape and randomization explain only part of the gap. The rest is most
likely split scoring: CatBoost chooses splits on first derivatives only and
uses the Hessian just for leaf values, whereas Alloy's Newton gain
`G²/H` favours leaves whose summed Hessian is tiny. YetiRank's
neighbour-only, position-decayed pairs leave most documents in a query with
near-zero Hessian, which is exactly the case leaf L2 tames. This is
inferred from the CatBoost source and the L2 sweep, not tested directly.

## Reproduce

The data are fixtures that ship with the reference libraries' sources; see
the script's docstring for the layout.

```bash
python benchmarks/yetirank_comparison.py DATA_DIR results.json 0,1,2,3,4
python benchmarks/yetirank_comparison.py --summarize results.json
```

Raw results: `docs/benchmarks/data/yetirank_v1/`.
