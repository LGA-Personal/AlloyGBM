> Historical bespoke-benchmark evidence. The harness is [archived](../archive/benchmarks/pre-tabarena-2026-10-20/README.md); these are not standard TabArena results.

# Frontier matrix v1

`benchmarks/frontier_matrix.py` asks one question the other suites do not:
**in which situations is AlloyGBM better or worse than LightGBM, XGBoost and
CatBoost?** It sweeps the data a user can hand a GBDT library (size, shape,
task, feature types, categorical columns, missing values, signal and noise)
and the hyperparameters they can choose (depth, tree count, learning rate,
regularization, growth strategy, early stopping), and reports a win/tie/loss
for AlloyGBM against each peer in every cell.

## What the suite covered before

| Suite | What it answers | Gaps for this question |
|---|---|---|
| `default_quality.py` | Each library at its defaults, 5 seeds, ~20 datasets, paired gate | Only defaults; every dataset under 20k rows; no categorical columns, no missing values, no ranking, no depth / tree-count / regularization sweep |
| `auto_policy_benchmark.py` | Alloy's auto policy against manual Alloy arms across row/feature shapes | Alloy only; no peers |
| `competitiveness/` (`pr_smoke`, `deep_scaling`) | Fit time and memory at one matched configuration (lr 0.05, depth 6, 40 rounds) | Timing first, one regime; the deep-scaling grid is regression only |
| `run_model_comparison.py` | Curated real datasets with a shallow/mid/deep profile grid | Needs downloads; single target encoding path; fixed shape per dataset, so size and type effects are confounded |
| `scale_comparison.py`, `deep_scaling_comparison.py` | Rows x threads, rows x features x depth | One synthetic regression target |

Each answers its own question well, but none varies one property of the
situation while holding the rest fixed, so a loss could not be attributed to
"high-cardinality categoricals" or "deep trees" rather than to a particular
dataset.

## How industry benchmarks are built

- **szilard/GBM-perf and benchm-ml** fix one realistic dataset (airline
  delays, 0.1M / 1M / 10M rows), fix the hyperparameters across libraries
  (100 trees, depth 10, learning rate 0.1), and report training time and AUC
  per size and per CPU/GPU. The lesson: size sweeps on a fixed generator
  separate scaling from dataset idiosyncrasy, and matched hyperparameters
  isolate the tree learner.
- **LightGBM's and XGBoost's own experiments** (Higgs, Yahoo LTR, MS LTR,
  Expo, Allstate) fix a matched configuration (for LightGBM, 255 leaves and
  500 iterations) and report accuracy and time curves per iteration, including
  ranking and categorical-heavy datasets.
- **CatBoost's benchmarks** (Adult, Amazon, Appetency, Click, Internet, Kick,
  Upselling: mostly categorical) report log loss at *defaults* and after an
  equal hyperparameter-search budget per library. Defaults versus tuned is
  the distinction users care about.
- **OpenML-CC18 (72 classification tasks), OpenML-CTR23 (35 regression
  tasks) and the AutoML Benchmark** fix curated suites with official
  cross-validation splits, so results compare across papers; scores are
  normalized per task before averaging and ranked across methods.
- **Grinsztajn et al. (NeurIPS 2022)** curate "medium-sized" datasets (about
  10,000 training rows), split numeric-only from categorical suites and
  regression from classification, and report normalized test score against
  the random-search budget. They also show which data properties (irrelevant
  features, non-rotation-invariance, irregular targets) favour trees.
- **TabArena (2025)** is a maintained "living" benchmark of 51 curated
  datasets (500 to 250,000 rows) with repeated cross-validation, three
  regimes per model (default, tuned, tuned plus ensembled), and Elo ratings
  and normalized scores across all of them.

Common rules across all of these: many datasets and seeds or folds, proper
scoring rules on held-out data, normalization before aggregation, results
reported per regime (defaults, matched, tuned), equal thread budgets, and
time reported next to quality.

## Design

**One factor at a time around a base point.** Every synthetic scenario is the
base dataset (10,000 rows x 20 numeric features, 8 informative, a mix of
linear, nonlinear and interaction signal, noise at half the signal's
standard deviation) with one property changed. The base datasets are the
shared reference level of every sweep, so `missing = 30% informative` differs
from `base` only in its missing values. Features are monotone or lossy views
of a standard-normal latent matrix, so `skewed` (lognormal, zero-inflated)
leaves the ideal trees unchanged and isolates binning, while `discrete` and
`sparse` remove information the way coarse or zero-heavy real columns do.

| Axis | Levels (tier) |
|---|---|
| rows | 300 (smoke), 2k, 100k (quick), 1M (full) |
| columns | 5, 200, 800 with 400 rows (p > n) (quick), 1,000 (full) |
| task | regression, binary, multiclass 5 (base), imbalanced binary 5% positive, multiclass 20, ranking 20 docs/query (quick), multiclass 3, ranking 100 docs/query (full) |
| feature type | skewed, discrete, sparse 90% zeros, 95% irrelevant (quick), redundant copies (full) |
| categorical | 4 x 8 levels, 2 x 1,000 levels Zipf (quick), mixed with 15% missing (full) |
| missing | 20% random, 30% informative (MNAR) (quick), 60% random (full) |
| signal | linear, interaction (quick), additive, step (full) |
| noise | low SNR, 6x base noise (quick), near noiseless (full) |

**Hyperparameter regimes** map library-neutral settings onto each library's
own knobs (`num_leaves = 2**depth` for LightGBM depth regimes; `lossguide` /
`Lossguide` / `tree_growth="leaf"` for leaf-wise growth; Bernoulli bootstrap
for CatBoost subsampling). Settings a library cannot express (CatBoost's L1
and minimum hessian, minimum rows per leaf on CatBoost's symmetric trees and
in XGBoost) are listed per fit under `unmapped`. Data scenarios run under
`defaults` and `standard`; the regime panel (regression and binary bases in
quick; multiclass and categorical-with-missing added in full) runs every
regime.

| Regime | Settings |
|---|---|
| `defaults` | every library at its own defaults (Alloy: auto policy and auto rounds) |
| `standard` | depth 6, 300 trees, lr 0.05 |
| `stumps` (full) | depth 1, 1,000 trees, lr 0.1 |
| `shallow` | depth 2, 500 trees, lr 0.1 |
| `deep` | depth 12, 300 trees, lr 0.05 |
| `few_trees` | depth 6, 10 trees, lr 0.3 |
| `many_trees` | depth 6, 2,000 trees, lr 0.02 |
| `very_many_trees` (full) | depth 6, 5,000 trees, lr 0.1 |
| `light_reg` | depth 8, 300 trees, lr 0.1, no L1/L2, minimum hessian 0, 1 row per leaf (Alloy `training_policy="manual"`) |
| `heavy_reg` | depth 8, 300 trees, lr 0.1, L2 10, L1 1, minimum hessian 10, 100 rows per leaf, 70% row and column sampling |
| `leafwise` | best-first growth, 63 leaves, depth cap 12, 300 trees, lr 0.05 |
| `early_stopping` | lr 0.05, depth 6, up to 5,000 trees, patience 50 on a shared 15% validation split |

In matched regimes Alloy keeps its default `training_policy="auto"` for any
setting the regime leaves open, the same way the peers keep their defaults;
`light_reg` is the exception because "no regularization" must also turn off
Alloy's automatic leaf regularization.

**Scoring.** RMSE for regression, log loss for classification (AUC and
accuracy are recorded as secondary metrics), 1 - NDCG@10 for ranking, on a
25% held-out split that changes with the seed (stratified for classifiers,
whole queries for ranking). Categorical columns reach each library in its
native form: pandas `category` for AlloyGBM, LightGBM and XGBoost
(`enable_categorical=True`), string `cat_features` for CatBoost.

**Aggregation.** Per cell and arm, the seed-mean loss. AlloyGBM is compared
with the best peer in the cell (geometric mean of the loss ratio across a
slice's cells) and with each peer separately (win/tie/loss, paired by seed;
a mean log-ratio inside two seed standard errors, or under 0.5%, is a tie).
`compare` reuses `default_quality.py`'s paired Wilcoxon gate, with each cell
as one "dataset", to gate a candidate build against a baseline.

**Tiers.** `smoke` (8 cells, 1 seed; CI runs it Alloy-only), `quick` (86
cells, 3 seeds; every axis and regime at modest size), `full` (148 cells, 5
seeds; 1M rows, 1,000 columns, extra levels and the full regime panel).
`--axes`, `--regimes` and `--cells` select slices.

**Real data on official folds.** `--openml-suite` adds every task of one or
more OpenML suites: `cc18` (suite 99, 72 classification tasks), `ctr23`
(suite 353, 35 regression tasks), `grinsztajn` (suites 334 to 337, the
NeurIPS 2022 tabular benchmark) or any suite id. Each task is scored on its
own repeat/fold splits from `task.get_train_test_split_indices`, so the seed
axis becomes the fold index and per-fold numbers line up with results
published on the same task. Categorical columns come from the dataset's
`categorical_indicator`; missing values are passed through as NaN. Tasks
above `--openml-max-rows` (default 100,000) are skipped rather than
subsampled, since subsampling would break the official folds;
`--openml-train-cap` instead caps training rows inside each fold (the
Grinsztajn protocol) and leaves test folds whole. `--openml-folds`,
`--openml-tasks` and `--openml-only` narrow a run. Long runs are safe to
interrupt: `--output` is rewritten after every cell (marked
`"complete": false` until the end), and `--resume` keeps the error-free
records already there and fits only the rest. Every result records the
library versions, thread count, CPU count, platform and git commit. This
path needs network access to openml.org and the `openml` package, so CI
tests it against a stand-in module only.

```bash
python benchmarks/frontier_matrix.py run --openml-only --openml-suite cc18 ctr23 \
    --regimes defaults --threads 8 --output openml.json --report openml.md
```

## First results (quick tier)

Quick tier, 3 seeds, 4 threads per library, on a 4-core Linux container,
build `66ae711` (AlloyGBM 1.0.0 from `main` at #149, LightGBM 4.7.0,
XGBoost 3.4.1, CatBoost 1.2.10). The full report is at
[`data/frontier_matrix_v1/quick.md`](data/frontier_matrix_v1/quick.md),
with the raw records in `quick.json`. Ratios are AlloyGBM's loss divided by
the best peer's (below 1.0 means AlloyGBM is better).

| Slice | Cells | Loss vs best peer | Best or tied | W/T/L vs LightGBM | W/T/L vs XGBoost | W/T/L vs CatBoost |
|---|---:|---:|---:|---:|---:|---:|
| all cells | 84 | 1.086 | 23 | 25/44/15 | 44/26/14 | 13/13/58 |
| all except categorical | 78 | 1.031 | 23 | 25/44/9 | 44/26/8 | 13/13/52 |
| data axes at defaults, except categorical | 31 | 1.020 | 14 | | | |
| data axes at `standard`, except categorical | 31 | 1.040 | 6 | | | |

What it says:

1. **Categorical columns are the largest gap, and the cause is a bug.**
   With the default `max_cat_threshold=0`, categorical columns go through
   target encoding, but `predict` never applies that encoding
   (`_core.py` `predict` maps only native-categorical columns). Predictions
   are therefore made on raw category codes, and log loss is worse than a
   constant even on the training rows. In the categorical cells AlloyGBM
   is 1.8 to 2.8 times the best peer's loss. A follow-up run with
   `max_cat_threshold=255` ([`cats.json`](data/frontier_matrix_v1/cats.json))
   isolates it. On the 8-level columns, native splits put AlloyGBM level
   with or ahead of every peer: binary log loss 0.521 against CatBoost's
   0.521, regression RMSE 0.565 against 0.574 at `standard`. The
   1,000-level Zipf columns exceed the native limit and fall back to the
   broken path. On regression they fail outright ("encoded cardinality 753
   exceeds supported max 256").
2. **Outside categoricals AlloyGBM beats LightGBM and XGBoost more often
   than it loses to them (25/44/9 and 44/26/8 W/T/L), but trails
   CatBoost (13/13/52).** CatBoost is the best peer in 65 of 78
   cells. AlloyGBM is best or tied in 23 of those 78 cells, with a geomean
   gap of 3%.
3. **Defaults are closer than matched settings.** At defaults AlloyGBM
   wins outright on sparse features, low signal-to-noise, 20% random
   missing values and 5-class multiclass. It is level on informative
   missingness, 100k rows and 20 classes. At the matched `standard` regime
   (depth 6, 300 trees, lr 0.05) it trails CatBoost almost everywhere by 1%
   to 10%. That points at the tree learner and the leaf regularization,
   not at the auto policy.
4. **Hyperparameter frontiers:** level or ahead at `shallow` (1.007),
   `few_trees` (1.008) and `leafwise` (1.022). The weakest are `many_trees`
   (1.084) and `deep` (1.076), where binary log loss degrades against
   CatBoost (1.12 and 1.15). This is consistent with the deep review's
   calibration finding (inferred, not yet tested here): CatBoost's ordered
   boosting and default L2 of 3 resist overfitting as trees grow larger or
   more numerous.
5. **Small data:** 300 and 2,000 rows trail by 6% to 17%, again mostly to
   CatBoost.
6. **Fit time:** AlloyGBM's median fit is 2.7 times the fastest peer's. At
   defaults much of that is auto rounds, which fit a holdout model and then
   refit. At matched regimes it is 1.1 to 3.9 times, and `fit_speed_v1.md`
   tracks that work.

Suggested order of work from this: fix target-encoded categorical
prediction and lift the 256-level cap (or route high cardinality to
encoding that predicts correctly), then look at leaf regularization for
deep and many-tree binary models and at small data. Re-run this tier after
each change and gate with `compare`.
