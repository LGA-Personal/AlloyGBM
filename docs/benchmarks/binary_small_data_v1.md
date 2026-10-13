# Binary Small-Data Defaults v1: Leaf L2 and Multi-Step Newton Leaves

- Harness: `benchmarks/binary_dev_suite.py` (test log loss, paired splits,
  every peer at its own defaults).
- Base: `main` at `b6f7262` (after PR #150). Candidate: this branch.
- Environment: Linux x86_64 container, 1 thread per fit (fit times below are
  from a shared 4-core box and only roughly comparable), Python 3.13,
  LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
- Raw records: [`data/binary_small_data_v1/`](data/binary_small_data_v1/)
  (`small.json.gz`: PMLB suite, every arm tried; `large.json`: dev-suite
  `.npz` binary tasks).

## Why

The OpenML pilot (`data/frontier_matrix_v1/openml_pilot.md`) put Alloy's
binary log loss at 1.12x the best peer (geomean), the only task family above
1.0. The gap is a small-data gap: every CC18 binary task where Alloy trails by
more than 10% has fewer than 5,000 rows, and CatBoost is the peer ahead on
most of them. CC18 is the held-out test set, so this work never tuned on it.

## Development data

1. **Small binary suite** (`binary_dev_suite.py`): 43 PMLB binary datasets
   with 150 to 10,000 rows (32 real, 11 synthetic logic/epistasis sets that
   stand in for madelon), plus 1,000-row stratified subsamples of 8 larger
   ones: 51 cells. CC18/CTR23 duplicates are removed by name and shape (wdbc,
   breast-w, pima, credit-g, credit-approval, tic-tac-toe, kr-vs-kp,
   spambase, churn, phoneme, sick). Splits: stratified 5-fold, 2 repeats
   below 3,000 rows.
2. **Larger binary tasks**: the 26 binary tasks of the shared dev suite
   (`dev_suite.py`, Grinsztajn + AMLB, 3k to 20k rows), 5-fold.

Aggregate: shifted geometric mean of log-loss ratios, `(a + 0.01) / (b +
0.01)`, so near-perfect cells do not dominate. W/L counts paired differences
beyond two standard errors and 0.5%.

## Root cause

The small suite reproduces the pilot: Alloy's log loss is **1.130x
CatBoost's**, CatBoost wins 27 of 51 cells and loses 5.

Ablating CatBoost's own defaults one at a time shows what carries its lead
(ratio to CatBoost defaults; higher means removing it hurts):

| CatBoost change | Log loss vs CatBoost | W/L vs CatBoost |
|---|---:|---:|
| `l2_leaf_reg` 3 → 0.001 | 1.230 | 11/34 |
| `leaf_estimation_iterations` 10 → 1 | 1.067 | 3/29 |
| symmetric trees → depth-wise | 1.039 | 6/36 |
| `random_strength` 1 → 0 | 1.011 | 8/29 |
| no bootstrap | 1.002 | 0/4 |
| `boost_from_average=True` | 0.990 | 12/16 |
| lr 0.05 × 250 trees instead of its tiny auto rate × 1,000 | 0.992 | 12/27 |

Alloy's auto policy for binary had no leaf L2 and instead a minimum child
Hessian of 1.0. That floor caps how confident the model can get: once rows
are classified well their Hessian `p(1-p)` is small, so the leaves that would
finish the job are illegal. Near-separable cells ended 3-10x worse than with
L2 (corral 0.035 vs 0.002, monk1 0.048 vs 0.010, irish 0.006 vs 0.0002).
Alloy also sets each leaf with a single Newton step, which falls short of the
regularized optimum of a log-loss leaf.

## Candidates (small suite, vs current defaults)

| Alloy config | vs base | vs CatBoost | W/L vs base | rounds |
|---|---:|---:|---:|---:|
| current defaults (min child Hessian 1, no L2) | 1.000 | 1.130 | - | 245 |
| `lambda_l2=3` (no Hessian floor) | 0.913 | 1.033 | 15/3 | 610 |
| `lambda_l2=3, min_child_hessian=1` | 0.988 | 1.117 | 11/6 | 401 |
| `lambda_l2=1` / `5` / `10` | 0.912 / 0.930 / 0.943 | | | |
| `lambda_l2=3, min_child_hessian=0.1` | 0.913 | 1.032 | 18/3 | 513 |
| + `row_subsample=0.8` | 0.898 | 1.015 | 25/3 | 480 |
| + 2 extra Newton steps per leaf (**shipped**) | **0.874** | **0.988** | **26/2** | 522 |
| same, 1 / 4 / 9 extra steps (at lr 0.1) | 0.867 / 0.849 / 0.868 | | | |
| + learning rate 0.1 | 0.849 | 0.960 | 25/3 | 390 |
| `min_data_in_leaf=20` alone | 0.969 | 1.095 | 15/2 | 278 |
| `max_depth=4` alone | 0.993 | 1.122 | 16/2 | 284 |
| holdout patience 300 alone | 0.975 | 1.103 | 3/0 | 342 |

AUC moves the same way: the shipped config's 1 − AUC is 0.825x the base's
(geomean).

Learning rate 0.1 is better still on small data but costs 0.6% on the larger
tasks (0/5 W/L there), so the auto learning rate stays at 0.05.

## Larger tasks (26 dev-suite binary tasks, vs current defaults)

| Alloy config | vs base | vs CatBoost | W/L vs base | rounds |
|---|---:|---:|---:|---:|
| `lambda_l2=3, min_child_hessian=0.1` | 1.000 | 0.995 | 2/2 | 224 |
| + `row_subsample=0.8` | 1.001 | 0.995 | 2/2 | 217 |
| + 2 extra Newton steps (**shipped**) | 1.002 | 0.996 | 2/4 | 230 |
| L2 + 2 Newton steps, no subsample change | 1.002 | 0.997 | 5/2 | 220 |
| shipped + learning rate 0.1 | 1.013 | 1.007 | 2/5 | 123 |

On data this size Alloy was already level with CatBoost and the change is
neutral (within noise).

## What changed

Under `training_policy="auto"`, binary log loss only:

- leaf L2 `lambda_l2 = 3.0` and a minimum child Hessian of 0.1 (was 1.0 and
  no L2), still only when the caller sets none of `lambda_l2`, `lambda_l1`,
  `min_child_hessian`;
- row subsample 0.8 at every size (was 1.0 below 2,048 rows, 0.9 below
  16,384) when `row_subsample` is left at 1.0;
- after each tree is built, two more Newton steps on every leaf, on all of
  the leaf's rows: `v ← v − (G(v) + λv) / (H(v) + λ)`. Off for L1, DRO,
  MorphBoost, PL leaves and monotone constraints. The rows' scores are
  shifted by each leaf's change, so the tree is not replayed.
  `ALLOYGBM_EXPERIMENT_BINARY_LEAF_NEWTON_STEPS=k` overrides the step count
  for A/B runs (0 turns it off).

Manual policy and multiclass are unchanged.

## Held-out check: OpenML-CC18 binary tasks

Run once, after the configuration above was fixed. Setup: the 35 CC18 binary tasks, their first 3 official folds, Alloy at its defaults with 10 threads on the pilot's Apple M4. `main` (`b6f7262`) and this branch (`bd2b8db`) ran one after the other on an idle machine. Peer numbers come from the PR #150 pilot, which used the same machine and the same folds. Raw records: `data/binary_small_data_v1/cc18_main.json.gz` and `cc18_pr155.json.gz`.

| | `main` | this branch |
|---|---:|---:|
| Log loss vs best peer (geomean) | 1.088 | **0.987** |
| Log loss vs CatBoost (shifted geomean) | 1.026 | **0.951** |
| Tasks better / worse than CatBoost (0.5%) | 10 / 23 | 16 / 16 |
| 1 − AUC vs `main` (geomean) | 1.000 | 0.880 |
| Mean trees | 300 | 447 |
| Fit time vs fastest peer (geomean) | 2.00x | 3.81x |

Compared with `main`, log loss is 0.906x (geomean). It improves on 29 tasks and gets worse on 6:

| Task | Change |
|---|---:|
| climate-model | +11.8% |
| sick | +7.1% |
| churn | +5.1% |
| electricity | +4.7% |
| madelon | +3.7% |
| bank-marketing | +0.2% |

The largest gains are on tic-tac-toe (−57%), banknote (−56%), kr-vs-kp (−40%), wdbc (−25%), breast-w (−18%) and wilt (−18%).

Fit time is 1.90x `main` (geomean), mostly from the extra rounds. The worst case is banknote, which went from 0.08 s to 0.64 s. Even so, fits still take 0.30x CatBoost's time.

## Cost

More rounds before the holdout stops (small suite 245 → 522 on average,
larger tasks 185 → 230), mostly on near-separable cells whose log loss keeps
falling. The Newton steps cost one routing pass and two scoring passes over
the rows per round: on 200 rounds, 4 threads, 20 features, fit time went from
0.97 s to 1.10 s at 20k rows and from 5.6 s to 6.7 s at 200k rows.
