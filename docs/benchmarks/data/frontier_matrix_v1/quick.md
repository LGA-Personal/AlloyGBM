# Frontier matrix (quick tier)

Arms: alloy 1.0.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10. Seeds: 3. Threads: 4. Commit: `66ae71126f2f`.

Loss ratios compare `alloy` with the best peer in each cell (below 1.0 means alloy is better). Win/tie/loss is per cell, paired by seed; a difference inside two seed standard errors or under 0.5% is a tie.

## Overall

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all cells | 84 | 1.086 | 23/84 | 25/44/15 | 44/26/14 | 13/13/58 | 2.72x | 2 |
| task: regression | 40 | 1.072 | 10/40 | 9/23/8 | 26/11/3 | 9/3/28 | 2.98x | 2 |
| task: binary | 38 | 1.108 | 10/38 | 13/18/7 | 15/14/9 | 3/8/27 | 2.59x | 0 |
| task: multiclass | 4 | 1.031 | 2/4 | 3/1/0 | 2/0/2 | 1/1/2 | 2.56x | 0 |
| task: ranking | 2 | 1.054 | 1/2 | 0/2/0 | 1/1/0 | 0/1/1 | 2.79x | 0 |

## Data axes, `defaults` regime (every library at its own defaults (Alloy: auto policy, auto rounds))

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base = base | 3 | 1.013 | 1/3 | 1/2/0 | 3/0/0 | 1/0/2 | 3.48x | 0 |
| categorical = 2 x 1000 levels (Zipf) | 1 | 1.843 | 0/1 | 0/0/1 | 0/0/1 | 0/0/1 | 3.09x | 1 |
| categorical = 4 x 8 levels | 2 | 1.996 | 0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 8.36x | 0 |
| columns = 200 | 2 | 1.026 | 0/2 | 0/2/0 | 2/0/0 | 0/0/2 | 4.71x | 0 |
| columns = 5 | 2 | 1.006 | 1/2 | 1/1/0 | 2/0/0 | 0/1/1 | 5.57x | 0 |
| columns = p>n | 1 | 1.084 | 1/1 | 0/1/0 | 1/0/0 | 0/1/0 | 18.72x | 0 |
| feature type = 95% irrelevant | 1 | 1.012 | 0/1 | 0/1/0 | 1/0/0 | 0/0/1 | 4.48x | 0 |
| feature type = discrete | 2 | 1.026 | 0/2 | 0/2/0 | 2/0/0 | 0/0/2 | 6.50x | 0 |
| feature type = skewed | 2 | 1.013 | 0/2 | 1/1/0 | 2/0/0 | 0/0/2 | 7.01x | 0 |
| feature type = sparse | 2 | 0.990 | 2/2 | 2/0/0 | 2/0/0 | 2/0/0 | 2.09x | 0 |
| missing = 20% random | 2 | 0.996 | 2/2 | 2/0/0 | 2/0/0 | 1/1/0 | 6.50x | 0 |
| missing = 30% informative | 2 | 0.996 | 2/2 | 1/1/0 | 2/0/0 | 1/1/0 | 5.41x | 0 |
| noise = low SNR | 2 | 0.991 | 2/2 | 2/0/0 | 2/0/0 | 2/0/0 | 2.70x | 0 |
| rows = 100k | 2 | 1.014 | 1/2 | 1/1/0 | 2/0/0 | 0/1/1 | 17.26x | 0 |
| rows = 2k | 2 | 1.060 | 0/2 | 1/1/0 | 2/0/0 | 0/0/2 | 4.05x | 0 |
| rows = 300 | 1 | 1.064 | 0/1 | 0/0/1 | 0/1/0 | 0/0/1 | 11.83x | 0 |
| signal = interaction | 1 | 1.090 | 0/1 | 1/0/0 | 1/0/0 | 0/0/1 | 24.03x | 0 |
| signal = linear | 1 | 1.029 | 0/1 | 0/0/1 | 1/0/0 | 0/0/1 | 8.74x | 0 |
| task = binary 5% positive | 1 | 1.014 | 1/1 | 1/0/0 | 1/0/0 | 0/1/0 | 2.63x | 0 |
| task = multiclass 20 | 1 | 0.998 | 1/1 | 1/0/0 | 1/0/0 | 0/1/0 | 4.16x | 0 |
| task = ranking | 1 | 1.072 | 0/1 | 0/1/0 | 0/1/0 | 0/0/1 | 3.42x | 0 |

## Data axes, `standard` regime (depth 6, 300 trees, lr 0.05)

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base = base | 3 | 1.049 | 0/3 | 1/2/0 | 1/1/1 | 0/0/3 | 2.11x | 0 |
| categorical = 2 x 1000 levels (Zipf) | 1 | 2.812 | 0/1 | 0/0/1 | 0/0/1 | 0/0/1 | 2.75x | 1 |
| categorical = 4 x 8 levels | 2 | 2.083 | 0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 2.36x | 0 |
| columns = 200 | 2 | 1.027 | 0/2 | 0/2/0 | 0/2/0 | 0/0/2 | 1.65x | 0 |
| columns = 5 | 2 | 1.032 | 0/2 | 0/2/0 | 1/1/0 | 0/0/2 | 2.30x | 0 |
| columns = p>n | 1 | 1.063 | 1/1 | 0/1/0 | 0/1/0 | 0/1/0 | 3.78x | 0 |
| feature type = 95% irrelevant | 1 | 1.029 | 0/1 | 0/1/0 | 0/1/0 | 0/0/1 | 1.92x | 0 |
| feature type = discrete | 2 | 1.036 | 0/2 | 1/1/0 | 2/0/0 | 0/0/2 | 2.45x | 0 |
| feature type = skewed | 2 | 1.026 | 0/2 | 1/1/0 | 2/0/0 | 0/0/2 | 2.43x | 0 |
| feature type = sparse | 2 | 1.011 | 0/2 | 0/2/0 | 1/1/0 | 0/0/2 | 3.59x | 0 |
| missing = 20% random | 2 | 1.013 | 1/2 | 0/2/0 | 1/1/0 | 1/0/1 | 2.10x | 0 |
| missing = 30% informative | 2 | 1.011 | 1/2 | 0/2/0 | 0/2/0 | 1/0/1 | 2.66x | 0 |
| noise = low SNR | 2 | 1.030 | 0/2 | 0/2/0 | 0/1/1 | 0/0/2 | 2.75x | 0 |
| rows = 100k | 2 | 1.000 | 2/2 | 0/2/0 | 1/1/0 | 1/1/0 | 2.88x | 0 |
| rows = 2k | 2 | 1.108 | 0/2 | 0/1/1 | 0/0/2 | 0/0/2 | 2.45x | 0 |
| rows = 300 | 1 | 1.082 | 0/1 | 0/0/1 | 0/1/0 | 0/0/1 | 2.17x | 0 |
| signal = interaction | 1 | 1.073 | 0/1 | 1/0/0 | 1/0/0 | 0/0/1 | 3.07x | 0 |
| signal = linear | 1 | 1.045 | 0/1 | 0/1/0 | 1/0/0 | 0/0/1 | 2.70x | 0 |
| task = binary 5% positive | 1 | 1.115 | 0/1 | 0/0/1 | 0/0/1 | 0/0/1 | 2.32x | 0 |
| task = multiclass 20 | 1 | 1.070 | 0/1 | 1/0/0 | 0/0/1 | 0/0/1 | 2.44x | 0 |
| task = ranking | 1 | 1.035 | 1/1 | 0/1/0 | 1/0/0 | 0/1/0 | 2.16x | 0 |

## Hyperparameter regimes (regime panel datasets)

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| defaults: every library at its own defaults (Alloy: auto policy, auto rounds) | 3 | 1.013 | 1/3 | 1/2/0 | 3/0/0 | 1/0/2 | 3.48x | 0 |
| standard: depth 6, 300 trees, lr 0.05 | 3 | 1.049 | 0/3 | 1/2/0 | 1/1/1 | 0/0/3 | 2.11x | 0 |
| shallow: depth 2, 500 trees, lr 0.1 | 2 | 1.007 | 1/2 | 1/1/0 | 0/2/0 | 0/1/1 | 2.95x | 0 |
| deep: depth 12, 300 trees, lr 0.05 | 2 | 1.076 | 0/2 | 1/0/1 | 1/0/1 | 1/0/1 | 1.20x | 0 |
| few_trees: depth 6, 10 trees, lr 0.3 | 2 | 1.008 | 0/2 | 0/0/2 | 0/2/0 | 1/1/0 | 1.12x | 0 |
| many_trees: depth 6, 2000 trees, lr 0.02 | 2 | 1.084 | 0/2 | 2/0/0 | 1/0/1 | 0/0/2 | 2.57x | 0 |
| light_reg: depth 8, 300 trees, lr 0.1, no L1/L2, min hessian 0, 1 row per leaf | 2 | 1.032 | 1/2 | 1/1/0 | 0/2/0 | 1/0/1 | 1.12x | 0 |
| heavy_reg: depth 8, 300 trees, lr 0.1, L2 10, L1 1, min hessian 10, 100 rows per leaf, 70% row and column sampling | 2 | 1.037 | 1/2 | 0/1/1 | 0/2/0 | 0/1/1 | 2.10x | 0 |
| leafwise: leaf-wise growth, 63 leaves, depth cap 12, 300 trees, lr 0.05 | 2 | 1.022 | 0/2 | 0/2/0 | 1/1/0 | 0/0/2 | 3.86x | 0 |
| early_stopping: lr 0.05, depth 6, up to 5000 trees, stop after 50 rounds without improvement on a shared 15% validation split | 2 | 1.030 | 0/2 | 0/2/0 | 0/2/0 | 0/0/2 | 2.62x | 0 |

## Largest gaps (alloy vs best peer)

| Cell | Task | Loss ratio | Best peer | Outcome |
|---|---|---:|---|---|
| `binary_cat_high|standard` | binary | 2.812 | catboost | loss |
| `regression_cat_low|defaults` | regression | 2.267 | catboost | loss |
| `regression_cat_low|standard` | regression | 2.248 | xgboost | loss |
| `binary_cat_low|standard` | binary | 1.931 | catboost | loss |
| `binary_cat_high|defaults` | binary | 1.843 | catboost | loss |
| `binary_cat_low|defaults` | binary | 1.758 | catboost | loss |
| `binary_rows2k|standard` | binary | 1.166 | catboost | loss |
| `binary_base|deep` | binary | 1.146 | catboost | loss |
| `binary_base|many_trees` | binary | 1.121 | catboost | loss |
| `binary_imbalanced|standard` | binary | 1.115 | catboost | loss |
| `regression_signal_interaction|defaults` | regression | 1.090 | catboost | loss |
| `regression_wide_short|defaults` | regression | 1.084 | lightgbm | tie |
| `regression_rows300|standard` | regression | 1.082 | catboost | loss |
| `multiclass_base|standard` | multiclass | 1.075 | catboost | loss |
| `regression_signal_interaction|standard` | regression | 1.073 | catboost | loss |

## Largest leads (alloy vs best peer)

| Cell | Task | Loss ratio | Best peer | Outcome |
|---|---|---:|---|---|
| `multiclass_base|defaults` | multiclass | 0.983 | catboost | win |
| `regression_sparse|defaults` | regression | 0.988 | catboost | win |
| `regression_noise_low_SNR|defaults` | regression | 0.990 | lightgbm | win |
| `binary_sparse|defaults` | binary | 0.992 | catboost | win |
| `binary_noise_low_SNR|defaults` | binary | 0.992 | catboost | win |
| `regression_missing_mcar20|defaults` | regression | 0.993 | lightgbm | win |
| `binary_missing_mnar30|defaults` | binary | 0.995 | catboost | tie |
| `regression_rows100k|standard` | regression | 0.996 | lightgbm | tie |
| `regression_missing_mnar30|defaults` | regression | 0.997 | lightgbm | tie |
| `regression_missing_mcar20|standard` | regression | 0.997 | lightgbm | tie |
| `multiclass_20|defaults` | multiclass | 0.998 | catboost | tie |
| `binary_missing_mcar20|defaults` | binary | 0.999 | catboost | tie |

## Failed fits

| Cell | Arm | Error |
|---|---|---|
| `regression_cat_high|defaults` | alloy | ValueError: encoded cardinality 760 exceeds supported max 256 |
| `regression_cat_high|standard` | alloy | ValueError: encoded cardinality 785 exceeds supported max 256 |
