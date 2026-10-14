# Numerai at-scale benchmark, first run

`benchmarks/numerai_scale.py` at commit `067cd7a`, Apple M4 MacBook Air (24 GB, 10 threads), 2026-10-14.
Libraries: alloy 1.0.0 (this branch), lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10. One seed (85).

Data: Numerai v5.2, `medium` feature set (780 features), target `target`. Three chronological era chunks,
4 embargoed eras between them:

| Chunk | Eras | Rows |
|---|---|---:|
| train | 0888 - 1043 (156) | 1,037,050 |
| valid (early stopping) | 1048 - 1099 (52) | 324,864 |
| test | 1104 - 1207 (104) | 675,124 |

Every library: depth 12 / 1024 leaves (leaf-wise), 25% features and 80% rows per tree, 10,000 rows per leaf,
min split gain 0.001, L1 0.01, L2 0.05, 511 bins, early stopping on validation mean per-era correlation.
CatBoost has no L1 or min split gain.

## Deep profile (lr 0.005, up to 30,000 trees, patience 500)

| Library | Test per-era corr | Sharpe | Positive eras | Test RMSE | Valid per-era corr | Trees | Fit (s) | s / tree | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alloy | **0.0083** | **0.514** | 71% | 0.22376 | 0.0117 | 1977 | 1102 | 0.557 | 9.4 |
| catboost | 0.0074 | 0.467 | 71% | 0.22385 | 0.0126 | 3035 | 2890 | 0.952 | 8.3 |
| lightgbm | 0.0071 | 0.445 | 74% | 0.22372 | 0.0111 | 1187 | 270 | 0.228 | 10.3 |
| xgboost | 0.0069 | 0.442 | 71% | 0.22370 | 0.0121 | 1003 | 736 | 0.734 | 10.4 |

## Quick profile (lr 0.05, up to 3,000 trees, patience 100)

| Library | Test per-era corr | Sharpe | Positive eras | Trees | Fit (s) |
|---|---:|---:|---:|---:|---:|
| alloy | 0.0075 | 0.473 | 66% | 189 | 138 |
| xgboost | 0.0076 | 0.526 | 78% | 249 | 184 |
| catboost | 0.0049 | 0.419 | 71% | 100 | 202 |
| lightgbm | 0.0029 | 0.213 | 58% | 75 | 29 |

## Reading it

- Alloy has the best test correlation and Sharpe on the deep profile, but on validation the order flips
  (CatBoost 0.0126 and XGBoost 0.0121 above Alloy 0.0117). With one seed, gaps of ~0.001 corr are inside
  era-to-era noise; several seeds (or more test eras) are needed before calling a winner.
- Speed per tree: LightGBM 0.23 s, Alloy 0.56 s, XGBoost 0.73 s, CatBoost 0.95 s.
- Peak memory is mostly the three loaded chunks (about 8-9 GB); fits add at most about 1.5 GB.
- Early stopping on validation RMSE (`--stop-metric rmse`) stops every library after 8-25 trees on this data,
  which is why correlation is the default stop metric.
