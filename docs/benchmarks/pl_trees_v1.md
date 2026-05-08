# PL Trees Benchmark Report

**Run ID**: `20260507T132049Z`  
**Mode**: quick  
**Date**: 2026-05-07

## Overview

This report compares `leaf_model='constant'` (standard scalar leaves) against `leaf_model='linear'` (piecewise-linear leaves) across regression and classification datasets. Both models use `training_policy='auto'` for the main comparison table and `training_policy='manual'` for convergence curves.

## Accuracy Comparison

| scenario | task | leaf_model | n_est | depth | lr | train_rows | n_feat | fit_s | rmse | accuracy | peak_mb |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | clf | constant | 200 | 6 | 0.05 | 455 | 30 | 0.215 | — | 0.9474 | 0.2 |
| breast_cancer | clf | linear | 200 | 6 | 0.05 | 455 | 30 | 0.973 | — | 0.9649 | 0.6 |
| california_housing | reg | constant | 200 | 6 | 0.05 | 16512 | 8 | 0.081 | 0.577053 | — | 1.2 |
| california_housing | reg | linear | 200 | 6 | 0.05 | 16512 | 8 | 0.683 | 0.556728 | — | 1.2 |
| synthetic_classification | clf | constant | 200 | 6 | 0.05 | 1600 | 6 | 0.057 | — | 0.8300 | 0.3 |
| synthetic_classification | clf | linear | 200 | 6 | 0.05 | 1600 | 6 | 0.043 | — | 0.8400 | 0.2 |
| synthetic_linear | reg | constant | 200 | 6 | 0.05 | 2400 | 8 | 0.049 | 0.748512 | — | 0.3 |
| synthetic_linear | reg | linear | 200 | 6 | 0.05 | 2400 | 8 | 0.062 | 0.805437 | — | 0.2 |

### RMSE Improvement Summary (Regression)

| scenario | constant RMSE | linear RMSE | RMSE improvement | train time overhead |
| --- | ---: | ---: | ---: | ---: |
| california_housing | 0.577053 | 0.556728 | +3.52% | +742.5% |
| synthetic_linear | 0.748512 | 0.805437 | -7.61% | +26.8% |

### Accuracy Improvement Summary (Classification)

| scenario | constant acc | linear acc | Δ accuracy |
| --- | ---: | ---: | ---: |
| breast_cancer | 0.9474 | 0.9649 | +0.0175 |
| synthetic_classification | 0.8300 | 0.8400 | +0.0100 |

## Convergence on Synthetic Linear Target

RMSE vs number of estimators on a linear-trend dataset (y = X @ w + noise). `training_policy='manual'`, no subsampling (row/col = 1.0) to show raw convergence without early stopping.

| n_estimators | constant RMSE | linear RMSE |
| ---: | ---: | ---: |
| 10 | 2.464750 | 0.254534 |
| 25 | 1.768262 | 0.213095 |
| 50 | 1.254476 | 0.212651 |
| 100 | 0.906023 | 0.219952 |
| 200 | 0.762706 | 0.229828 |

## `lambda_l2` Sweep for Linear Leaves

Synthetic linear dataset with 8 features. `training_policy='manual'`, no subsampling. Lower RMSE is better.

| lambda_l2 | linear RMSE | fit_seconds |
| ---: | ---: | ---: |
| 0.0 | 0.229828 | 1.178 |
| 0.01 | 0.197563 | 1.329 |
| 0.1 | 0.200357 | 1.302 |
| 1.0 | 0.218227 | 1.308 |
| 10.0 | 0.251713 | 1.340 |

**Recommended default**: `lambda_l2=0.01` (lowest RMSE = 0.197563). For `leaf_model='linear'`, non-zero `lambda_l2` provides regularization for the linear weights.

## Notes

- Peak RSS measured with Python `tracemalloc` (Python heap only).
- Main comparison uses `training_policy='auto'` with `row_subsample=0.8`,   `col_subsample=0.8`, `deterministic=True`. Auto-policy includes dataset-aware   early stopping, which may truncate training before `n_estimators` rounds.
- Convergence curves use `training_policy='manual'`, no subsampling, to show   raw gradient descent behavior.
- Linear leaves use the closed-form ridge solve   `α* = -(XᵀHX + λI)⁻¹ Xᵀg` per node.
- `lambda_l2` regularizes both the scalar leaf (standard NR) and the   linear leaf weights.

## v0.6.0 SIMD Speedup

v0.6.0 vectorises the per-row matrix-histogram accumulation via `wide::f32x8`.
Below is the head-to-head wall-time comparison on the same hardware (Apple
Silicon, M-series; expect ~5× on x86_64 AVX2 due to the wider 256-bit lanes).
Constant-leaf timings are unchanged.

| Scenario (regression, depth=6, manual policy) | v0.5.0 linear | v0.6.0 linear (SIMD) | Speedup |
| --- | ---: | ---: | ---: |
| n=20K, n_features=8, n_est=200 | 6.84 s | 2.31 s | **2.96×** |
| n=50K, n_features=8, n_est=200 | 16.02 s | 4.99 s | **3.21×** |
| n=100K, n_features=8, n_est=200 | 31.17 s | 9.49 s | **3.28×** |
| n=50K, n_features=8, n_est=500 | 40.07 s | 12.40 s | **3.23×** |

The slowdown of linear-leaf vs constant-leaf training drops from ~30-44× in
v0.5.0 to ~10-13× in v0.6.0 on Apple Silicon. Matrix histograms remain
fundamentally more expensive than scalar grad/hess histograms because each bin
stores 47 floats (`Xᵀg` + `XᵀHX`) instead of 3, but with SIMD the per-bin
update is 8× wider in registers.

