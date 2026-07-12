# DRO Robustness Benchmark

This deterministic synthetic benchmark fits on clean and contaminated training labels,
then measures RMSE against clean held-out targets. The corruption penalty is
`corrupted-train RMSE - clean-train RMSE`; smaller is more robust to the injected outliers.

- Seeds: 7, 13, 29, 47, 61
- Contaminated training labels: 12% per output
- `dro_radius`: 0.050
- All models: 100 trees, depth 4, learning rate 0.06, `lambda_l2=1.0`.

## Scalar Regressor

| Solver | Clean-label-fit RMSE | Contaminated-label-fit RMSE | Corruption penalty |
| --- | ---: | ---: | ---: |
| `standard` | 0.73269 | 0.97510 | +0.22875 |
| `dro` | 0.71300 | 1.01008 | +0.23433 |

## Joint Shared-Tree Multi-Label Regressor

| Solver | Clean-label-fit RMSE | Contaminated-label-fit RMSE | Corruption penalty |
| --- | ---: | ---: | ---: |
| `standard` | 0.78446 | 0.86088 | +0.05724 |
| `dro` | 0.78346 | 0.86901 | +0.06435 |

The joint path applies DRO only to final leaf values; its shared-histogram split
selection does not retain gradient-square statistics and therefore remains standard.
This benchmark is evidence for deciding whether the extra joint histogram memory required
for robust split selection is justified; it is not a claim that DRO wins every dataset.
