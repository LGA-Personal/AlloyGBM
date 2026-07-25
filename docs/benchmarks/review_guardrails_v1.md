# Review Evidence Guardrails

## Configuration

- Sections: quantile, goss, dart
- Seeds: 7, 13, 29
- Mode: full
- Quantile fixture: 512 training rows, 256 held-out rows.
- Boosting fixture: 512 training rows, 256 held-out rows.
- Model settings: depth 4, learning rate 0.06, lambda_l2=1.0, manual policy, deterministic quantile binning.
- GOSS rates: (0.10, 0.10), (0.20, 0.10), (0.20, 0.20), (0.30, 0.10).
- DART configs: (50, 0.05, 50, default_like), (100, 0.10, 50, default_like), (200, 0.20, 50, stress_profile), (100, 0.10, 5, default_like), (100, 0.10, 20, default_like).
- Timing is descriptive only; no wall-clock threshold is a quality gate.

## Quantile Split Selection

| Alpha | Arm | Median loss | No-split loss | Median gain |
|---:|---|---:|---:|---:|
| 0.10 | proxy | 0.036050 | 0.037225 | 0.855765 |
| 0.10 | smooth_0.05 | 0.036670 | 0.037225 | 0.557790 |
| 0.10 | smooth_0.10 | 0.036670 | 0.037225 | 0.803548 |
| 0.50 | proxy | 0.091041 | 0.092294 | 1.115308 |
| 0.50 | smooth_0.05 | 0.091766 | 0.092294 | 0.594674 |
| 0.50 | smooth_0.10 | 0.091766 | 0.092294 | 0.585669 |
| 0.90 | proxy | 0.040107 | 0.040797 | 1.222332 |
| 0.90 | smooth_0.05 | 0.040107 | 0.040797 | 1.072241 |
| 0.90 | smooth_0.10 | 0.040107 | 0.040797 | 1.138840 |

Smoothed-pinball median tie (absolute tolerance `1e-12`): `smooth_0.05=0.04010663203691039`, `smooth_0.10=0.04010663203691039`.
This identifies evidence for a later production decision; it does not recommend a production default.

## GOSS Rate Sweep

| Arm | Matched control | Median RMSE | Baseline RMSE | Fit seconds |
|---|---|---:|---:|---:|
| goss_0.10_0.10 | uniform_0.20 | 0.417081 | 1.173886 | 0.0111 |
| goss_0.20_0.10 | uniform_0.30 | 0.452279 | 1.173886 | 0.0114 |
| goss_0.20_0.20 | uniform_0.40 | 0.507839 | 1.173886 | 0.0114 |
| goss_0.30_0.10 | uniform_0.40 | 0.490197 | 1.173886 | 0.0114 |
| standard_full | - | 0.671367 | 1.173886 | 0.0119 |
| uniform_0.20 | - | 0.545133 | 1.173886 | 0.0093 |
| uniform_0.30 | - | 0.553216 | 1.173886 | 0.0099 |
| uniform_0.40 | - | 0.539075 | 1.173886 | 0.0103 |

## DART Dropout Profile

The configured dropout pressure is an expected-work proxy, not an observed drop count.
The 1.50x RMSE quality gate applies only to explicitly configured `default_like` rows.
`stress_profile` rows remain visible and must satisfy finite, control-matching, and completion contracts, but their quality is non-blocking.
Standard-time ratios use unrounded median fit times; displayed fit times are rounded.

| Arm | Profile | Matched standard | Median RMSE | Fit seconds | Seconds/round | Standard time ratio | Dropout pressure |
|---|---|---|---:|---:|---:|---:|---:|
| dart_100_0.10_20 | default_like | standard_100 | 0.972156 | 0.0337 | 0.000337 | 2.881 | 499.50 |
| dart_100_0.10_5 | default_like | standard_100 | 0.946943 | 0.0273 | 0.000273 | 2.339 | 377.00 |
| dart_100_0.10_50 | default_like | standard_100 | 0.972156 | 0.0354 | 0.000354 | 3.028 | 499.50 |
| dart_200_0.20_50 | stress_profile | standard_200 | 1.046657 | 0.1869 | 0.000935 | 8.276 | 3982.00 |
| dart_50_0.05_50 | default_like | standard_50 | 0.941069 | 0.0095 | 0.000190 | 1.528 | 70.75 |
| standard_100 | standard_control | - | 0.671367 | 0.0117 | 0.000117 | - | - |
| standard_200 | standard_control | - | 0.571675 | 0.0226 | 0.000113 | - | - |
| standard_50 | standard_control | - | 0.746986 | 0.0062 | 0.000125 | - | - |

## Gate Summary

| Gate | Result | Detail |
|---|---|---|
| quantile_contract | pass | exact requested seed/alpha/arm matrix, finite values, unique rows, and children |
| quantile_quality | pass | maximum loss/no-split ratio=0.994 (limit 1.100) |
| goss_contract | pass | exact requested seed/rate/control matrix, finite metrics, and unique rows |
| goss_completion | pass | all GOSS fits completed requested rounds |
| goss_quality | pass | maximum GOSS/uniform ratio=0.942 (limit 1.350) |
| goss_baseline | pass | every GOSS median beats its mean-predictor baseline |
| dart_contract | pass | exact requested seed/config/profile/control matrix, finite metrics, unique rows, and pressure |
| dart_completion | pass | all DART fits completed requested rounds |
| dart_quality | pass | maximum default-like DART/standard ratio=1.448 (limit 1.500; stress/profile arms excluded) |
