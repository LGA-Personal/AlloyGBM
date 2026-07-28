# Monotone Constraint Acceptance Benchmark

## Environment

- python: 3.13.5
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- numpy: 2.5.0

## Acceptance Contract

- Strict zero monotone violations at tolerance `1e-6` with finite grid predictions and differences.
- Regression constrained/unconstrained loss ratio at most `1.25`.
- Binary constrained error degradation at most `0.08`.
- The constrained model must beat the constant predictor.
- Scenario rows are training rows; holdouts use an independent deterministic stream with 512 to 4,096 rows.
- Fit timing is descriptive only and never gates acceptance.

## Summary

- Scenarios: 216
- Records: 216
- Gate failures: 0

## Records

| Scenario | Objective | Direction | Growth | Seed | Constrained fit s | Unconstrained fit s | Timing ratio | Constrained loss | Unconstrained loss | Constant loss | Grid pairs | Grid finite | Worst signed margin | Violations | Constrained rounds | Unconstrained rounds |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.001470 | 0.001199 | 1.226462 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.001167 | 0.001079 | 1.082007 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.001112 | 0.001209 | 0.919809 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.001145 | 0.001093 | 1.047590 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.001116 | 0.001084 | 1.029354 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.001107 | 0.001083 | 1.022124 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.001149 | 0.001127 | 1.020003 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.001150 | 0.001138 | 1.009993 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.001155 | 0.001181 | 0.977950 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.001154 | 0.001101 | 1.047949 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.001116 | 0.001138 | 0.980812 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.001157 | 0.001123 | 1.030417 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.001074 | 0.001020 | 1.053114 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.001195 | 0.001294 | 0.923560 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.001155 | 0.001038 | 1.112891 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.001005 | 0.000963 | 1.043604 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.001197 | 0.001225 | 0.977382 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.001125 | 0.001094 | 1.028645 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.000979 | 0.001232 | 0.794475 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.001072 | 0.001035 | 1.035915 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.000996 | 0.001112 | 0.895367 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.000986 | 0.001130 | 0.872659 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.001054 | 0.001119 | 0.941799 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.000977 | 0.001051 | 0.929060 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.011210 | 0.011256 | 0.995910 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.011757 | 0.011854 | 0.991849 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.011826 | 0.011854 | 0.997649 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.011455 | 0.011164 | 1.026093 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.011754 | 0.012188 | 0.964452 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.012328 | 0.012088 | 1.019899 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.012173 | 0.011350 | 1.072587 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.011436 | 0.011523 | 0.992500 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.011669 | 0.010965 | 1.064178 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.012190 | 0.011627 | 1.048436 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.011587 | 0.011695 | 0.990769 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.011929 | 0.011788 | 1.011976 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.011415 | 0.011422 | 0.999362 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.010896 | 0.010935 | 0.996395 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.010890 | 0.010918 | 0.997432 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.011035 | 0.010226 | 1.079094 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.010793 | 0.011108 | 0.971645 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.010883 | 0.011294 | 0.963603 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.011113 | 0.011130 | 0.998476 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.011868 | 0.012190 | 0.973608 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.011248 | 0.011291 | 0.996229 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.011151 | 0.010958 | 1.017671 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.011722 | 0.012418 | 0.943946 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.011742 | 0.011394 | 1.030606 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.026358 | 0.025820 | 1.020835 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.025910 | 0.025759 | 1.005854 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.026270 | 0.026458 | 0.992898 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.028932 | 0.029535 | 0.979594 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.028310 | 0.028349 | 0.998608 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.029465 | 0.029948 | 0.983871 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.026217 | 0.026018 | 1.007649 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.026419 | 0.026391 | 1.001064 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.025289 | 0.025813 | 0.979681 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.029056 | 0.028169 | 1.031491 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.030110 | 0.029261 | 1.029008 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.028240 | 0.029222 | 0.966397 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.025912 | 0.024929 | 1.039468 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.026238 | 0.026732 | 0.981525 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.025323 | 0.025453 | 0.994904 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.027097 | 0.028007 | 0.967502 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.030616 | 0.030526 | 1.002941 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.029457 | 0.028685 | 1.026908 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.025631 | 0.026285 | 0.975105 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.026015 | 0.025821 | 1.007509 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.025843 | 0.026092 | 0.990439 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.029612 | 0.029566 | 1.001563 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.029691 | 0.030212 | 0.982768 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.029078 | 0.030023 | 0.968506 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.016989 | 0.017626 | 0.963832 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.016974 | 0.017583 | 0.965381 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.017269 | 0.017543 | 0.984395 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.014781 | 0.015012 | 0.984631 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.014895 | 0.014910 | 0.999025 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.015055 | 0.014874 | 1.012129 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.017425 | 0.017234 | 1.011109 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.017252 | 0.017282 | 0.998274 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.017028 | 0.017261 | 0.986480 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.015026 | 0.014775 | 1.016977 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.014636 | 0.014904 | 0.982049 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.014962 | 0.014764 | 1.013417 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.018287 | 0.017849 | 1.024537 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.017795 | 0.017815 | 0.998873 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.017898 | 0.018133 | 0.987006 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.015579 | 0.015450 | 1.008309 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.015392 | 0.015453 | 0.996026 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.015114 | 0.015477 | 0.976575 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.018379 | 0.018025 | 1.019653 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.018833 | 0.018240 | 1.032486 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.017887 | 0.017630 | 1.014561 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.015909 | 0.015687 | 1.014184 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.016098 | 0.015522 | 1.037057 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.015081 | 0.015003 | 1.005191 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.031371 | 0.033734 | 0.929945 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.034240 | 0.033594 | 1.019227 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.034426 | 0.034144 | 1.008266 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.037817 | 0.037825 | 0.999768 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.037611 | 0.037874 | 0.993065 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.038665 | 0.038377 | 1.007523 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.034217 | 0.034158 | 1.001698 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.034106 | 0.033785 | 1.009497 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.034950 | 0.034402 | 1.015942 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.037632 | 0.037914 | 0.992563 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.038743 | 0.037945 | 1.021012 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.038458 | 0.038200 | 1.006754 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.034255 | 0.035004 | 0.978598 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.033692 | 0.034252 | 0.983648 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.034235 | 0.033771 | 1.013717 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.038352 | 0.038593 | 0.993756 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.038097 | 0.038623 | 0.986367 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.038570 | 0.038222 | 1.009116 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.034512 | 0.035506 | 0.972015 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.035758 | 0.035111 | 1.018430 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.033751 | 0.034404 | 0.981033 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.038702 | 0.038437 | 1.006892 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.038569 | 0.038295 | 1.007167 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.038502 | 0.039429 | 0.976491 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.072577 | 0.078218 | 0.927876 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.071948 | 0.071283 | 1.009324 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.071758 | 0.072012 | 0.996468 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.085476 | 0.086612 | 0.986885 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.084745 | 0.085524 | 0.990889 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.085706 | 0.085632 | 1.000855 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.071922 | 0.071838 | 1.001172 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.072760 | 0.072087 | 1.009341 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.072933 | 0.071248 | 1.023657 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.085457 | 0.085833 | 0.995618 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.085478 | 0.084694 | 1.009255 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.085602 | 0.085128 | 1.005565 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.071094 | 0.071963 | 0.987923 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.070432 | 0.070820 | 0.994515 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.070606 | 0.070711 | 0.998510 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.084926 | 0.085211 | 0.996645 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.084288 | 0.084270 | 1.000208 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.083560 | 0.084169 | 0.992761 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.074881 | 0.072178 | 1.037460 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.071889 | 0.070933 | 1.013479 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.071912 | 0.071722 | 1.002651 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.084565 | 0.084814 | 0.997058 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.084558 | 0.083758 | 1.009557 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.085808 | 0.085149 | 1.007741 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.107116 | 0.106340 | 1.007297 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.105849 | 0.106535 | 0.993566 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.105193 | 0.105714 | 0.995072 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.107690 | 0.109875 | 0.980119 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.107432 | 0.108116 | 0.993680 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.107009 | 0.108017 | 0.990673 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.104707 | 0.105718 | 0.990432 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.104803 | 0.104668 | 1.001293 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.107589 | 0.105661 | 1.018249 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.107630 | 0.107939 | 0.997141 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.106283 | 0.107207 | 0.991379 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.107164 | 0.107612 | 0.995838 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.154400 | 0.125808 | 1.227267 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.122917 | 0.125720 | 0.977707 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.122703 | 0.130321 | 0.941540 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.125769 | 0.124283 | 1.011957 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.123065 | 0.124980 | 0.984676 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.126734 | 0.126190 | 1.004316 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.122639 | 0.120893 | 1.014446 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.120070 | 0.122608 | 0.979300 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.121050 | 0.121288 | 0.998038 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.126340 | 0.123624 | 1.021969 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.122317 | 0.125598 | 0.973878 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.124116 | 0.123866 | 1.002020 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.160607 | 0.161230 | 0.996137 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.162012 | 0.163618 | 0.990183 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.162226 | 0.163053 | 0.994930 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.177034 | 0.177100 | 0.999629 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.177632 | 0.178306 | 0.996222 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.176728 | 0.177933 | 0.993228 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.163747 | 0.163571 | 1.001079 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.163503 | 0.163222 | 1.001725 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.163149 | 0.163475 | 0.998004 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.178753 | 0.178236 | 1.002899 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.178334 | 0.180383 | 0.988639 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.178585 | 0.178058 | 1.002959 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.178872 | 0.182256 | 0.981430 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.180995 | 0.180877 | 1.000652 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.179903 | 0.181093 | 0.993432 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.194375 | 0.194838 | 0.997621 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.194995 | 0.196008 | 0.994835 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.194700 | 0.195886 | 0.993946 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.179732 | 0.179407 | 1.001809 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.179410 | 0.179409 | 1.000003 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.178632 | 0.179400 | 0.995717 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.193074 | 0.194297 | 0.993703 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.194335 | 0.193725 | 1.003152 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.193269 | 0.194433 | 0.994018 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.307622 | 0.307708 | 0.999721 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.307503 | 0.304980 | 1.008271 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.307438 | 0.307323 | 1.000374 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.288392 | 0.289164 | 0.997332 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.287919 | 0.288749 | 0.997124 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.289615 | 0.288931 | 1.002368 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.310534 | 0.311791 | 0.995969 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.307557 | 0.309561 | 0.993528 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.309330 | 0.308698 | 1.002048 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.289260 | 0.289363 | 0.999643 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.289388 | 0.297477 | 0.972806 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.287371 | 0.291030 | 0.987429 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.326144 | 0.323267 | 1.008898 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.320772 | 0.323390 | 0.991905 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.322343 | 0.322286 | 1.000175 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.301796 | 0.303425 | 0.994632 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.302416 | 0.304267 | 0.993916 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.300808 | 0.302306 | 0.995045 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.327455 | 0.328567 | 0.996616 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.322488 | 0.322895 | 0.998739 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.320245 | 0.317573 | 1.008412 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.305491 | 0.305108 | 1.001253 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.303783 | 0.306247 | 0.991957 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.303625 | 0.302037 | 1.005257 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
