# Monotone Constraint Acceptance Benchmark

## Environment

- python: 3.13.5
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- numpy: 2.5.0
- source commit: 72e4709b9706127a7191e51bfe40228515d3b05b

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
| 128x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.001623 | 0.001193 | 1.360469 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.001190 | 0.001145 | 1.039493 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.001185 | 0.001109 | 1.068217 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.001316 | 0.001208 | 1.089881 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.001199 | 0.001150 | 1.042511 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.001153 | 0.001185 | 0.972652 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.001281 | 0.001184 | 1.082239 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.001229 | 0.001181 | 1.040284 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.001226 | 0.001164 | 1.053436 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.001203 | 0.001182 | 1.017090 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.001121 | 0.001131 | 0.991452 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.001307 | 0.001305 | 1.001341 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.001160 | 0.001066 | 1.087241 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.001263 | 0.001349 | 0.936299 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.001200 | 0.001261 | 0.951748 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.001110 | 0.001075 | 1.032531 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.001236 | 0.001324 | 0.933957 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.001276 | 0.001040 | 1.226517 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.001010 | 0.001400 | 0.721491 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.001185 | 0.001100 | 1.077221 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.001041 | 0.001182 | 0.880514 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.001031 | 0.001181 | 0.873200 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.001075 | 0.001164 | 0.923871 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.001026 | 0.001212 | 0.846767 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.010774 | 0.010359 | 1.040088 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.010583 | 0.011122 | 0.951536 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.010996 | 0.010595 | 1.037912 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.010251 | 0.010630 | 0.964355 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.010574 | 0.010772 | 0.981623 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.010851 | 0.010645 | 1.019384 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.010906 | 0.010534 | 1.035267 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.010730 | 0.010663 | 1.006350 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.010820 | 0.010691 | 1.012081 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.011148 | 0.010317 | 1.080545 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.010621 | 0.011328 | 0.937632 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.010865 | 0.010664 | 1.018912 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.010938 | 0.010401 | 1.051655 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.009892 | 0.010507 | 0.941495 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.009942 | 0.010030 | 0.991234 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.010687 | 0.010555 | 1.012522 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.009966 | 0.010277 | 0.969723 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.010044 | 0.010064 | 0.998042 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.010550 | 0.010469 | 1.007773 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.011749 | 0.011979 | 0.980817 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.011432 | 0.011076 | 1.032143 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.010363 | 0.009952 | 1.041306 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.010707 | 0.011398 | 0.939426 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.010469 | 0.010507 | 0.996380 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.026450 | 0.026673 | 0.991664 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.025716 | 0.025620 | 1.003768 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.026280 | 0.025943 | 1.012995 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.028308 | 0.028311 | 0.999894 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.028169 | 0.028161 | 1.000260 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.029273 | 0.028274 | 1.035310 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.025653 | 0.025938 | 0.989006 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.025776 | 0.025916 | 0.994601 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.025906 | 0.026485 | 0.978131 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.028771 | 0.028485 | 1.010070 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.029567 | 0.028867 | 1.024269 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.028077 | 0.029072 | 0.965777 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.025627 | 0.025594 | 1.001307 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.027164 | 0.027826 | 0.976200 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.026562 | 0.026270 | 1.011096 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.027410 | 0.027001 | 1.015121 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.029988 | 0.030296 | 0.989834 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.028016 | 0.028706 | 0.975975 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.026076 | 0.026477 | 0.984863 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.026313 | 0.026358 | 0.998290 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.026000 | 0.025982 | 1.000712 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.029303 | 0.028902 | 1.013888 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.029432 | 0.028816 | 1.021383 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.029313 | 0.028674 | 1.022282 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.017749 | 0.017849 | 0.994386 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.017622 | 0.017699 | 0.995638 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.017999 | 0.017499 | 1.028554 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.014949 | 0.015197 | 0.983733 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.015111 | 0.015159 | 0.996820 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.015270 | 0.015013 | 1.017141 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.017948 | 0.017447 | 1.028731 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.017831 | 0.018283 | 0.975280 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.018399 | 0.017802 | 1.033569 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.015087 | 0.015028 | 1.003934 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.015040 | 0.015190 | 0.990141 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.015501 | 0.015676 | 0.988855 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.019467 | 0.019070 | 1.020831 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.018915 | 0.018489 | 1.023050 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.018562 | 0.019088 | 0.972465 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.016031 | 0.016141 | 0.993213 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.016501 | 0.016499 | 1.000116 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.015660 | 0.015900 | 0.984937 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.018933 | 0.019038 | 0.994489 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.019680 | 0.018981 | 1.036822 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.018632 | 0.018150 | 1.026587 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.016282 | 0.015896 | 1.024275 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.016392 | 0.015820 | 1.036197 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.015534 | 0.015429 | 1.006811 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.032683 | 0.035030 | 0.932994 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.034534 | 0.034515 | 1.000548 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.035876 | 0.034729 | 1.033026 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.037857 | 0.038820 | 0.975191 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.038137 | 0.038334 | 0.994858 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.039249 | 0.040585 | 0.967081 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.035913 | 0.036224 | 0.991416 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.035529 | 0.036338 | 0.977735 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.034849 | 0.035909 | 0.970486 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.040426 | 0.039157 | 1.032418 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.039435 | 0.039193 | 1.006177 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.039915 | 0.039180 | 1.018757 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.035841 | 0.035808 | 1.000936 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.035084 | 0.034579 | 1.014620 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.035060 | 0.034428 | 1.018361 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.038305 | 0.038339 | 0.999111 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.039558 | 0.039889 | 0.991706 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.039519 | 0.039028 | 1.012585 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.035834 | 0.035421 | 1.011655 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.035100 | 0.034853 | 1.007092 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.034598 | 0.034625 | 0.999227 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.040060 | 0.038296 | 1.046060 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.038083 | 0.038911 | 0.978717 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.039523 | 0.039526 | 0.999934 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.078542 | 0.078137 | 1.005178 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.077342 | 0.077128 | 1.002782 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.076875 | 0.077685 | 0.989573 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.091129 | 0.090453 | 1.007479 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.090916 | 0.092486 | 0.983025 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.090961 | 0.091291 | 0.996394 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.076957 | 0.077637 | 0.991241 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.077109 | 0.076730 | 1.004950 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.077185 | 0.076646 | 1.007029 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.091472 | 0.091591 | 0.998700 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.091190 | 0.090720 | 1.005184 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.090805 | 0.091503 | 0.992378 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.077751 | 0.076946 | 1.010467 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.076357 | 0.076571 | 0.997200 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.076146 | 0.077305 | 0.985016 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.090862 | 0.091080 | 0.997612 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.089763 | 0.091292 | 0.983243 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.089066 | 0.091008 | 0.978664 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.077150 | 0.077147 | 1.000033 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.076491 | 0.077608 | 0.985603 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.077614 | 0.078160 | 0.993020 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.089590 | 0.090061 | 0.994776 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.089575 | 0.090091 | 0.994269 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.090993 | 0.090282 | 1.007876 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.109319 | 0.108075 | 1.011512 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.107890 | 0.108719 | 0.992372 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.107597 | 0.108518 | 0.991511 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.110137 | 0.111556 | 0.987278 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.109875 | 0.110734 | 0.992248 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.108939 | 0.110264 | 0.987980 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.108597 | 0.109063 | 0.995728 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.107613 | 0.108232 | 0.994283 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.107842 | 0.108753 | 0.991622 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.110287 | 0.112544 | 0.979945 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.110983 | 0.112374 | 0.987625 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.111832 | 0.111916 | 0.999249 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.127988 | 0.126509 | 1.011691 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.126990 | 0.126274 | 1.005670 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.126000 | 0.125012 | 1.007905 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.128114 | 0.126336 | 1.014074 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.127423 | 0.131764 | 0.967053 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.127763 | 0.127595 | 1.001316 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.126756 | 0.124965 | 1.014333 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.122849 | 0.125942 | 0.975440 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.124314 | 0.124539 | 0.998194 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.128207 | 0.125529 | 1.021336 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.124691 | 0.128913 | 0.967254 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.126461 | 0.125728 | 1.005824 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.164509 | 0.165143 | 0.996161 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.165505 | 0.165551 | 0.999722 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.164904 | 0.166034 | 0.993197 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.178711 | 0.178969 | 0.998555 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.179849 | 0.181112 | 0.993023 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.178985 | 0.180494 | 0.991636 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.167613 | 0.168295 | 0.995945 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.166984 | 0.166793 | 1.001142 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.165611 | 0.165488 | 1.000745 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.180411 | 0.180099 | 1.001731 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.180222 | 0.180464 | 0.998662 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.181016 | 0.180167 | 1.004709 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.181410 | 0.180956 | 1.002508 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.182553 | 0.182132 | 1.002312 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.181558 | 0.182750 | 0.993480 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.195398 | 0.195876 | 0.997557 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.196694 | 0.197292 | 0.996970 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.196506 | 0.197413 | 0.995406 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.181463 | 0.181159 | 1.001680 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.181893 | 0.180927 | 1.005341 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.180462 | 0.181065 | 0.996670 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.194753 | 0.195619 | 0.995576 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.194949 | 0.194345 | 1.003109 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.194851 | 0.195087 | 0.998792 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.315088 | 0.315992 | 0.997140 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.317419 | 0.313192 | 1.013498 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.315651 | 0.315606 | 1.000144 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.300115 | 0.300039 | 1.000254 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.296171 | 0.299230 | 0.989777 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.300865 | 0.299004 | 1.006223 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.317544 | 0.315182 | 1.007495 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.318451 | 0.315611 | 1.008998 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.313861 | 0.317509 | 0.988512 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.299176 | 0.298644 | 1.001784 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.296589 | 0.301100 | 0.985019 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.301515 | 0.301244 | 1.000902 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.328430 | 0.332709 | 0.987138 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.324895 | 0.328894 | 0.987840 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.327551 | 0.383776 | 0.853496 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.312386 | 0.314287 | 0.993953 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.312692 | 0.315833 | 0.990055 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.312404 | 0.313543 | 0.996366 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.334484 | 0.325441 | 1.027786 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.328745 | 0.329624 | 0.997333 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.326296 | 0.324833 | 1.004505 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.316061 | 0.316163 | 0.999676 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.312897 | 0.316348 | 0.989091 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.312914 | 0.313280 | 0.998831 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
