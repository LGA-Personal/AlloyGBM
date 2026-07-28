# Monotone Constraint Acceptance Benchmark

## Environment

- python: 3.13.5
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- numpy: 2.5.0
- source commit: 446bc49113d35bbe654465373df96438e6e398d0

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
| 128x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.001675 | 0.001286 | 1.302579 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.001283 | 0.001253 | 1.023543 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.001239 | 0.001188 | 1.043242 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.001301 | 0.001202 | 1.082097 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.001147 | 0.001158 | 0.990214 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.001159 | 0.001098 | 1.055663 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.001301 | 0.001278 | 1.018395 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.001238 | 0.001110 | 1.115149 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.001234 | 0.001151 | 1.072274 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.001176 | 0.001185 | 0.992578 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.001122 | 0.001127 | 0.995638 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.001190 | 0.001223 | 0.973326 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.001222 | 0.001177 | 1.038808 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.001421 | 0.001390 | 1.022029 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.001222 | 0.001125 | 1.086561 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.001140 | 0.001033 | 1.103118 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.001317 | 0.001289 | 1.021395 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.001257 | 0.001097 | 1.145641 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.001025 | 0.001219 | 0.840526 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.001188 | 0.001077 | 1.103227 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.001126 | 0.001146 | 0.982542 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.001053 | 0.001312 | 0.802826 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.001062 | 0.001098 | 0.967295 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.001003 | 0.001147 | 0.874023 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.010652 | 0.010209 | 1.043417 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.010902 | 0.010980 | 0.992820 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.010381 | 0.010910 | 0.951517 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.010499 | 0.010157 | 1.033742 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.010856 | 0.010922 | 0.994022 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.010909 | 0.011010 | 0.990804 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.010933 | 0.010860 | 1.006722 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.010487 | 0.010626 | 0.986907 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.010916 | 0.010335 | 1.056213 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.010748 | 0.010759 | 0.998935 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.010413 | 0.010484 | 0.993264 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.010698 | 0.010337 | 1.035017 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.010494 | 0.010751 | 0.976115 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.010290 | 0.010017 | 1.027204 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.009937 | 0.010144 | 0.979597 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.010493 | 0.010466 | 1.002608 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.010160 | 0.009993 | 1.016729 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.009467 | 0.010269 | 0.921882 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.010527 | 0.010587 | 0.994321 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.011394 | 0.012193 | 0.934469 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.010782 | 0.011154 | 0.966620 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.010306 | 0.010399 | 0.991061 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.011125 | 0.010914 | 1.019390 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.010394 | 0.010663 | 0.974727 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.026389 | 0.026099 | 1.011132 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.025783 | 0.025991 | 0.992015 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.026228 | 0.026017 | 1.008120 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.028271 | 0.029172 | 0.969112 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.028694 | 0.028743 | 0.998318 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.029668 | 0.028529 | 1.039932 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.025636 | 0.025844 | 0.991945 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.026311 | 0.025335 | 1.038554 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.025136 | 0.026235 | 0.958097 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.028333 | 0.027911 | 1.015094 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.028308 | 0.028786 | 0.983392 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.028666 | 0.027491 | 1.042740 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.025629 | 0.025054 | 1.022962 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.026136 | 0.026592 | 0.982844 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.025591 | 0.026143 | 0.978920 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.026618 | 0.027090 | 0.982581 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.030254 | 0.030619 | 0.988067 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.028220 | 0.028233 | 0.999519 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.026008 | 0.026993 | 0.963500 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.026596 | 0.026565 | 1.001161 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.026547 | 0.026804 | 0.990395 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.028848 | 0.029553 | 0.976130 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.028998 | 0.028869 | 1.004497 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.029133 | 0.029294 | 0.994520 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.017868 | 0.017591 | 1.015732 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.017783 | 0.018070 | 0.984078 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.017953 | 0.017934 | 1.001015 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.014932 | 0.015029 | 0.993593 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.014960 | 0.014997 | 0.997530 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.015157 | 0.015033 | 1.008251 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.017691 | 0.017529 | 1.009249 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.017665 | 0.018670 | 0.946205 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.019408 | 0.019078 | 1.017311 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.015245 | 0.015109 | 1.009062 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.014834 | 0.015207 | 0.975488 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.014990 | 0.015162 | 0.988629 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.018368 | 0.018328 | 1.002160 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.018735 | 0.018724 | 1.000585 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.018469 | 0.018286 | 1.009976 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.015670 | 0.015517 | 1.009871 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.015698 | 0.015794 | 0.993943 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.015471 | 0.015868 | 0.974991 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.019201 | 0.018713 | 1.026072 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.019302 | 0.019371 | 0.996419 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.018423 | 0.018591 | 0.990957 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.016162 | 0.015812 | 1.022167 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.016608 | 0.016214 | 1.024320 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.016323 | 0.016246 | 1.004722 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.033530 | 0.034194 | 0.980592 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.034704 | 0.034207 | 1.014525 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.035474 | 0.034771 | 1.020220 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.037480 | 0.037396 | 1.002233 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.037983 | 0.038093 | 0.997107 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.038554 | 0.037691 | 1.022915 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.034970 | 0.034501 | 1.013589 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.034950 | 0.034267 | 1.019928 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.034312 | 0.034383 | 0.997931 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.037313 | 0.037623 | 0.991778 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.038437 | 0.039319 | 0.977562 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.038846 | 0.039554 | 0.982113 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.035049 | 0.035489 | 0.987591 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.035792 | 0.035992 | 0.994442 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.035049 | 0.034141 | 1.026609 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.037701 | 0.037770 | 0.998172 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.038154 | 0.038773 | 0.984030 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.038601 | 0.038887 | 0.992635 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.036518 | 0.034867 | 1.047340 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.035102 | 0.034933 | 1.004816 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.034372 | 0.034171 | 1.005871 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.039021 | 0.039120 | 0.997478 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.037999 | 0.038755 | 0.980486 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.037834 | 0.039055 | 0.968726 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.076460 | 0.077409 | 0.987740 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.077383 | 0.078361 | 0.987522 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.076211 | 0.078179 | 0.974838 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.090626 | 0.091668 | 0.988637 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.089749 | 0.091308 | 0.982929 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.090535 | 0.091085 | 0.993955 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.077660 | 0.077001 | 1.008557 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.078157 | 0.077926 | 1.002966 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.077732 | 0.076666 | 1.013916 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.091661 | 0.090176 | 1.016460 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.090192 | 0.091071 | 0.990347 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.090055 | 0.090515 | 0.994918 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.076613 | 0.076576 | 1.000479 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.075245 | 0.075621 | 0.995032 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.076173 | 0.077599 | 0.981618 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.090252 | 0.089263 | 1.011078 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.089020 | 0.090054 | 0.988521 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.087917 | 0.090980 | 0.966330 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.077399 | 0.078109 | 0.990916 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.076425 | 0.076563 | 0.998198 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.077482 | 0.077301 | 1.002351 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.091216 | 0.090880 | 1.003699 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.089839 | 0.090706 | 0.990435 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.091524 | 0.092100 | 0.993741 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.107879 | 0.107670 | 1.001942 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.107257 | 0.108274 | 0.990605 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.106712 | 0.107555 | 0.992160 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.109327 | 0.110021 | 0.993699 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.109231 | 0.110732 | 0.986449 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.108975 | 0.110577 | 0.985513 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.108565 | 0.109336 | 0.992953 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.107902 | 0.108445 | 0.994991 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.108138 | 0.108548 | 0.996226 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.110428 | 0.110690 | 0.997636 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.108887 | 0.109574 | 0.993725 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.109655 | 0.110506 | 0.992296 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.125013 | 0.124273 | 1.005954 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.125133 | 0.125963 | 0.993407 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.124922 | 0.124848 | 1.000593 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.127116 | 0.125713 | 1.011161 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.125491 | 0.126958 | 0.988448 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.128024 | 0.130204 | 0.983254 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.126172 | 0.124005 | 1.017475 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.122588 | 0.125974 | 0.973120 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.123975 | 0.123682 | 1.002369 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.128841 | 0.125522 | 1.026441 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.124768 | 0.127504 | 0.978543 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.126283 | 0.126061 | 1.001761 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.166279 | 0.165929 | 1.002110 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.166480 | 0.167380 | 0.994622 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.164940 | 0.166535 | 0.990423 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.179047 | 0.179823 | 0.995684 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.179483 | 0.180195 | 0.996050 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.178437 | 0.180634 | 0.987838 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.166988 | 0.166260 | 1.004379 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.166855 | 0.166387 | 1.002814 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.166890 | 0.165738 | 1.006950 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.181143 | 0.180933 | 1.001162 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.181421 | 0.182529 | 0.993929 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.181667 | 0.180556 | 1.006158 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.182364 | 0.182032 | 1.001824 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.183633 | 0.183838 | 0.998887 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.182550 | 0.188266 | 0.969641 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.194578 | 0.195683 | 0.994354 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.197317 | 0.198812 | 0.992478 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.196749 | 0.198152 | 0.992922 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.182062 | 0.182302 | 0.998688 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.181765 | 0.181887 | 0.999331 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.183636 | 0.184906 | 0.993133 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.199666 | 0.197408 | 1.011441 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.197395 | 0.195562 | 1.009370 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.195966 | 0.195933 | 1.000166 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.316111 | 0.317873 | 0.994457 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.316782 | 0.313163 | 1.011556 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.315670 | 0.316503 | 0.997370 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.301643 | 0.302522 | 0.997091 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.298969 | 0.299368 | 0.998667 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.300306 | 0.299527 | 1.002600 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.316308 | 0.319724 | 0.989316 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.318437 | 0.317219 | 1.003839 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.317980 | 0.318431 | 0.998583 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.303350 | 0.305187 | 0.993980 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.302336 | 0.304805 | 0.991898 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.300611 | 0.303220 | 0.991393 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.327053 | 0.327596 | 0.998344 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.325367 | 0.330464 | 0.984577 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.330383 | 0.329463 | 1.002793 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.323242 | 0.315981 | 1.022978 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.313997 | 0.315649 | 0.994768 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.313996 | 0.322623 | 0.973260 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.373392 | 0.386524 | 0.966026 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.458776 | 0.347089 | 1.321782 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.327201 | 0.323833 | 1.010402 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.319872 | 0.315889 | 1.012608 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.314348 | 0.315834 | 0.995293 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.314118 | 0.314614 | 0.998426 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
