# Monotone Constraint Acceptance Benchmark

## Environment

- python: 3.13.5
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
- numpy: 2.5.0
- source commit: 7c35814a9e825cced8bea2204d1740586c7ca4e4

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
| 128x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.001652 | 0.001249 | 1.323146 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.001190 | 0.001190 | 0.999755 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.001299 | 0.001158 | 1.121721 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.001222 | 0.001363 | 0.896795 | 0.258695 | 0.252555 | 1.556348 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.001273 | 0.001241 | 1.025718 | 0.224685 | 0.221238 | 1.578915 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.001214 | 0.001092 | 1.111484 | 0.286030 | 0.286937 | 1.632978 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.001251 | 0.001330 | 0.940347 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.001197 | 0.001187 | 1.008672 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.001206 | 0.001180 | 1.021748 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.001362 | 0.001185 | 1.149130 | 0.254901 | 0.245079 | 1.545103 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.001164 | 0.001141 | 1.020416 | 0.264072 | 0.258448 | 1.594431 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.001283 | 0.001288 | 0.995408 | 0.322099 | 0.321576 | 1.583984 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.001264 | 0.001105 | 1.144587 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.001268 | 0.001361 | 0.931799 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.001200 | 0.001237 | 0.969724 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.001209 | 0.001167 | 1.036357 | 0.281250 | 0.330078 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.001237 | 0.001261 | 0.980973 | 0.300781 | 0.287109 | 0.535156 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.001178 | 0.001158 | 1.017412 | 0.283203 | 0.281250 | 0.541016 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.001125 | 0.001247 | 0.902634 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.001334 | 0.001099 | 1.213886 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.001155 | 0.001167 | 0.989614 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.001015 | 0.001192 | 0.850978 | 0.283203 | 0.308594 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.001130 | 0.001063 | 1.062936 | 0.312500 | 0.314453 | 0.492188 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.001043 | 0.001180 | 0.884099 | 0.251953 | 0.263672 | 0.537109 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.010965 | 0.010338 | 1.060637 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.010868 | 0.011171 | 0.972873 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.010985 | 0.010723 | 1.024410 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.010352 | 0.010517 | 0.984304 | 0.331650 | 0.337493 | 1.566427 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.010686 | 0.010737 | 0.995196 | 0.316387 | 0.323504 | 1.486592 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.010698 | 0.010866 | 0.984497 | 0.328299 | 0.336358 | 1.610809 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.010872 | 0.010819 | 1.004856 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.010797 | 0.010830 | 0.996914 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.010793 | 0.010871 | 0.992836 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.010976 | 0.010305 | 1.065056 | 0.309648 | 0.313254 | 1.550359 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.010579 | 0.010873 | 0.972976 | 0.296087 | 0.292778 | 1.594609 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.010752 | 0.010258 | 1.048129 | 0.313040 | 0.314956 | 1.566994 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.010700 | 0.010555 | 1.013738 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.010038 | 0.010473 | 0.958469 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.010082 | 0.009918 | 1.016528 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.010596 | 0.010905 | 0.971698 | 0.257812 | 0.253906 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.010300 | 0.009787 | 1.052406 | 0.281250 | 0.269531 | 0.525391 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.010246 | 0.010347 | 0.990263 | 0.357422 | 0.376953 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.010990 | 0.010533 | 1.043306 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.011726 | 0.011435 | 1.025492 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.010459 | 0.010977 | 0.952740 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.010196 | 0.009831 | 1.037139 | 0.312500 | 0.332031 | 0.453125 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.011021 | 0.011176 | 0.986079 | 0.302734 | 0.300781 | 0.490234 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.010822 | 0.011782 | 0.918519 | 0.283203 | 0.279297 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.027005 | 0.025700 | 1.050767 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.026355 | 0.026792 | 0.983691 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.026527 | 0.027053 | 0.980549 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.028577 | 0.029063 | 0.983262 | 0.340289 | 0.345032 | 1.572866 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.028921 | 0.028413 | 1.017860 | 0.344665 | 0.346422 | 1.612689 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.029570 | 0.029149 | 1.014416 | 0.316298 | 0.321910 | 1.556179 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.026449 | 0.026103 | 1.013268 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.026383 | 0.026870 | 0.981874 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.025397 | 0.025831 | 0.983224 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.028793 | 0.028617 | 1.006149 | 0.336968 | 0.344893 | 1.559295 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.028849 | 0.028595 | 1.008897 | 0.321120 | 0.330052 | 1.579182 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.028396 | 0.028646 | 0.991267 | 0.317499 | 0.328161 | 1.593076 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.025392 | 0.025524 | 0.994820 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.026758 | 0.027031 | 0.989897 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.026270 | 0.026221 | 1.001866 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.026841 | 0.026626 | 1.008067 | 0.292969 | 0.287109 | 0.476562 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.030099 | 0.029837 | 1.008774 | 0.318359 | 0.328125 | 0.464844 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.027997 | 0.027989 | 1.000287 | 0.326172 | 0.320312 | 0.478516 | 4096 | True | -0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.025678 | 0.026037 | 0.986177 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.025829 | 0.026169 | 0.987017 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.025988 | 0.025935 | 1.002036 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.028629 | 0.029087 | 0.984254 | 0.261719 | 0.259766 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.028897 | 0.028540 | 1.012491 | 0.292969 | 0.279297 | 0.494141 | 4096 | True | 0 | 0 | 64 | 64 |
| 128x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.029146 | 0.028938 | 1.007188 | 0.345703 | 0.341797 | 0.500000 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.017910 | 0.018526 | 0.966796 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.017955 | 0.017650 | 1.017323 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.018047 | 0.017833 | 1.011991 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.014917 | 0.015030 | 0.992468 | 0.168769 | 0.163460 | 1.607932 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.014957 | 0.014970 | 0.999129 | 0.170228 | 0.166816 | 1.544791 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.014899 | 0.015066 | 0.988921 | 0.166833 | 0.164970 | 1.603959 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.017557 | 0.017558 | 0.999941 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.018341 | 0.018473 | 0.992850 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.017814 | 0.017228 | 1.034008 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.014976 | 0.014810 | 1.011231 | 0.165912 | 0.165126 | 1.590497 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.014739 | 0.015035 | 0.980274 | 0.166387 | 0.165047 | 1.609638 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.014915 | 0.014952 | 0.997559 | 0.167089 | 0.166065 | 1.562893 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.018035 | 0.018349 | 0.982935 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.018246 | 0.018627 | 0.979504 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.018810 | 0.018747 | 1.003363 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.015504 | 0.015446 | 1.003742 | 0.232422 | 0.229492 | 0.481445 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.015440 | 0.015408 | 1.002052 | 0.241211 | 0.242188 | 0.526367 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.015291 | 0.015497 | 0.986697 | 0.247070 | 0.241211 | 0.505859 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.018607 | 0.018819 | 0.988720 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.019442 | 0.018843 | 1.031802 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.018006 | 0.018424 | 0.977277 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.015914 | 0.015663 | 1.016068 | 0.257812 | 0.248047 | 0.508789 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.016101 | 0.015556 | 1.034997 | 0.256836 | 0.259766 | 0.526367 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.015138 | 0.015301 | 0.989312 | 0.261719 | 0.268555 | 0.476562 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.032510 | 0.033945 | 0.957729 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.034952 | 0.034269 | 1.019940 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.034180 | 0.034974 | 0.977282 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.038211 | 0.038152 | 1.001539 | 0.181473 | 0.180104 | 1.584954 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.037962 | 0.037654 | 1.008175 | 0.190308 | 0.189662 | 1.588016 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.037934 | 0.037857 | 1.002023 | 0.190246 | 0.187673 | 1.610400 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.034200 | 0.034281 | 0.997646 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.034812 | 0.034203 | 1.017813 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.034171 | 0.034352 | 0.994729 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.037762 | 0.037938 | 0.995365 | 0.180462 | 0.179246 | 1.563646 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.037781 | 0.038348 | 0.985223 | 0.195181 | 0.194568 | 1.605009 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.038450 | 0.038429 | 1.000546 | 0.189380 | 0.186563 | 1.564615 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.036564 | 0.035262 | 1.036925 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.034331 | 0.035446 | 0.968555 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.034541 | 0.033679 | 1.025581 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.037541 | 0.038538 | 0.974119 | 0.241211 | 0.239258 | 0.488281 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.038869 | 0.038576 | 1.007607 | 0.249023 | 0.254883 | 0.503906 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.038978 | 0.039139 | 0.995881 | 0.252930 | 0.249023 | 0.512695 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.035221 | 0.034493 | 1.021087 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.034453 | 0.035015 | 0.983971 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.034207 | 0.034513 | 0.991147 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.038473 | 0.037681 | 1.021025 | 0.224609 | 0.227539 | 0.480469 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.038131 | 0.038050 | 1.002122 | 0.273438 | 0.279297 | 0.499023 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.037952 | 0.038679 | 0.981210 | 0.249023 | 0.256836 | 0.498047 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.077477 | 0.076794 | 1.008891 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.077848 | 0.077934 | 0.998909 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.076693 | 0.078161 | 0.981218 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.092098 | 0.091703 | 1.004306 | 0.183281 | 0.182495 | 1.592553 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.090135 | 0.091868 | 0.981138 | 0.183022 | 0.181414 | 1.546147 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.091826 | 0.091826 | 1.000000 | 0.185178 | 0.181299 | 1.564028 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.078945 | 0.077903 | 1.013370 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.078427 | 0.078298 | 1.001648 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.077482 | 0.076009 | 1.019383 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.090820 | 0.091008 | 0.997933 | 0.186643 | 0.183947 | 1.590332 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.091257 | 0.090744 | 1.005662 | 0.182812 | 0.179769 | 1.551654 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.091519 | 0.091310 | 1.002294 | 0.184105 | 0.183117 | 1.565667 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.076653 | 0.076727 | 0.999036 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.075722 | 0.075739 | 0.999773 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.075800 | 0.076629 | 0.989189 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.090048 | 0.089810 | 1.002645 | 0.221680 | 0.223633 | 0.483398 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.089524 | 0.090558 | 0.988581 | 0.248047 | 0.249023 | 0.484375 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.089716 | 0.090745 | 0.988661 | 0.251953 | 0.250000 | 0.490234 | 4096 | True | -0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.077625 | 0.077893 | 0.996561 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.076432 | 0.077086 | 0.991520 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.077729 | 0.077960 | 0.997039 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.090220 | 0.091105 | 0.990285 | 0.242188 | 0.245117 | 0.489258 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.089321 | 0.090519 | 0.986772 | 0.264648 | 0.264648 | 0.493164 | 4096 | True | 0 | 0 | 64 | 64 |
| 4096x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.090553 | 0.091533 | 0.989297 | 0.213867 | 0.215820 | 0.481445 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.108244 | 0.107811 | 1.004015 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.107411 | 0.108833 | 0.986939 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.107577 | 0.107990 | 0.996177 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.109658 | 0.110099 | 0.995992 | 0.162541 | 0.160341 | 1.570258 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.108961 | 0.110262 | 0.988195 | 0.158868 | 0.157885 | 1.561938 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.108204 | 0.109597 | 0.987289 | 0.164033 | 0.159570 | 1.595960 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.108383 | 0.108388 | 0.999950 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.107184 | 0.108057 | 0.991921 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.107575 | 0.108420 | 0.992208 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.110498 | 0.110271 | 1.002057 | 0.163492 | 0.159953 | 1.592153 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.109183 | 0.110291 | 0.989949 | 0.160072 | 0.156880 | 1.607976 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.109466 | 0.110285 | 0.992573 | 0.161803 | 0.158357 | 1.586942 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.125630 | 0.123469 | 1.017502 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.124134 | 0.123837 | 1.002403 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.123488 | 0.123045 | 1.003605 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.126509 | 0.124744 | 1.014146 | 0.247070 | 0.244873 | 0.482666 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.128588 | 0.125701 | 1.022962 | 0.253174 | 0.254150 | 0.510254 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.126825 | 0.125523 | 1.010376 | 0.243896 | 0.243408 | 0.500488 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.124640 | 0.123173 | 1.011911 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.122225 | 0.124964 | 0.978079 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.123028 | 0.122258 | 1.006298 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.126797 | 0.124224 | 1.020712 | 0.243164 | 0.242920 | 0.489746 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.123011 | 0.126116 | 0.975382 | 0.242432 | 0.240234 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x2-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.124848 | 0.123541 | 1.010580 | 0.243652 | 0.243896 | 0.495117 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.165251 | 0.164972 | 1.001693 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.165604 | 0.165443 | 1.000971 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.164162 | 0.165480 | 0.992035 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.178740 | 0.179862 | 0.993763 | 0.180237 | 0.178099 | 1.579222 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.179125 | 0.179877 | 0.995819 | 0.178330 | 0.176340 | 1.592388 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.178382 | 0.180148 | 0.990196 | 0.180301 | 0.177843 | 1.562493 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.165714 | 0.165839 | 0.999241 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.164144 | 0.165729 | 0.990434 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.165392 | 0.164778 | 1.003724 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.179788 | 0.180038 | 0.998613 | 0.176434 | 0.175063 | 1.561410 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.179182 | 0.180331 | 0.993628 | 0.177041 | 0.174465 | 1.561859 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.180062 | 0.179466 | 1.003321 | 0.179466 | 0.176906 | 1.563255 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.180135 | 0.179770 | 1.002031 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.180908 | 0.181357 | 0.997525 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.180015 | 0.181556 | 0.991513 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.193704 | 0.193880 | 0.999095 | 0.258301 | 0.261230 | 0.489746 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.195542 | 0.195879 | 0.998282 | 0.236328 | 0.235840 | 0.470947 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.195363 | 0.196234 | 0.995559 | 0.253662 | 0.253906 | 0.482910 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.180130 | 0.181118 | 0.994545 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.180663 | 0.179991 | 1.003734 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.180016 | 0.180539 | 0.997099 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.193758 | 0.194341 | 0.996996 | 0.242920 | 0.240479 | 0.500732 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.194340 | 0.195013 | 0.996548 | 0.242676 | 0.241699 | 0.487061 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x16-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.193043 | 0.194510 | 0.992460 | 0.236328 | 0.238281 | 0.479492 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed0 | regression | -1 | level | 0 | 0.315386 | 0.318046 | 0.991638 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed1 | regression | -1 | level | 1 | 0.317657 | 0.314367 | 1.010466 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-level-seed2 | regression | -1 | level | 2 | 0.317958 | 0.317936 | 1.000068 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed0 | regression | -1 | leaf | 0 | 0.305802 | 0.303118 | 1.008855 | 0.178066 | 0.175332 | 1.571304 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed1 | regression | -1 | leaf | 1 | 0.298721 | 0.307025 | 0.972953 | 0.181510 | 0.178993 | 1.558636 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-decreasing-leaf-seed2 | regression | -1 | leaf | 2 | 0.346751 | 0.301902 | 1.148558 | 0.180125 | 0.178333 | 1.569355 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed0 | regression | 1 | level | 0 | 0.319241 | 0.318919 | 1.001010 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed1 | regression | 1 | level | 1 | 0.318825 | 0.317694 | 1.003561 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-level-seed2 | regression | 1 | level | 2 | 0.317166 | 0.316838 | 1.001037 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed0 | regression | 1 | leaf | 0 | 0.303362 | 0.302428 | 1.003087 | 0.178262 | 0.176881 | 1.575251 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed1 | regression | 1 | leaf | 1 | 0.300032 | 0.302144 | 0.993010 | 0.177764 | 0.174770 | 1.554789 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-regression-increasing-leaf-seed2 | regression | 1 | leaf | 2 | 0.299824 | 0.300709 | 0.997058 | 0.179948 | 0.176805 | 1.579471 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed0 | binary | -1 | level | 0 | 0.328233 | 0.330135 | 0.994237 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed1 | binary | -1 | level | 1 | 0.327685 | 0.330456 | 0.991616 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-level-seed2 | binary | -1 | level | 2 | 0.330517 | 0.332916 | 0.992796 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed0 | binary | -1 | leaf | 0 | 0.315943 | 0.314839 | 1.003506 | 0.248047 | 0.247070 | 0.482422 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed1 | binary | -1 | leaf | 1 | 0.312314 | 0.318340 | 0.981070 | 0.238037 | 0.236084 | 0.500000 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-decreasing-leaf-seed2 | binary | -1 | leaf | 2 | 0.312922 | 0.315433 | 0.992042 | 0.243896 | 0.244385 | 0.495361 | 4096 | True | -0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed0 | binary | 1 | level | 0 | 0.348798 | 0.327624 | 1.064627 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed1 | binary | 1 | level | 1 | 0.332120 | 0.332199 | 0.999764 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-level-seed2 | binary | 1 | level | 2 | 0.361923 | 0.422666 | 0.856285 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed0 | binary | 1 | leaf | 0 | 0.356025 | 0.336823 | 1.057008 | 0.248291 | 0.247803 | 0.485596 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed1 | binary | 1 | leaf | 1 | 0.318219 | 0.317689 | 1.001666 | 0.241211 | 0.241455 | 0.497070 | 4096 | True | 0 | 0 | 64 | 64 |
| 32768x128-binary-increasing-leaf-seed2 | binary | 1 | leaf | 2 | 0.313987 | 0.316088 | 0.993353 | 0.244141 | 0.241943 | 0.506836 | 4096 | True | 0 | 0 | 64 | 64 |
