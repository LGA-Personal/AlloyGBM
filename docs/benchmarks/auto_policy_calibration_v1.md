# Auto-Policy Calibration Benchmark

- Command: `/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/auto_policy_benchmark.py --gate --output-json /tmp/auto-policy-calibration-final.json --output-report docs/benchmarks/auto_policy_calibration_v1.md`
- Selected outcome: `current_auto`
- Gate passed: `true`
- Timing is descriptive only and is not used as a quality gate.

## Environment

- Git commit: `3578e9d1cd5cba9f714f2fec9406330cbebda840`
- OS/platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Architecture: `arm64`
- Python: `3.13.5`
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- NumPy: `2.5.0`
- AlloyGBM: `0.12.10`

## Matrix Completeness

- Matrix evidence complete: `true`
- Expected records: `600`
- Complete records: `600`
- Total records: `600`
- Distinct fixtures: `50`
- Declared exact shapes: `10`
- Distinct objectives: `5`
- Distinct seeds: `3`
- Distinct arms: `4`

## Gate Detail

```text
keep current: no candidate meets every quality gate
manual_default rejected: small-wide/multiclass/seed=7 exceeds the 3% primary-loss limit (1.6363x)
small-wide/multiclass/seed=13 exceeds the 3% primary-loss limit (1.6358x)
small-wide/multiclass/seed=29 exceeds the 3% primary-loss limit (1.5891x)
small-wide/sparse_regression/seed=7 exceeds the 3% primary-loss limit (1.0481x)
medium-narrow shape median is worse than current auto (1.0101x)
small-wide shape median is worse than current auto (1.0006x)
overall median primary loss does not reach the required 1% improvement (0.9998x)
no_gain_floor rejected: overall median primary loss does not reach the required 1% improvement (1.0000x)
quality_first rejected: large-narrow shape median is worse than current auto (1.0004x)
medium-narrow shape median is worse than current auto (1.0033x)
overall median primary loss does not reach the required 1% improvement (1.0000x)
```

## Candidate Gate Results

| Arm | Result | Overall loss ratio |
|---|---|---:|
| manual_default | fail | 0.999800 |
| no_gain_floor | fail | 1.000000 |
| quality_first | fail | 1.000000 |

### manual_default

```text
small-wide/multiclass/seed=7 exceeds the 3% primary-loss limit (1.6363x)
small-wide/multiclass/seed=13 exceeds the 3% primary-loss limit (1.6358x)
small-wide/multiclass/seed=29 exceeds the 3% primary-loss limit (1.5891x)
small-wide/sparse_regression/seed=7 exceeds the 3% primary-loss limit (1.0481x)
medium-narrow shape median is worse than current auto (1.0101x)
small-wide shape median is worse than current auto (1.0006x)
overall median primary loss does not reach the required 1% improvement (0.9998x)
```

### no_gain_floor

```text
overall median primary loss does not reach the required 1% improvement (1.0000x)
```

### no_gain_floor record-level differences

- Differing record-level primary metrics: `13 of 150`
- Normalized-loss ratio range: `0.99940224` to `1.00038293`
- Overall and protected-stratum median normalized-loss ratios: `1.000000`

### quality_first

```text
large-narrow shape median is worse than current auto (1.0004x)
medium-narrow shape median is worse than current auto (1.0033x)
overall median primary loss does not reach the required 1% improvement (1.0000x)
```

## Exact-Shape/Objective Loss Ratios

| Arm | Rows | Features | Objective | Median normalized loss |
|---|---:|---:|---|---:|
| manual_default | 512 | 8 | binary | 1.000000 |
| manual_default | 512 | 8 | multiclass | 1.000000 |
| manual_default | 512 | 8 | ranking | 1.000000 |
| manual_default | 512 | 8 | regression | 1.000000 |
| manual_default | 512 | 8 | sparse_regression | 1.000000 |
| manual_default | 512 | 128 | binary | 1.000000 |
| manual_default | 512 | 128 | multiclass | 1.508581 |
| manual_default | 512 | 128 | ranking | 0.873694 |
| manual_default | 512 | 128 | regression | 0.999842 |
| manual_default | 512 | 128 | sparse_regression | 1.015368 |
| manual_default | 1023 | 16 | binary | 1.000000 |
| manual_default | 1023 | 16 | multiclass | 1.000000 |
| manual_default | 1023 | 16 | ranking | 1.000000 |
| manual_default | 1023 | 16 | regression | 1.000000 |
| manual_default | 1023 | 16 | sparse_regression | 1.000000 |
| manual_default | 1023 | 256 | binary | 1.000000 |
| manual_default | 1023 | 256 | multiclass | 1.763043 |
| manual_default | 1023 | 256 | ranking | 0.822361 |
| manual_default | 1023 | 256 | regression | 1.001346 |
| manual_default | 1023 | 256 | sparse_regression | 1.026128 |
| manual_default | 2048 | 16 | binary | 1.001104 |
| manual_default | 2048 | 16 | multiclass | 1.030913 |
| manual_default | 2048 | 16 | ranking | 1.132879 |
| manual_default | 2048 | 16 | regression | 1.024207 |
| manual_default | 2048 | 16 | sparse_regression | 0.993572 |
| manual_default | 2048 | 128 | binary | 0.977187 |
| manual_default | 2048 | 128 | multiclass | 1.011608 |
| manual_default | 2048 | 128 | ranking | 0.820401 |
| manual_default | 2048 | 128 | regression | 0.864736 |
| manual_default | 2048 | 128 | sparse_regression | 1.040410 |
| manual_default | 8192 | 16 | binary | 1.002311 |
| manual_default | 8192 | 16 | multiclass | 1.000556 |
| manual_default | 8192 | 16 | ranking | 0.986659 |
| manual_default | 8192 | 16 | regression | 1.005969 |
| manual_default | 8192 | 16 | sparse_regression | 1.003735 |
| manual_default | 8192 | 256 | binary | 0.952046 |
| manual_default | 8192 | 256 | multiclass | 0.984553 |
| manual_default | 8192 | 256 | ranking | 0.793127 |
| manual_default | 8192 | 256 | regression | 0.703464 |
| manual_default | 8192 | 256 | sparse_regression | 0.940113 |
| manual_default | 16384 | 16 | binary | 0.995075 |
| manual_default | 16384 | 16 | multiclass | 0.999599 |
| manual_default | 16384 | 16 | ranking | 0.981461 |
| manual_default | 16384 | 16 | regression | 1.012979 |
| manual_default | 16384 | 16 | sparse_regression | 0.971733 |
| manual_default | 16384 | 256 | binary | 0.950614 |
| manual_default | 16384 | 256 | multiclass | 0.976919 |
| manual_default | 16384 | 256 | ranking | 0.759015 |
| manual_default | 16384 | 256 | regression | 0.692186 |
| manual_default | 16384 | 256 | sparse_regression | 0.876017 |
| no_gain_floor | 512 | 8 | binary | 1.000000 |
| no_gain_floor | 512 | 8 | multiclass | 1.000000 |
| no_gain_floor | 512 | 8 | ranking | 1.000000 |
| no_gain_floor | 512 | 8 | regression | 1.000000 |
| no_gain_floor | 512 | 8 | sparse_regression | 1.000000 |
| no_gain_floor | 512 | 128 | binary | 1.000000 |
| no_gain_floor | 512 | 128 | multiclass | 1.000000 |
| no_gain_floor | 512 | 128 | ranking | 1.000000 |
| no_gain_floor | 512 | 128 | regression | 1.000000 |
| no_gain_floor | 512 | 128 | sparse_regression | 1.000000 |
| no_gain_floor | 1023 | 16 | binary | 1.000000 |
| no_gain_floor | 1023 | 16 | multiclass | 1.000000 |
| no_gain_floor | 1023 | 16 | ranking | 1.000000 |
| no_gain_floor | 1023 | 16 | regression | 1.000000 |
| no_gain_floor | 1023 | 16 | sparse_regression | 1.000000 |
| no_gain_floor | 1023 | 256 | binary | 1.000000 |
| no_gain_floor | 1023 | 256 | multiclass | 1.000000 |
| no_gain_floor | 1023 | 256 | ranking | 1.000000 |
| no_gain_floor | 1023 | 256 | regression | 1.000000 |
| no_gain_floor | 1023 | 256 | sparse_regression | 1.000000 |
| no_gain_floor | 2048 | 16 | binary | 1.000000 |
| no_gain_floor | 2048 | 16 | multiclass | 1.000000 |
| no_gain_floor | 2048 | 16 | ranking | 1.000000 |
| no_gain_floor | 2048 | 16 | regression | 1.000000 |
| no_gain_floor | 2048 | 16 | sparse_regression | 1.000000 |
| no_gain_floor | 2048 | 128 | binary | 1.000000 |
| no_gain_floor | 2048 | 128 | multiclass | 1.000000 |
| no_gain_floor | 2048 | 128 | ranking | 1.000000 |
| no_gain_floor | 2048 | 128 | regression | 1.000000 |
| no_gain_floor | 2048 | 128 | sparse_regression | 1.000000 |
| no_gain_floor | 8192 | 16 | binary | 1.000000 |
| no_gain_floor | 8192 | 16 | multiclass | 1.000001 |
| no_gain_floor | 8192 | 16 | ranking | 1.000000 |
| no_gain_floor | 8192 | 16 | regression | 1.000000 |
| no_gain_floor | 8192 | 16 | sparse_regression | 1.000000 |
| no_gain_floor | 8192 | 256 | binary | 1.000000 |
| no_gain_floor | 8192 | 256 | multiclass | 1.000000 |
| no_gain_floor | 8192 | 256 | ranking | 1.000000 |
| no_gain_floor | 8192 | 256 | regression | 1.000000 |
| no_gain_floor | 8192 | 256 | sparse_regression | 1.000000 |
| no_gain_floor | 16384 | 16 | binary | 1.000000 |
| no_gain_floor | 16384 | 16 | multiclass | 1.000000 |
| no_gain_floor | 16384 | 16 | ranking | 1.000000 |
| no_gain_floor | 16384 | 16 | regression | 1.000000 |
| no_gain_floor | 16384 | 16 | sparse_regression | 1.000000 |
| no_gain_floor | 16384 | 256 | binary | 1.000000 |
| no_gain_floor | 16384 | 256 | multiclass | 1.000000 |
| no_gain_floor | 16384 | 256 | ranking | 1.000000 |
| no_gain_floor | 16384 | 256 | regression | 1.000000 |
| no_gain_floor | 16384 | 256 | sparse_regression | 1.000000 |
| quality_first | 512 | 8 | binary | 1.000000 |
| quality_first | 512 | 8 | multiclass | 1.000000 |
| quality_first | 512 | 8 | ranking | 1.000000 |
| quality_first | 512 | 8 | regression | 1.000000 |
| quality_first | 512 | 8 | sparse_regression | 1.000000 |
| quality_first | 512 | 128 | binary | 1.000000 |
| quality_first | 512 | 128 | multiclass | 1.000000 |
| quality_first | 512 | 128 | ranking | 1.000000 |
| quality_first | 512 | 128 | regression | 1.000000 |
| quality_first | 512 | 128 | sparse_regression | 1.000000 |
| quality_first | 1023 | 16 | binary | 1.000000 |
| quality_first | 1023 | 16 | multiclass | 1.000000 |
| quality_first | 1023 | 16 | ranking | 1.000000 |
| quality_first | 1023 | 16 | regression | 1.000000 |
| quality_first | 1023 | 16 | sparse_regression | 1.000000 |
| quality_first | 1023 | 256 | binary | 1.000000 |
| quality_first | 1023 | 256 | multiclass | 1.000000 |
| quality_first | 1023 | 256 | ranking | 1.000000 |
| quality_first | 1023 | 256 | regression | 1.000000 |
| quality_first | 1023 | 256 | sparse_regression | 1.000000 |
| quality_first | 2048 | 16 | binary | 1.002019 |
| quality_first | 2048 | 16 | multiclass | 1.003368 |
| quality_first | 2048 | 16 | ranking | 1.034745 |
| quality_first | 2048 | 16 | regression | 1.019049 |
| quality_first | 2048 | 16 | sparse_regression | 0.989685 |
| quality_first | 2048 | 128 | binary | 0.970324 |
| quality_first | 2048 | 128 | multiclass | 0.996395 |
| quality_first | 2048 | 128 | ranking | 0.828771 |
| quality_first | 2048 | 128 | regression | 0.869294 |
| quality_first | 2048 | 128 | sparse_regression | 1.015584 |
| quality_first | 8192 | 16 | binary | 1.004508 |
| quality_first | 8192 | 16 | multiclass | 1.002303 |
| quality_first | 8192 | 16 | ranking | 1.014869 |
| quality_first | 8192 | 16 | regression | 1.003200 |
| quality_first | 8192 | 16 | sparse_regression | 0.996476 |
| quality_first | 8192 | 256 | binary | 0.954115 |
| quality_first | 8192 | 256 | multiclass | 0.984869 |
| quality_first | 8192 | 256 | ranking | 0.831949 |
| quality_first | 8192 | 256 | regression | 0.697408 |
| quality_first | 8192 | 256 | sparse_regression | 0.908515 |
| quality_first | 16384 | 16 | binary | 1.000919 |
| quality_first | 16384 | 16 | multiclass | 0.995014 |
| quality_first | 16384 | 16 | ranking | 1.032240 |
| quality_first | 16384 | 16 | regression | 1.008771 |
| quality_first | 16384 | 16 | sparse_regression | 0.994838 |
| quality_first | 16384 | 256 | binary | 0.954241 |
| quality_first | 16384 | 256 | multiclass | 0.979088 |
| quality_first | 16384 | 256 | ranking | 0.779419 |
| quality_first | 16384 | 256 | regression | 0.690238 |
| quality_first | 16384 | 256 | sparse_regression | 0.906911 |

## Resolved Policy Observations

- Current-auto records activating automatic split-L2: `0 of 150`
- Distinct current-auto effective split-L2 values: `0.000000`

Python public current-auto did not activate the engine-only auto split-L2 rule in this matrix.

## Decision

Keep the production auto-policy heuristics unchanged. Neither experimental candidate met the predeclared 1% overall improvement requirement without a protected shape/objective regression.

## Resolved Policy Diagnostics

| Fixture | Seed | Arm | Mode | Requested rounds | Round cap | Min rows | Min split gain | Row sample | Col sample | Auto split-L2 | Effective split-L2 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| small-narrow-512x8-regression | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-regression | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-sparse_regression | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-binary | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-multiclass | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-512x8-ranking | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-regression | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-sparse_regression | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-binary | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-multiclass | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 7 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 7 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 7 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 13 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 13 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 13 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 29 | current_auto | auto | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 29 | no_gain_floor | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-narrow-1023x16-ranking | 29 | quality_first | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-regression | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-sparse_regression | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 7 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 7 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 7 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 13 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 13 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 13 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 29 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 29 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-binary | 29 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-multiclass | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-512x128-ranking | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-regression | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-sparse_regression | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 7 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 7 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 7 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 13 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 13 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 13 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 29 | current_auto | auto | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 29 | no_gain_floor | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-binary | 29 | quality_first | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-multiclass | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 7 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 7 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 7 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 7 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 13 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 13 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 13 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 13 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 29 | current_auto | auto | 300 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 29 | manual_default | manual | 300 | 300 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 29 | no_gain_floor | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| small-wide-1023x256-ranking | 29 | quality_first | manual | 96 | 96 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-regression | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-sparse_regression | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-binary | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-multiclass | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-2048x16-ranking | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-sparse_regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-binary | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-multiclass | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 7 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 13 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 29 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 1.0 | False | 0.0 |
| medium-narrow-8192x16-ranking | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 7 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 13 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 29 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-regression | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-regression | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 7 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 13 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 29 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-sparse_regression | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 7 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 13 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 29 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-binary | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-binary | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 7 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 13 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 29 | current_auto | auto | 40 | 40 | 8 | 9.999999747378752e-05 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-multiclass | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 7 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 7 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 7 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 13 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 13 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 13 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 29 | current_auto | auto | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-2048x128-ranking | 29 | no_gain_floor | manual | 40 | 40 | 8 | 0.0 | 0.8999999761581421 | 0.6499999761581421 | False | 0.0 |
| medium-wide-2048x128-ranking | 29 | quality_first | manual | 40 | 40 | 8 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-sparse_regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-binary | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-binary | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-multiclass | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 7 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 13 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 29 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| medium-wide-8192x256-ranking | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.8999999761581421 | 0.5 | False | 0.0 |
| medium-wide-8192x256-ranking | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-sparse_regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-binary | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-multiclass | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 7 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 13 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 29 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 1.0 | False | 0.0 |
| large-narrow-16384x16-ranking | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-sparse_regression | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-binary | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-binary | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 7 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 13 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 29 | current_auto | auto | 40 | 40 | 16 | 9.999999747378752e-05 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-multiclass | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-multiclass | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 7 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 7 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 7 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 7 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 13 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 13 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 13 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 13 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 29 | current_auto | auto | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 29 | manual_default | manual | 40 | 40 | 1 | 0.0 | 1.0 | 1.0 | False | 0.0 |
| large-wide-16384x256-ranking | 29 | no_gain_floor | manual | 40 | 40 | 16 | 0.0 | 0.800000011920929 | 0.5 | False | 0.0 |
| large-wide-16384x256-ranking | 29 | quality_first | manual | 40 | 40 | 16 | 0.0 | 1.0 | 1.0 | False | 0.0 |

## Records

| Fixture | Stratum | Objective | Seed | Arm | Primary loss | Accuracy | NDCG@10 | Rounds | Fit seconds | Error |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| small-narrow-512x8-regression | small-narrow | regression | 7 | current_auto | 1.356280 |  |  | 40 | 0.012937 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | manual_default | 1.356280 |  |  | 40 | 0.011668 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | no_gain_floor | 1.356280 |  |  | 40 | 0.011815 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | quality_first | 1.356280 |  |  | 40 | 0.011809 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | current_auto | 1.246731 |  |  | 40 | 0.012003 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | manual_default | 1.246731 |  |  | 40 | 0.011747 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | no_gain_floor | 1.246731 |  |  | 40 | 0.011476 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | quality_first | 1.246731 |  |  | 40 | 0.011787 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | current_auto | 1.401883 |  |  | 40 | 0.011815 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | manual_default | 1.401883 |  |  | 40 | 0.011640 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | no_gain_floor | 1.401883 |  |  | 40 | 0.012048 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | quality_first | 1.401883 |  |  | 40 | 0.011992 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | current_auto | 0.945423 |  |  | 40 | 0.005738 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | manual_default | 0.945423 |  |  | 40 | 0.006160 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | no_gain_floor | 0.945423 |  |  | 40 | 0.005976 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | quality_first | 0.945423 |  |  | 40 | 0.006299 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | current_auto | 1.350555 |  |  | 40 | 0.005859 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | manual_default | 1.350555 |  |  | 40 | 0.005720 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | no_gain_floor | 1.350555 |  |  | 40 | 0.005828 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | quality_first | 1.350555 |  |  | 40 | 0.005715 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | current_auto | 0.965109 |  |  | 40 | 0.006183 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | manual_default | 0.965109 |  |  | 40 | 0.006780 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | no_gain_floor | 0.965109 |  |  | 40 | 0.005884 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | quality_first | 0.965109 |  |  | 40 | 0.005919 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | current_auto | 0.587465 | 0.742188 |  | 40 | 0.011946 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | manual_default | 0.587465 | 0.742188 |  | 40 | 0.011924 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | no_gain_floor | 0.587465 | 0.742188 |  | 40 | 0.011927 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | quality_first | 0.587465 | 0.742188 |  | 40 | 0.011978 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | current_auto | 0.466435 | 0.812500 |  | 40 | 0.012238 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | manual_default | 0.466435 | 0.812500 |  | 40 | 0.012069 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | no_gain_floor | 0.466435 | 0.812500 |  | 40 | 0.012086 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | quality_first | 0.466435 | 0.812500 |  | 40 | 0.012003 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | current_auto | 0.538330 | 0.757812 |  | 40 | 0.012288 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | manual_default | 0.538330 | 0.757812 |  | 40 | 0.012416 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | no_gain_floor | 0.538330 | 0.757812 |  | 40 | 0.012121 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | quality_first | 0.538330 | 0.757812 |  | 40 | 0.011715 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | current_auto | 1.121017 | 0.539062 |  | 40 | 0.049117 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | manual_default | 1.121017 | 0.539062 |  | 40 | 0.049949 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | no_gain_floor | 1.121017 | 0.539062 |  | 40 | 0.048965 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | quality_first | 1.121017 | 0.539062 |  | 40 | 0.049451 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | current_auto | 1.076570 | 0.562500 |  | 40 | 0.050053 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | manual_default | 1.076570 | 0.562500 |  | 40 | 0.049707 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | no_gain_floor | 1.076570 | 0.562500 |  | 40 | 0.049223 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | quality_first | 1.076570 | 0.562500 |  | 40 | 0.049167 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | current_auto | 1.094530 | 0.617188 |  | 40 | 0.049369 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | manual_default | 1.094530 | 0.617188 |  | 40 | 0.049524 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | no_gain_floor | 1.094530 | 0.617188 |  | 40 | 0.049443 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | quality_first | 1.094530 | 0.617188 |  | 40 | 0.049733 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | current_auto | 0.045750 |  | 0.954250 | 40 | 0.015283 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | manual_default | 0.045750 |  | 0.954250 | 40 | 0.014494 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | no_gain_floor | 0.045750 |  | 0.954250 | 40 | 0.014952 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | quality_first | 0.045750 |  | 0.954250 | 40 | 0.014008 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | current_auto | 0.054510 |  | 0.945490 | 40 | 0.014194 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | manual_default | 0.054510 |  | 0.945490 | 40 | 0.014306 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | no_gain_floor | 0.054510 |  | 0.945490 | 40 | 0.014122 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | quality_first | 0.054510 |  | 0.945490 | 40 | 0.014047 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | current_auto | 0.109613 |  | 0.890387 | 40 | 0.014545 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | manual_default | 0.109613 |  | 0.890387 | 40 | 0.014192 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | no_gain_floor | 0.109613 |  | 0.890387 | 40 | 0.014292 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | quality_first | 0.109613 |  | 0.890387 | 40 | 0.014197 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | current_auto | 1.106549 |  |  | 40 | 0.020729 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | manual_default | 1.106549 |  |  | 40 | 0.020373 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | no_gain_floor | 1.106549 |  |  | 40 | 0.020272 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | quality_first | 1.106549 |  |  | 40 | 0.020411 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | current_auto | 1.023434 |  |  | 40 | 0.021011 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | manual_default | 1.023434 |  |  | 40 | 0.020320 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | no_gain_floor | 1.023434 |  |  | 40 | 0.020325 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | quality_first | 1.023434 |  |  | 40 | 0.020395 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | current_auto | 0.989188 |  |  | 40 | 0.020462 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | manual_default | 0.989188 |  |  | 40 | 0.020468 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | no_gain_floor | 0.989188 |  |  | 40 | 0.020598 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | quality_first | 0.989188 |  |  | 40 | 0.020325 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | current_auto | 1.088986 |  |  | 40 | 0.013155 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | manual_default | 1.088986 |  |  | 40 | 0.012799 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | no_gain_floor | 1.088986 |  |  | 40 | 0.012757 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | quality_first | 1.088986 |  |  | 40 | 0.013314 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | current_auto | 1.129539 |  |  | 40 | 0.013249 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | manual_default | 1.129539 |  |  | 40 | 0.013096 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | no_gain_floor | 1.129539 |  |  | 40 | 0.013421 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | quality_first | 1.129539 |  |  | 40 | 0.013143 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | current_auto | 1.197693 |  |  | 40 | 0.012496 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | manual_default | 1.197693 |  |  | 40 | 0.012873 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | no_gain_floor | 1.197693 |  |  | 40 | 0.012899 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | quality_first | 1.197693 |  |  | 40 | 0.012981 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | current_auto | 0.411863 | 0.815686 |  | 40 | 0.021246 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | manual_default | 0.411863 | 0.815686 |  | 40 | 0.020748 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | no_gain_floor | 0.411863 | 0.815686 |  | 40 | 0.019893 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | quality_first | 0.411863 | 0.815686 |  | 40 | 0.020420 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | current_auto | 0.573400 | 0.713725 |  | 40 | 0.021162 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | manual_default | 0.573400 | 0.713725 |  | 40 | 0.019588 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | no_gain_floor | 0.573400 | 0.713725 |  | 40 | 0.020050 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | quality_first | 0.573400 | 0.713725 |  | 40 | 0.020445 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | current_auto | 0.501474 | 0.760784 |  | 40 | 0.020553 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | manual_default | 0.501474 | 0.760784 |  | 40 | 0.020606 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | no_gain_floor | 0.501474 | 0.760784 |  | 40 | 0.020182 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | quality_first | 0.501474 | 0.760784 |  | 40 | 0.020078 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | current_auto | 0.832246 | 0.670588 |  | 40 | 0.073665 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | manual_default | 0.832246 | 0.670588 |  | 40 | 0.074365 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | no_gain_floor | 0.832246 | 0.670588 |  | 40 | 0.074628 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | quality_first | 0.832246 | 0.670588 |  | 40 | 0.073633 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | current_auto | 1.011430 | 0.596078 |  | 40 | 0.074177 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | manual_default | 1.011430 | 0.596078 |  | 40 | 0.074989 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | no_gain_floor | 1.011430 | 0.596078 |  | 40 | 0.073883 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | quality_first | 1.011430 | 0.596078 |  | 40 | 0.074664 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | current_auto | 0.984423 | 0.623529 |  | 40 | 0.075525 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | manual_default | 0.984423 | 0.623529 |  | 40 | 0.074607 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | no_gain_floor | 0.984423 | 0.623529 |  | 40 | 0.075000 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | quality_first | 0.984423 | 0.623529 |  | 40 | 0.075796 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | current_auto | 0.020205 |  | 0.979795 | 40 | 0.023451 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | manual_default | 0.020205 |  | 0.979795 | 40 | 0.022963 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | no_gain_floor | 0.020205 |  | 0.979795 | 40 | 0.023073 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | quality_first | 0.020205 |  | 0.979795 | 40 | 0.023573 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | current_auto | 0.022759 |  | 0.977241 | 40 | 0.023993 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | manual_default | 0.022759 |  | 0.977241 | 40 | 0.022873 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | no_gain_floor | 0.022759 |  | 0.977241 | 40 | 0.023609 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | quality_first | 0.022759 |  | 0.977241 | 40 | 0.023951 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | current_auto | 0.055414 |  | 0.944586 | 40 | 0.022638 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | manual_default | 0.055414 |  | 0.944586 | 40 | 0.023079 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | no_gain_floor | 0.055414 |  | 0.944586 | 40 | 0.023775 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | quality_first | 0.055414 |  | 0.944586 | 40 | 0.023160 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | current_auto | 1.198654 |  |  | 96 | 0.171546 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | manual_default | 1.198586 |  |  | 300 | 0.480313 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | no_gain_floor | 1.198654 |  |  | 96 | 0.166534 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | quality_first | 1.198654 |  |  | 96 | 0.178744 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | current_auto | 1.394643 |  |  | 96 | 0.167911 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | manual_default | 1.393267 |  |  | 300 | 0.530578 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | no_gain_floor | 1.394643 |  |  | 96 | 0.163134 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | quality_first | 1.394643 |  |  | 96 | 0.202769 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | current_auto | 1.178323 |  |  | 96 | 0.165433 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | manual_default | 1.178138 |  |  | 300 | 0.482857 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | no_gain_floor | 1.178323 |  |  | 96 | 0.168319 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | quality_first | 1.178323 |  |  | 96 | 0.173313 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | current_auto | 1.124911 |  |  | 96 | 0.068033 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | manual_default | 1.193825 |  |  | 300 | 0.206791 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | no_gain_floor | 1.124911 |  |  | 96 | 0.067757 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | quality_first | 1.124911 |  |  | 96 | 0.067077 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | current_auto | 1.785465 |  |  | 96 | 0.071333 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | manual_default | 1.745283 |  |  | 300 | 0.211967 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | no_gain_floor | 1.785465 |  |  | 96 | 0.071018 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | quality_first | 1.785465 |  |  | 96 | 0.070259 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | current_auto | 1.392686 |  |  | 96 | 0.067677 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | manual_default | 1.414088 |  |  | 300 | 0.203854 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | no_gain_floor | 1.392686 |  |  | 96 | 0.068237 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | quality_first | 1.392686 |  |  | 96 | 0.068086 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | current_auto | 0.938921 | 0.718750 |  | 300 | 0.481220 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | manual_default | 0.938921 | 0.718750 |  | 300 | 0.478800 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | no_gain_floor | 0.938921 | 0.718750 |  | 300 | 0.476133 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | quality_first | 0.938921 | 0.718750 |  | 300 | 0.478303 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | current_auto | 0.811247 | 0.757812 |  | 300 | 0.460695 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | manual_default | 0.811247 | 0.757812 |  | 300 | 0.461934 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | no_gain_floor | 0.811247 | 0.757812 |  | 300 | 0.458074 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | quality_first | 0.811247 | 0.757812 |  | 300 | 0.459315 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | current_auto | 1.003693 | 0.804688 |  | 300 | 0.503279 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | manual_default | 1.003693 | 0.804688 |  | 300 | 0.501256 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | no_gain_floor | 1.003693 | 0.804688 |  | 300 | 0.499934 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | quality_first | 1.003693 | 0.804688 |  | 300 | 0.500397 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | current_auto | 1.508700 | 0.554688 |  | 96 | 0.603157 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | manual_default | 2.393751 | 0.585938 |  | 300 | 1.851916 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | no_gain_floor | 1.508700 | 0.554688 |  | 96 | 0.601210 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | quality_first | 1.508700 | 0.554688 |  | 96 | 0.603172 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | current_auto | 1.720285 | 0.570312 |  | 96 | 0.621608 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | manual_default | 2.595189 | 0.562500 |  | 300 | 1.862620 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | no_gain_floor | 1.720285 | 0.570312 |  | 96 | 0.618946 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | quality_first | 1.720285 | 0.570312 |  | 96 | 0.614767 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | current_auto | 1.518329 | 0.601562 |  | 96 | 0.619491 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | manual_default | 2.064210 | 0.593750 |  | 300 | 1.896226 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | no_gain_floor | 1.518329 | 0.601562 |  | 96 | 0.628887 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | quality_first | 1.518329 | 0.601562 |  | 96 | 0.624711 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | current_auto | 0.041628 |  | 0.958372 | 96 | 0.176443 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | manual_default | 0.036371 |  | 0.963629 | 300 | 0.552806 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | no_gain_floor | 0.041628 |  | 0.958372 | 96 | 0.178638 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | quality_first | 0.041628 |  | 0.958372 | 96 | 0.179729 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | current_auto | 0.044087 |  | 0.955913 | 96 | 0.179385 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | manual_default | 0.041659 |  | 0.958341 | 300 | 0.557818 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | no_gain_floor | 0.044087 |  | 0.955913 | 96 | 0.181250 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | quality_first | 0.044087 |  | 0.955913 | 96 | 0.179093 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | current_auto | 0.028972 |  | 0.971028 | 96 | 0.176894 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | manual_default | 0.020522 |  | 0.979478 | 300 | 0.553623 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | no_gain_floor | 0.028972 |  | 0.971028 | 96 | 0.177946 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | quality_first | 0.028972 |  | 0.971028 | 96 | 0.216842 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | current_auto | 1.032054 |  |  | 96 | 0.295655 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | manual_default | 1.033443 |  |  | 300 | 0.852796 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | no_gain_floor | 1.032054 |  |  | 96 | 0.292326 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | quality_first | 1.032054 |  |  | 96 | 0.323969 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | current_auto | 1.083313 |  |  | 96 | 0.303820 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | manual_default | 1.078950 |  |  | 300 | 0.888198 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | no_gain_floor | 1.083313 |  |  | 96 | 0.300954 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | quality_first | 1.083313 |  |  | 96 | 0.300732 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | current_auto | 1.109971 |  |  | 96 | 0.295758 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | manual_default | 1.115278 |  |  | 300 | 0.878670 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | no_gain_floor | 1.109971 |  |  | 96 | 0.293158 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | quality_first | 1.109971 |  |  | 96 | 0.309218 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | current_auto | 0.955110 |  |  | 96 | 0.091372 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | manual_default | 0.988537 |  |  | 300 | 0.272576 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | no_gain_floor | 0.955110 |  |  | 96 | 0.091992 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | quality_first | 0.955110 |  |  | 96 | 0.090381 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | current_auto | 1.020762 |  |  | 96 | 0.091425 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | manual_default | 1.047432 |  |  | 300 | 0.292672 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | no_gain_floor | 1.020762 |  |  | 96 | 0.110704 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | quality_first | 1.020762 |  |  | 96 | 0.142858 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | current_auto | 1.243298 |  |  | 96 | 0.108538 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | manual_default | 1.272168 |  |  | 300 | 0.296683 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | no_gain_floor | 1.243298 |  |  | 96 | 0.100469 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | quality_first | 1.243298 |  |  | 96 | 0.102457 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | current_auto | 1.115527 | 0.713725 |  | 300 | 0.980988 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | manual_default | 1.115527 | 0.713725 |  | 300 | 0.907292 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | no_gain_floor | 1.115527 | 0.713725 |  | 300 | 0.908394 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | quality_first | 1.115527 | 0.713725 |  | 300 | 0.912700 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | current_auto | 0.863139 | 0.737255 |  | 300 | 0.912587 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | manual_default | 0.863139 | 0.737255 |  | 300 | 0.914768 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | no_gain_floor | 0.863139 | 0.737255 |  | 300 | 0.911654 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | quality_first | 0.863139 | 0.737255 |  | 300 | 0.914172 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | current_auto | 0.730849 | 0.725490 |  | 300 | 0.991611 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | manual_default | 0.730849 | 0.725490 |  | 300 | 0.897898 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | no_gain_floor | 0.730849 | 0.725490 |  | 300 | 0.909498 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | quality_first | 0.730849 | 0.725490 |  | 300 | 0.908816 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | current_auto | 1.240497 | 0.592157 |  | 96 | 1.152411 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | manual_default | 2.091386 | 0.588235 |  | 300 | 3.550460 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | no_gain_floor | 1.240497 | 0.592157 |  | 96 | 1.149350 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | quality_first | 1.240497 | 0.592157 |  | 96 | 1.153432 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | current_auto | 1.235065 | 0.588235 |  | 96 | 1.188142 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | manual_default | 2.177473 | 0.592157 |  | 300 | 3.648881 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | no_gain_floor | 1.235065 | 0.588235 |  | 96 | 1.185330 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | quality_first | 1.235065 | 0.588235 |  | 96 | 1.183286 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | current_auto | 1.214883 | 0.568627 |  | 96 | 1.155545 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | manual_default | 2.209509 | 0.568627 |  | 300 | 3.651295 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | no_gain_floor | 1.214883 | 0.568627 |  | 96 | 1.204563 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | quality_first | 1.214883 | 0.568627 |  | 96 | 1.166149 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | current_auto | 0.046265 |  | 0.953735 | 96 | 0.335376 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | manual_default | 0.038047 |  | 0.961953 | 300 | 1.003295 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | no_gain_floor | 0.046265 |  | 0.953735 | 96 | 0.324072 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | quality_first | 0.046265 |  | 0.953735 | 96 | 0.327306 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | current_auto | 0.045150 |  | 0.954850 | 96 | 0.324802 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | manual_default | 0.034720 |  | 0.965280 | 300 | 1.001973 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | no_gain_floor | 0.045150 |  | 0.954850 | 96 | 0.323166 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | quality_first | 0.045150 |  | 0.954850 | 96 | 0.320223 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | current_auto | 0.040948 |  | 0.959052 | 96 | 0.323240 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | manual_default | 0.039767 |  | 0.960233 | 300 | 1.002507 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | no_gain_floor | 0.040948 |  | 0.959052 | 96 | 0.321026 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | quality_first | 0.040948 |  | 0.959052 | 96 | 0.324664 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | current_auto | 0.936773 |  |  | 40 | 0.026101 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | manual_default | 0.959450 |  |  | 40 | 0.027252 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | no_gain_floor | 0.936773 |  |  | 40 | 0.027502 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | quality_first | 0.954618 |  |  | 40 | 0.027037 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | current_auto | 0.925707 |  |  | 40 | 0.027157 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | manual_default | 0.962787 |  |  | 40 | 0.027701 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | no_gain_floor | 0.925707 |  |  | 40 | 0.027680 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | quality_first | 0.955503 |  |  | 40 | 0.026778 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | current_auto | 0.991331 |  |  | 40 | 0.027815 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | manual_default | 1.010573 |  |  | 40 | 0.027053 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | no_gain_floor | 0.991331 |  |  | 40 | 0.027623 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | quality_first | 1.009384 |  |  | 40 | 0.027011 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | current_auto | 1.165999 |  |  | 40 | 0.018684 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | manual_default | 1.143000 |  |  | 40 | 0.018353 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | no_gain_floor | 1.165999 |  |  | 40 | 0.017739 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | quality_first | 1.209558 |  |  | 40 | 0.018301 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | current_auto | 1.168801 |  |  | 40 | 0.017631 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | manual_default | 1.161288 |  |  | 40 | 0.018319 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | no_gain_floor | 1.168801 |  |  | 40 | 0.018080 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | quality_first | 1.156744 |  |  | 40 | 0.018025 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | current_auto | 1.213547 |  |  | 40 | 0.017587 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | manual_default | 1.252843 |  |  | 40 | 0.018594 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | no_gain_floor | 1.213547 |  |  | 40 | 0.017499 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | quality_first | 1.183996 |  |  | 40 | 0.017787 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | current_auto | 0.470797 | 0.767578 |  | 40 | 0.026378 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | manual_default | 0.470915 | 0.753906 |  | 40 | 0.026378 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | no_gain_floor | 0.470797 | 0.767578 |  | 40 | 0.026212 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | quality_first | 0.467156 | 0.769531 |  | 40 | 0.027209 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | current_auto | 0.488263 | 0.746094 |  | 40 | 0.028043 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | manual_default | 0.499859 | 0.740234 |  | 40 | 0.027479 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | no_gain_floor | 0.488263 | 0.746094 |  | 40 | 0.027981 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | quality_first | 0.491261 | 0.748047 |  | 40 | 0.027301 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | current_auto | 0.487371 | 0.744141 |  | 40 | 0.026665 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | manual_default | 0.487910 | 0.757812 |  | 40 | 0.027943 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | no_gain_floor | 0.487371 | 0.744141 |  | 40 | 0.027506 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | quality_first | 0.488356 | 0.750000 |  | 40 | 0.026865 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | current_auto | 0.888976 | 0.617188 |  | 40 | 0.099406 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | manual_default | 0.916457 | 0.621094 |  | 40 | 0.103860 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | no_gain_floor | 0.888976 | 0.617188 |  | 40 | 0.100972 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | quality_first | 0.891970 | 0.623047 |  | 40 | 0.103415 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | current_auto | 0.837018 | 0.652344 |  | 40 | 0.098557 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | manual_default | 0.833373 | 0.658203 |  | 40 | 0.102675 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | no_gain_floor | 0.837018 | 0.652344 |  | 40 | 0.098727 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | quality_first | 0.830199 | 0.656250 |  | 40 | 0.101053 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | current_auto | 0.952523 | 0.591797 |  | 40 | 0.100702 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | manual_default | 0.983401 | 0.566406 |  | 40 | 0.103117 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | no_gain_floor | 0.952523 | 0.591797 |  | 40 | 0.097978 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | quality_first | 0.982138 | 0.568359 |  | 40 | 0.099792 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | current_auto | 0.056037 |  | 0.943963 | 40 | 0.031488 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | manual_default | 0.049562 |  | 0.950438 | 40 | 0.031452 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | no_gain_floor | 0.056037 |  | 0.943963 | 40 | 0.032015 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | quality_first | 0.055997 |  | 0.944003 | 40 | 0.032378 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | current_auto | 0.054065 |  | 0.945935 | 40 | 0.032722 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | manual_default | 0.061249 |  | 0.938751 | 40 | 0.031825 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | no_gain_floor | 0.054065 |  | 0.945935 | 40 | 0.033430 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | quality_first | 0.059670 |  | 0.940330 | 40 | 0.032556 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | current_auto | 0.043007 |  | 0.956993 | 40 | 0.032715 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | manual_default | 0.059333 |  | 0.940667 | 40 | 0.032011 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | no_gain_floor | 0.043007 |  | 0.956993 | 40 | 0.032961 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | quality_first | 0.044501 |  | 0.955499 | 40 | 0.033104 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | current_auto | 0.907927 |  |  | 40 | 0.057155 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | manual_default | 0.910924 |  |  | 40 | 0.055586 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | no_gain_floor | 0.907927 |  |  | 40 | 0.056211 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | quality_first | 0.910832 |  |  | 40 | 0.054856 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | current_auto | 0.950398 |  |  | 40 | 0.054663 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | manual_default | 0.956071 |  |  | 40 | 0.055074 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | no_gain_floor | 0.950398 |  |  | 40 | 0.055055 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | quality_first | 0.951897 |  |  | 40 | 0.054548 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | current_auto | 0.928650 |  |  | 40 | 0.055277 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | manual_default | 0.939883 |  |  | 40 | 0.054147 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | no_gain_floor | 0.928650 |  |  | 40 | 0.055439 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | quality_first | 0.932671 |  |  | 40 | 0.054868 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | current_auto | 1.070219 |  |  | 40 | 0.046116 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | manual_default | 1.092623 |  |  | 40 | 0.045523 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | no_gain_floor | 1.070219 |  |  | 40 | 0.045793 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | quality_first | 1.066447 |  |  | 40 | 0.045794 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | current_auto | 1.121034 |  |  | 40 | 0.045147 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | manual_default | 1.125220 |  |  | 40 | 0.044824 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | no_gain_floor | 1.121034 |  |  | 40 | 0.045893 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | quality_first | 1.132881 |  |  | 40 | 0.044887 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | current_auto | 0.976190 |  |  | 40 | 0.045390 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | manual_default | 0.964284 |  |  | 40 | 0.045066 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | no_gain_floor | 0.976190 |  |  | 40 | 0.045530 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | quality_first | 0.971045 |  |  | 40 | 0.045142 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | current_auto | 0.438248 | 0.794922 |  | 40 | 0.059757 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | manual_default | 0.440579 | 0.803711 |  | 40 | 0.056872 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | no_gain_floor | 0.438248 | 0.794922 |  | 40 | 0.058376 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | quality_first | 0.440884 | 0.791992 |  | 40 | 0.057182 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | current_auto | 0.448574 | 0.783203 |  | 40 | 0.059283 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | manual_default | 0.446551 | 0.790039 |  | 40 | 0.056761 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | no_gain_floor | 0.448574 | 0.783203 |  | 40 | 0.057629 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | quality_first | 0.448503 | 0.777344 |  | 40 | 0.057565 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | current_auto | 0.467210 | 0.782227 |  | 40 | 0.058002 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | manual_default | 0.468290 | 0.773438 |  | 40 | 0.056710 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | no_gain_floor | 0.467210 | 0.782227 |  | 40 | 0.057491 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | quality_first | 0.469317 | 0.775391 |  | 40 | 0.056906 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | current_auto | 0.873238 | 0.636719 |  | 40 | 0.217307 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | manual_default | 0.873152 | 0.643555 |  | 40 | 0.223947 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | no_gain_floor | 0.873240 | 0.636719 |  | 40 | 0.218429 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | quality_first | 0.875250 | 0.634766 |  | 40 | 0.222519 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | current_auto | 0.883946 | 0.636719 |  | 40 | 0.217736 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | manual_default | 0.884437 | 0.643555 |  | 40 | 0.221149 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | no_gain_floor | 0.883945 | 0.636719 |  | 40 | 0.218556 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | quality_first | 0.885673 | 0.643555 |  | 40 | 0.221697 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | current_auto | 0.871736 | 0.652344 |  | 40 | 0.217431 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | manual_default | 0.879004 | 0.650391 |  | 40 | 0.221177 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | no_gain_floor | 0.871737 | 0.652344 |  | 40 | 0.216300 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | quality_first | 0.876522 | 0.651367 |  | 40 | 0.220090 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | current_auto | 0.036837 |  | 0.963163 | 40 | 0.072214 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | manual_default | 0.036345 |  | 0.963655 | 40 | 0.071079 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | no_gain_floor | 0.036837 |  | 0.963163 | 40 | 0.073318 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | quality_first | 0.038254 |  | 0.961746 | 40 | 0.072510 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | current_auto | 0.040332 |  | 0.959668 | 40 | 0.074964 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | manual_default | 0.047581 |  | 0.952419 | 40 | 0.073253 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | no_gain_floor | 0.040332 |  | 0.959668 | 40 | 0.074616 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | quality_first | 0.040932 |  | 0.959068 | 40 | 0.073732 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | current_auto | 0.032483 |  | 0.967517 | 40 | 0.074088 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | manual_default | 0.029343 |  | 0.970657 | 40 | 0.073699 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | no_gain_floor | 0.032483 |  | 0.967517 | 40 | 0.074637 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | quality_first | 0.026819 |  | 0.973181 | 40 | 0.073706 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | current_auto | 1.139516 |  |  | 40 | 0.070614 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | manual_default | 1.001547 |  |  | 40 | 0.089805 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | no_gain_floor | 1.139516 |  |  | 40 | 0.072763 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | quality_first | 0.991976 |  |  | 40 | 0.088586 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | current_auto | 1.258807 |  |  | 40 | 0.071908 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | manual_default | 1.088536 |  |  | 40 | 0.087928 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | no_gain_floor | 1.258807 |  |  | 40 | 0.070748 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | quality_first | 1.094273 |  |  | 40 | 0.086553 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | current_auto | 1.330950 |  |  | 40 | 0.073880 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | manual_default | 1.094226 |  |  | 40 | 0.089038 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | no_gain_floor | 1.330950 |  |  | 40 | 0.072224 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | quality_first | 1.085990 |  |  | 40 | 0.087293 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | current_auto | 0.967074 |  |  | 40 | 0.030165 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | manual_default | 1.006154 |  |  | 40 | 0.036425 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | no_gain_floor | 0.967074 |  |  | 40 | 0.029296 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | quality_first | 0.985487 |  |  | 40 | 0.034310 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | current_auto | 0.965502 |  |  | 40 | 0.030762 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | manual_default | 1.038211 |  |  | 40 | 0.035425 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | no_gain_floor | 0.965502 |  |  | 40 | 0.030848 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | quality_first | 0.980549 |  |  | 40 | 0.034706 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | current_auto | 1.009232 |  |  | 40 | 0.029845 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | manual_default | 0.978235 |  |  | 40 | 0.035353 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | no_gain_floor | 1.009232 |  |  | 40 | 0.030032 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | quality_first | 0.893986 |  |  | 40 | 0.033760 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | current_auto | 0.479360 | 0.769531 |  | 40 | 0.071912 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | manual_default | 0.474076 | 0.769531 |  | 40 | 0.088169 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | no_gain_floor | 0.479360 | 0.769531 |  | 40 | 0.071648 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | quality_first | 0.465135 | 0.775391 |  | 40 | 0.086858 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | current_auto | 0.467152 | 0.781250 |  | 40 | 0.071612 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | manual_default | 0.456495 | 0.791016 |  | 40 | 0.087994 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | no_gain_floor | 0.467152 | 0.781250 |  | 40 | 0.072036 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | quality_first | 0.456160 | 0.800781 |  | 40 | 0.091085 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | current_auto | 0.462049 | 0.787109 |  | 40 | 0.071389 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | manual_default | 0.440985 | 0.802734 |  | 40 | 0.087947 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | no_gain_floor | 0.462049 | 0.787109 |  | 40 | 0.071339 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | quality_first | 0.434184 | 0.802734 |  | 40 | 0.085764 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | current_auto | 0.906088 | 0.613281 |  | 40 | 0.268313 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | manual_default | 0.925677 | 0.609375 |  | 40 | 0.339908 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | no_gain_floor | 0.906435 | 0.613281 |  | 40 | 0.263090 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | quality_first | 0.915058 | 0.623047 |  | 40 | 0.329448 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | current_auto | 0.932231 | 0.619141 |  | 40 | 0.264204 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | manual_default | 0.943052 | 0.601562 |  | 40 | 0.341054 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | no_gain_floor | 0.931674 | 0.621094 |  | 40 | 0.266481 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | quality_first | 0.928870 | 0.605469 |  | 40 | 0.332655 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | current_auto | 0.903675 | 0.650391 |  | 40 | 0.268441 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | manual_default | 0.901041 | 0.634766 |  | 40 | 0.337718 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | no_gain_floor | 0.903674 | 0.650391 |  | 40 | 0.264687 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | quality_first | 0.878114 | 0.654297 |  | 40 | 0.322394 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | current_auto | 0.052335 |  | 0.947665 | 40 | 0.078597 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | manual_default | 0.040324 |  | 0.959676 | 40 | 0.094696 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | no_gain_floor | 0.052335 |  | 0.947665 | 40 | 0.079208 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | quality_first | 0.043374 |  | 0.956626 | 40 | 0.094335 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | current_auto | 0.054497 |  | 0.945503 | 40 | 0.077663 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | manual_default | 0.054565 |  | 0.945435 | 40 | 0.094181 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | no_gain_floor | 0.054497 |  | 0.945503 | 40 | 0.077878 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | quality_first | 0.055063 |  | 0.944937 | 40 | 0.093475 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | current_auto | 0.054830 |  | 0.945170 | 40 | 0.076140 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | manual_default | 0.044983 |  | 0.955017 | 40 | 0.093112 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | no_gain_floor | 0.054830 |  | 0.945170 | 40 | 0.075185 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | quality_first | 0.043895 |  | 0.956105 | 40 | 0.092717 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | current_auto | 1.404662 |  |  | 40 | 0.147942 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | manual_default | 0.988130 |  |  | 40 | 0.216311 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | no_gain_floor | 1.404662 |  |  | 40 | 0.143662 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | quality_first | 0.992830 |  |  | 40 | 0.217038 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | current_auto | 1.397172 |  |  | 40 | 0.152069 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | manual_default | 0.973534 |  |  | 40 | 0.217341 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | no_gain_floor | 1.397172 |  |  | 40 | 0.184404 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | quality_first | 0.974189 |  |  | 40 | 0.216061 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | current_auto | 1.319181 |  |  | 40 | 0.148655 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | manual_default | 0.928280 |  |  | 40 | 0.219525 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | no_gain_floor | 1.319181 |  |  | 40 | 0.142970 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | quality_first | 0.920007 |  |  | 40 | 0.216610 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | current_auto | 0.975513 |  |  | 40 | 0.078960 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | manual_default | 0.917171 |  |  | 40 | 0.111281 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | no_gain_floor | 0.975513 |  |  | 40 | 0.075749 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | quality_first | 0.934531 |  |  | 40 | 0.108698 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | current_auto | 1.018249 |  |  | 40 | 0.079694 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | manual_default | 0.957270 |  |  | 40 | 0.111393 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | no_gain_floor | 1.018249 |  |  | 40 | 0.076660 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | quality_first | 0.920485 |  |  | 40 | 0.110385 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | current_auto | 1.049444 |  |  | 40 | 0.079596 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | manual_default | 0.950134 |  |  | 40 | 0.112327 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | no_gain_floor | 1.049444 |  |  | 40 | 0.075434 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | quality_first | 0.953436 |  |  | 40 | 0.109966 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | current_auto | 0.477624 | 0.765625 |  | 40 | 0.150407 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | manual_default | 0.454305 | 0.772461 |  | 40 | 0.221912 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | no_gain_floor | 0.477624 | 0.765625 |  | 40 | 0.144723 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | quality_first | 0.455709 | 0.785156 |  | 40 | 0.219799 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | current_auto | 0.487223 | 0.770508 |  | 40 | 0.146627 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | manual_default | 0.463858 | 0.785156 |  | 40 | 0.220120 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | no_gain_floor | 0.487223 | 0.770508 |  | 40 | 0.143356 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | quality_first | 0.463091 | 0.782227 |  | 40 | 0.219282 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | current_auto | 0.501657 | 0.762695 |  | 40 | 0.150017 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | manual_default | 0.482422 | 0.762695 |  | 40 | 0.220667 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | no_gain_floor | 0.501657 | 0.762695 |  | 40 | 0.143775 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | quality_first | 0.482142 | 0.759766 |  | 40 | 0.222356 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | current_auto | 0.914937 | 0.622070 |  | 40 | 0.545823 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | manual_default | 0.900804 | 0.625000 |  | 40 | 0.832742 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | no_gain_floor | 0.914937 | 0.622070 |  | 40 | 0.525482 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | quality_first | 0.901093 | 0.630859 |  | 40 | 0.825109 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | current_auto | 0.902941 | 0.642578 |  | 40 | 0.528410 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | manual_default | 0.884845 | 0.639648 |  | 40 | 0.866967 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | no_gain_floor | 0.902941 | 0.642578 |  | 40 | 0.522996 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | quality_first | 0.877802 | 0.645508 |  | 40 | 0.822188 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | current_auto | 0.932150 | 0.608398 |  | 40 | 0.587495 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | manual_default | 0.925418 | 0.602539 |  | 40 | 0.981715 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | no_gain_floor | 0.932150 | 0.608398 |  | 40 | 0.536526 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | quality_first | 0.924685 | 0.612305 |  | 40 | 0.829867 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | current_auto | 0.038051 |  | 0.961949 | 40 | 0.189474 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | manual_default | 0.030179 |  | 0.969821 | 40 | 0.292718 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | no_gain_floor | 0.038051 |  | 0.961949 | 40 | 0.197761 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | quality_first | 0.028134 |  | 0.971866 | 40 | 0.260663 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | current_auto | 0.046119 |  | 0.953881 | 40 | 0.165936 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | manual_default | 0.045575 |  | 0.954425 | 40 | 0.237909 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | no_gain_floor | 0.046119 |  | 0.953881 | 40 | 0.160044 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | quality_first | 0.044349 |  | 0.955651 | 40 | 0.239937 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | current_auto | 0.041931 |  | 0.958069 | 40 | 0.167758 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | manual_default | 0.031306 |  | 0.968694 | 40 | 0.238959 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | no_gain_floor | 0.041931 |  | 0.958069 | 40 | 0.162197 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | quality_first | 0.034885 |  | 0.965115 | 40 | 0.241514 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | current_auto | 0.911667 |  |  | 40 | 0.092486 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | manual_default | 0.899458 |  |  | 40 | 0.089329 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | no_gain_floor | 0.911667 |  |  | 40 | 0.088572 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | quality_first | 0.903352 |  |  | 40 | 0.089080 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | current_auto | 0.888338 |  |  | 40 | 0.089391 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | manual_default | 0.907969 |  |  | 40 | 0.089226 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | no_gain_floor | 0.888338 |  |  | 40 | 0.088062 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | quality_first | 0.902927 |  |  | 40 | 0.089455 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | current_auto | 0.926783 |  |  | 40 | 0.089057 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | manual_default | 0.938812 |  |  | 40 | 0.088741 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | no_gain_floor | 0.926783 |  |  | 40 | 0.089709 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | quality_first | 0.934912 |  |  | 40 | 0.090030 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | current_auto | 1.020661 |  |  | 40 | 0.078929 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | manual_default | 1.015931 |  |  | 40 | 0.083215 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | no_gain_floor | 1.020661 |  |  | 40 | 0.078897 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | quality_first | 1.015392 |  |  | 40 | 0.083684 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | current_auto | 0.955216 |  |  | 40 | 0.078500 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | manual_default | 0.928215 |  |  | 40 | 0.083773 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | no_gain_floor | 0.955216 |  |  | 40 | 0.077383 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | quality_first | 0.944869 |  |  | 40 | 0.083281 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | current_auto | 1.061646 |  |  | 40 | 0.078743 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | manual_default | 1.005228 |  |  | 40 | 0.082981 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | no_gain_floor | 1.061646 |  |  | 40 | 0.078926 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | quality_first | 1.064457 |  |  | 40 | 0.083231 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | current_auto | 0.442674 | 0.795898 |  | 40 | 0.096663 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | manual_default | 0.440494 | 0.800781 |  | 40 | 0.095512 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | no_gain_floor | 0.442674 | 0.795898 |  | 40 | 0.095675 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | quality_first | 0.443080 | 0.799805 |  | 40 | 0.097076 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | current_auto | 0.450509 | 0.787109 |  | 40 | 0.096900 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | manual_default | 0.447729 | 0.788086 |  | 40 | 0.097122 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | no_gain_floor | 0.450509 | 0.787109 |  | 40 | 0.095160 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | quality_first | 0.450698 | 0.784180 |  | 40 | 0.096581 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | current_auto | 0.449667 | 0.777344 |  | 40 | 0.096721 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | manual_default | 0.451824 | 0.778320 |  | 40 | 0.096114 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | no_gain_floor | 0.449667 | 0.777344 |  | 40 | 0.096202 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | quality_first | 0.452321 | 0.782227 |  | 40 | 0.094868 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | current_auto | 0.844281 | 0.646484 |  | 40 | 0.359853 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | manual_default | 0.843943 | 0.653320 |  | 40 | 0.379066 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | no_gain_floor | 0.844281 | 0.646484 |  | 40 | 0.360275 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | quality_first | 0.840072 | 0.654297 |  | 40 | 0.380991 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | current_auto | 0.847900 | 0.631836 |  | 40 | 0.361915 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | manual_default | 0.848186 | 0.639648 |  | 40 | 0.379611 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | no_gain_floor | 0.847900 | 0.631836 |  | 40 | 0.359623 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | quality_first | 0.842316 | 0.638672 |  | 40 | 0.381426 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | current_auto | 0.855436 | 0.656250 |  | 40 | 0.359404 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | manual_default | 0.854029 | 0.657227 |  | 40 | 0.376438 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | no_gain_floor | 0.855436 | 0.656250 |  | 40 | 0.358088 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | quality_first | 0.853687 | 0.659180 |  | 40 | 0.378273 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | current_auto | 0.028092 |  | 0.971908 | 40 | 0.120602 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | manual_default | 0.026607 |  | 0.973393 | 40 | 0.120070 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | no_gain_floor | 0.028092 |  | 0.971908 | 40 | 0.121799 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | quality_first | 0.027389 |  | 0.972611 | 40 | 0.123796 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | current_auto | 0.030977 |  | 0.969023 | 40 | 0.125411 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | manual_default | 0.030402 |  | 0.969598 | 40 | 0.124050 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | no_gain_floor | 0.030977 |  | 0.969023 | 40 | 0.123942 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | quality_first | 0.032129 |  | 0.967871 | 40 | 0.123944 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | current_auto | 0.028123 |  | 0.971877 | 40 | 0.125136 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | manual_default | 0.029712 |  | 0.970288 | 40 | 0.123882 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | no_gain_floor | 0.028123 |  | 0.971877 | 40 | 0.123880 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | quality_first | 0.029030 |  | 0.970970 | 40 | 0.160877 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | current_auto | 1.254501 |  |  | 40 | 0.216488 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | manual_default | 0.876513 |  |  | 40 | 0.328108 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | no_gain_floor | 1.254501 |  |  | 40 | 0.206265 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | quality_first | 0.884073 |  |  | 40 | 0.329925 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | current_auto | 1.367765 |  |  | 40 | 0.218497 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | manual_default | 0.893836 |  |  | 40 | 0.332920 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | no_gain_floor | 1.367765 |  |  | 40 | 0.208600 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | quality_first | 0.897455 |  |  | 40 | 0.335105 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | current_auto | 1.206074 |  |  | 40 | 0.226939 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | manual_default | 0.834827 |  |  | 40 | 0.330783 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | no_gain_floor | 1.206074 |  |  | 40 | 0.215136 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | quality_first | 0.832478 |  |  | 40 | 0.348804 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | current_auto | 1.123433 |  |  | 40 | 0.150628 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | manual_default | 0.984147 |  |  | 40 | 0.202908 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | no_gain_floor | 1.123433 |  |  | 40 | 0.119971 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | quality_first | 1.003638 |  |  | 40 | 0.190834 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | current_auto | 1.064186 |  |  | 40 | 0.138066 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | manual_default | 0.953405 |  |  | 40 | 0.205525 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | no_gain_floor | 1.064186 |  |  | 40 | 0.118214 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | quality_first | 1.022102 |  |  | 40 | 0.186307 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | current_auto | 1.118783 |  |  | 40 | 0.135342 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | manual_default | 0.975560 |  |  | 40 | 0.197886 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | no_gain_floor | 1.118783 |  |  | 40 | 0.116597 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | quality_first | 1.014636 |  |  | 40 | 0.189549 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | current_auto | 0.487697 | 0.765625 |  | 40 | 0.250619 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | manual_default | 0.467187 | 0.772461 |  | 40 | 0.334969 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | no_gain_floor | 0.487697 | 0.765625 |  | 40 | 0.216492 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | quality_first | 0.467222 | 0.774414 |  | 40 | 0.350306 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | current_auto | 0.489207 | 0.773438 |  | 40 | 0.226107 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | manual_default | 0.465047 | 0.777344 |  | 40 | 0.348997 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | no_gain_floor | 0.489207 | 0.773438 |  | 40 | 0.215532 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | quality_first | 0.466821 | 0.775391 |  | 40 | 0.350658 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | current_auto | 0.502915 | 0.763672 |  | 40 | 0.226042 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | manual_default | 0.472571 | 0.777344 |  | 40 | 0.351737 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | no_gain_floor | 0.502915 | 0.763672 |  | 40 | 0.215977 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | quality_first | 0.471856 | 0.768555 |  | 40 | 0.355964 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | current_auto | 0.883432 | 0.633789 |  | 40 | 0.783022 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | manual_default | 0.878500 | 0.627930 |  | 40 | 1.322634 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | no_gain_floor | 0.883432 | 0.633789 |  | 40 | 0.775620 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | quality_first | 0.875623 | 0.624023 |  | 40 | 1.304135 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | current_auto | 0.902899 | 0.623047 |  | 40 | 0.772739 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | manual_default | 0.880031 | 0.627930 |  | 40 | 1.311463 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | no_gain_floor | 0.902899 | 0.623047 |  | 40 | 0.772800 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | quality_first | 0.879740 | 0.618164 |  | 40 | 1.292446 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | current_auto | 0.899215 | 0.644531 |  | 40 | 0.771138 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | manual_default | 0.878460 | 0.635742 |  | 40 | 1.290548 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | no_gain_floor | 0.899215 | 0.644531 |  | 40 | 0.771031 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | quality_first | 0.880410 | 0.637695 |  | 40 | 1.302177 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | current_auto | 0.043758 |  | 0.956242 | 40 | 0.257055 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | manual_default | 0.031206 |  | 0.968794 | 40 | 0.381842 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | no_gain_floor | 0.043758 |  | 0.956242 | 40 | 0.247298 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | quality_first | 0.034106 |  | 0.965894 | 40 | 0.382624 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | current_auto | 0.032260 |  | 0.967740 | 40 | 0.252877 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | manual_default | 0.027507 |  | 0.972493 | 40 | 0.381245 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | no_gain_floor | 0.032260 |  | 0.967740 | 40 | 0.243620 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | quality_first | 0.029329 |  | 0.970671 | 40 | 0.380197 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | current_auto | 0.040766 |  | 0.959234 | 40 | 0.256240 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | manual_default | 0.030942 |  | 0.969058 | 40 | 0.378379 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | no_gain_floor | 0.040766 |  | 0.959234 | 40 | 0.242459 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | quality_first | 0.031514 |  | 0.968486 | 40 | 0.380929 |  |
