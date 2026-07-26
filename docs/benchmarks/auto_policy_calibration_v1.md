# Auto-Policy Calibration Benchmark

- Command: `/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/auto_policy_benchmark.py --gate --output-json /tmp/auto-policy-calibration.json --output-report docs/benchmarks/auto_policy_calibration_v1.md`
- Selected outcome: `current_auto`
- Gate passed: `true`
- Timing is descriptive only and is not used as a quality gate.

## Environment

- Git commit: `8fb73802fc8618b4b629d8b6197180ce97593994`
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
| small-narrow-512x8-regression | small-narrow | regression | 7 | current_auto | 1.356280 |  |  | 40 | 0.012372 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | manual_default | 1.356280 |  |  | 40 | 0.011251 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | no_gain_floor | 1.356280 |  |  | 40 | 0.011371 |  |
| small-narrow-512x8-regression | small-narrow | regression | 7 | quality_first | 1.356280 |  |  | 40 | 0.011511 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | current_auto | 1.246731 |  |  | 40 | 0.011395 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | manual_default | 1.246731 |  |  | 40 | 0.011374 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | no_gain_floor | 1.246731 |  |  | 40 | 0.011304 |  |
| small-narrow-512x8-regression | small-narrow | regression | 13 | quality_first | 1.246731 |  |  | 40 | 0.011443 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | current_auto | 1.401883 |  |  | 40 | 0.011360 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | manual_default | 1.401883 |  |  | 40 | 0.011368 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | no_gain_floor | 1.401883 |  |  | 40 | 0.011174 |  |
| small-narrow-512x8-regression | small-narrow | regression | 29 | quality_first | 1.401883 |  |  | 40 | 0.011232 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | current_auto | 0.945423 |  |  | 40 | 0.005966 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | manual_default | 0.945423 |  |  | 40 | 0.005450 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | no_gain_floor | 0.945423 |  |  | 40 | 0.005397 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 7 | quality_first | 0.945423 |  |  | 40 | 0.005373 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | current_auto | 1.350555 |  |  | 40 | 0.005490 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | manual_default | 1.350555 |  |  | 40 | 0.005264 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | no_gain_floor | 1.350555 |  |  | 40 | 0.005540 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 13 | quality_first | 1.350555 |  |  | 40 | 0.005319 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | current_auto | 0.965109 |  |  | 40 | 0.006248 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | manual_default | 0.965109 |  |  | 40 | 0.006136 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | no_gain_floor | 0.965109 |  |  | 40 | 0.005824 |  |
| small-narrow-512x8-sparse_regression | small-narrow | sparse_regression | 29 | quality_first | 0.965109 |  |  | 40 | 0.005848 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | current_auto | 0.587465 | 0.742188 |  | 40 | 0.011511 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | manual_default | 0.587465 | 0.742188 |  | 40 | 0.011541 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | no_gain_floor | 0.587465 | 0.742188 |  | 40 | 0.011768 |  |
| small-narrow-512x8-binary | small-narrow | binary | 7 | quality_first | 0.587465 | 0.742188 |  | 40 | 0.011633 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | current_auto | 0.466435 | 0.812500 |  | 40 | 0.011863 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | manual_default | 0.466435 | 0.812500 |  | 40 | 0.012119 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | no_gain_floor | 0.466435 | 0.812500 |  | 40 | 0.011817 |  |
| small-narrow-512x8-binary | small-narrow | binary | 13 | quality_first | 0.466435 | 0.812500 |  | 40 | 0.011953 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | current_auto | 0.538330 | 0.757812 |  | 40 | 0.011521 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | manual_default | 0.538330 | 0.757812 |  | 40 | 0.012393 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | no_gain_floor | 0.538330 | 0.757812 |  | 40 | 0.011945 |  |
| small-narrow-512x8-binary | small-narrow | binary | 29 | quality_first | 0.538330 | 0.757812 |  | 40 | 0.012090 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | current_auto | 1.121017 | 0.539062 |  | 40 | 0.048401 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | manual_default | 1.121017 | 0.539062 |  | 40 | 0.048560 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | no_gain_floor | 1.121017 | 0.539062 |  | 40 | 0.048387 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 7 | quality_first | 1.121017 | 0.539062 |  | 40 | 0.048676 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | current_auto | 1.076570 | 0.562500 |  | 40 | 0.048529 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | manual_default | 1.076570 | 0.562500 |  | 40 | 0.048882 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | no_gain_floor | 1.076570 | 0.562500 |  | 40 | 0.048390 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 13 | quality_first | 1.076570 | 0.562500 |  | 40 | 0.047772 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | current_auto | 1.094530 | 0.617188 |  | 40 | 0.049095 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | manual_default | 1.094530 | 0.617188 |  | 40 | 0.048463 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | no_gain_floor | 1.094530 | 0.617188 |  | 40 | 0.048548 |  |
| small-narrow-512x8-multiclass | small-narrow | multiclass | 29 | quality_first | 1.094530 | 0.617188 |  | 40 | 0.049320 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | current_auto | 0.045750 |  | 0.954250 | 40 | 0.014692 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | manual_default | 0.045750 |  | 0.954250 | 40 | 0.014550 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | no_gain_floor | 0.045750 |  | 0.954250 | 40 | 0.014853 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 7 | quality_first | 0.045750 |  | 0.954250 | 40 | 0.014110 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | current_auto | 0.054510 |  | 0.945490 | 40 | 0.014410 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | manual_default | 0.054510 |  | 0.945490 | 40 | 0.013874 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | no_gain_floor | 0.054510 |  | 0.945490 | 40 | 0.014531 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 13 | quality_first | 0.054510 |  | 0.945490 | 40 | 0.014084 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | current_auto | 0.109613 |  | 0.890387 | 40 | 0.014234 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | manual_default | 0.109613 |  | 0.890387 | 40 | 0.014580 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | no_gain_floor | 0.109613 |  | 0.890387 | 40 | 0.014082 |  |
| small-narrow-512x8-ranking | small-narrow | ranking | 29 | quality_first | 0.109613 |  | 0.890387 | 40 | 0.014260 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | current_auto | 1.106549 |  |  | 40 | 0.020935 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | manual_default | 1.106549 |  |  | 40 | 0.020517 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | no_gain_floor | 1.106549 |  |  | 40 | 0.020463 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 7 | quality_first | 1.106549 |  |  | 40 | 0.020506 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | current_auto | 1.023434 |  |  | 40 | 0.020505 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | manual_default | 1.023434 |  |  | 40 | 0.020927 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | no_gain_floor | 1.023434 |  |  | 40 | 0.020519 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 13 | quality_first | 1.023434 |  |  | 40 | 0.020359 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | current_auto | 0.989188 |  |  | 40 | 0.020721 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | manual_default | 0.989188 |  |  | 40 | 0.020885 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | no_gain_floor | 0.989188 |  |  | 40 | 0.020483 |  |
| small-narrow-1023x16-regression | small-narrow | regression | 29 | quality_first | 0.989188 |  |  | 40 | 0.020361 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | current_auto | 1.088986 |  |  | 40 | 0.012628 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | manual_default | 1.088986 |  |  | 40 | 0.012808 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | no_gain_floor | 1.088986 |  |  | 40 | 0.012861 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 7 | quality_first | 1.088986 |  |  | 40 | 0.012924 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | current_auto | 1.129539 |  |  | 40 | 0.012642 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | manual_default | 1.129539 |  |  | 40 | 0.013019 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | no_gain_floor | 1.129539 |  |  | 40 | 0.013002 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 13 | quality_first | 1.129539 |  |  | 40 | 0.013443 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | current_auto | 1.197693 |  |  | 40 | 0.012936 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | manual_default | 1.197693 |  |  | 40 | 0.012819 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | no_gain_floor | 1.197693 |  |  | 40 | 0.013009 |  |
| small-narrow-1023x16-sparse_regression | small-narrow | sparse_regression | 29 | quality_first | 1.197693 |  |  | 40 | 0.012931 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | current_auto | 0.411863 | 0.815686 |  | 40 | 0.020414 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | manual_default | 0.411863 | 0.815686 |  | 40 | 0.020643 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | no_gain_floor | 0.411863 | 0.815686 |  | 40 | 0.020666 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 7 | quality_first | 0.411863 | 0.815686 |  | 40 | 0.020438 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | current_auto | 0.573400 | 0.713725 |  | 40 | 0.020396 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | manual_default | 0.573400 | 0.713725 |  | 40 | 0.020587 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | no_gain_floor | 0.573400 | 0.713725 |  | 40 | 0.020657 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 13 | quality_first | 0.573400 | 0.713725 |  | 40 | 0.020721 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | current_auto | 0.501474 | 0.760784 |  | 40 | 0.020153 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | manual_default | 0.501474 | 0.760784 |  | 40 | 0.020403 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | no_gain_floor | 0.501474 | 0.760784 |  | 40 | 0.020481 |  |
| small-narrow-1023x16-binary | small-narrow | binary | 29 | quality_first | 0.501474 | 0.760784 |  | 40 | 0.019803 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | current_auto | 0.832246 | 0.670588 |  | 40 | 0.074343 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | manual_default | 0.832246 | 0.670588 |  | 40 | 0.073334 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | no_gain_floor | 0.832246 | 0.670588 |  | 40 | 0.073133 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 7 | quality_first | 0.832246 | 0.670588 |  | 40 | 0.071164 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | current_auto | 1.011430 | 0.596078 |  | 40 | 0.075301 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | manual_default | 1.011430 | 0.596078 |  | 40 | 0.074167 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | no_gain_floor | 1.011430 | 0.596078 |  | 40 | 0.074935 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 13 | quality_first | 1.011430 | 0.596078 |  | 40 | 0.073642 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | current_auto | 0.984423 | 0.623529 |  | 40 | 0.073600 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | manual_default | 0.984423 | 0.623529 |  | 40 | 0.074638 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | no_gain_floor | 0.984423 | 0.623529 |  | 40 | 0.075285 |  |
| small-narrow-1023x16-multiclass | small-narrow | multiclass | 29 | quality_first | 0.984423 | 0.623529 |  | 40 | 0.074906 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | current_auto | 0.020205 |  | 0.979795 | 40 | 0.022445 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | manual_default | 0.020205 |  | 0.979795 | 40 | 0.023502 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | no_gain_floor | 0.020205 |  | 0.979795 | 40 | 0.023237 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 7 | quality_first | 0.020205 |  | 0.979795 | 40 | 0.023428 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | current_auto | 0.022759 |  | 0.977241 | 40 | 0.023353 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | manual_default | 0.022759 |  | 0.977241 | 40 | 0.023285 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | no_gain_floor | 0.022759 |  | 0.977241 | 40 | 0.023544 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 13 | quality_first | 0.022759 |  | 0.977241 | 40 | 0.023285 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | current_auto | 0.055414 |  | 0.944586 | 40 | 0.023580 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | manual_default | 0.055414 |  | 0.944586 | 40 | 0.023331 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | no_gain_floor | 0.055414 |  | 0.944586 | 40 | 0.023525 |  |
| small-narrow-1023x16-ranking | small-narrow | ranking | 29 | quality_first | 0.055414 |  | 0.944586 | 40 | 0.022835 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | current_auto | 1.198654 |  |  | 96 | 0.162836 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | manual_default | 1.198586 |  |  | 300 | 0.459110 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | no_gain_floor | 1.198654 |  |  | 96 | 0.160694 |  |
| small-wide-512x128-regression | small-wide | regression | 7 | quality_first | 1.198654 |  |  | 96 | 0.158644 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | current_auto | 1.394643 |  |  | 96 | 0.158954 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | manual_default | 1.393267 |  |  | 300 | 0.465810 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | no_gain_floor | 1.394643 |  |  | 96 | 0.159694 |  |
| small-wide-512x128-regression | small-wide | regression | 13 | quality_first | 1.394643 |  |  | 96 | 0.159591 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | current_auto | 1.178323 |  |  | 96 | 0.160322 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | manual_default | 1.178138 |  |  | 300 | 0.466528 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | no_gain_floor | 1.178323 |  |  | 96 | 0.160828 |  |
| small-wide-512x128-regression | small-wide | regression | 29 | quality_first | 1.178323 |  |  | 96 | 0.161662 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | current_auto | 1.124911 |  |  | 96 | 0.067192 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | manual_default | 1.193825 |  |  | 300 | 0.202729 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | no_gain_floor | 1.124911 |  |  | 96 | 0.065786 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 7 | quality_first | 1.124911 |  |  | 96 | 0.066617 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | current_auto | 1.785465 |  |  | 96 | 0.069315 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | manual_default | 1.745283 |  |  | 300 | 0.208148 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | no_gain_floor | 1.785465 |  |  | 96 | 0.069409 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 13 | quality_first | 1.785465 |  |  | 96 | 0.068866 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | current_auto | 1.392686 |  |  | 96 | 0.067639 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | manual_default | 1.414088 |  |  | 300 | 0.200406 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | no_gain_floor | 1.392686 |  |  | 96 | 0.066941 |  |
| small-wide-512x128-sparse_regression | small-wide | sparse_regression | 29 | quality_first | 1.392686 |  |  | 96 | 0.065721 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | current_auto | 0.938921 | 0.718750 |  | 300 | 0.472667 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | manual_default | 0.938921 | 0.718750 |  | 300 | 0.463927 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | no_gain_floor | 0.938921 | 0.718750 |  | 300 | 0.467599 |  |
| small-wide-512x128-binary | small-wide | binary | 7 | quality_first | 0.938921 | 0.718750 |  | 300 | 0.469073 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | current_auto | 0.811247 | 0.757812 |  | 300 | 0.445908 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | manual_default | 0.811247 | 0.757812 |  | 300 | 0.444728 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | no_gain_floor | 0.811247 | 0.757812 |  | 300 | 0.447017 |  |
| small-wide-512x128-binary | small-wide | binary | 13 | quality_first | 0.811247 | 0.757812 |  | 300 | 0.446651 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | current_auto | 1.003693 | 0.804688 |  | 300 | 0.487281 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | manual_default | 1.003693 | 0.804688 |  | 300 | 0.489479 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | no_gain_floor | 1.003693 | 0.804688 |  | 300 | 0.485660 |  |
| small-wide-512x128-binary | small-wide | binary | 29 | quality_first | 1.003693 | 0.804688 |  | 300 | 0.487596 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | current_auto | 1.508700 | 0.554688 |  | 96 | 0.588814 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | manual_default | 2.393751 | 0.585938 |  | 300 | 1.802574 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | no_gain_floor | 1.508700 | 0.554688 |  | 96 | 0.588609 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 7 | quality_first | 1.508700 | 0.554688 |  | 96 | 0.588821 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | current_auto | 1.720285 | 0.570312 |  | 96 | 0.602582 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | manual_default | 2.595189 | 0.562500 |  | 300 | 1.820958 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | no_gain_floor | 1.720285 | 0.570312 |  | 96 | 0.607603 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 13 | quality_first | 1.720285 | 0.570312 |  | 96 | 0.602902 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | current_auto | 1.518329 | 0.601562 |  | 96 | 0.606234 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | manual_default | 2.064210 | 0.593750 |  | 300 | 1.816660 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | no_gain_floor | 1.518329 | 0.601562 |  | 96 | 0.605704 |  |
| small-wide-512x128-multiclass | small-wide | multiclass | 29 | quality_first | 1.518329 | 0.601562 |  | 96 | 0.598769 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | current_auto | 0.041628 |  | 0.958372 | 96 | 0.170543 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | manual_default | 0.036371 |  | 0.963629 | 300 | 0.532040 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | no_gain_floor | 0.041628 |  | 0.958372 | 96 | 0.172800 |  |
| small-wide-512x128-ranking | small-wide | ranking | 7 | quality_first | 0.041628 |  | 0.958372 | 96 | 0.170233 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | current_auto | 0.044087 |  | 0.955913 | 96 | 0.170143 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | manual_default | 0.041659 |  | 0.958341 | 300 | 0.534918 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | no_gain_floor | 0.044087 |  | 0.955913 | 96 | 0.171162 |  |
| small-wide-512x128-ranking | small-wide | ranking | 13 | quality_first | 0.044087 |  | 0.955913 | 96 | 0.172375 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | current_auto | 0.028972 |  | 0.971028 | 96 | 0.171265 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | manual_default | 0.020522 |  | 0.979478 | 300 | 0.534470 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | no_gain_floor | 0.028972 |  | 0.971028 | 96 | 0.169590 |  |
| small-wide-512x128-ranking | small-wide | ranking | 29 | quality_first | 0.028972 |  | 0.971028 | 96 | 0.170171 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | current_auto | 1.032054 |  |  | 96 | 0.279467 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | manual_default | 1.033443 |  |  | 300 | 0.810276 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | no_gain_floor | 1.032054 |  |  | 96 | 0.277907 |  |
| small-wide-1023x256-regression | small-wide | regression | 7 | quality_first | 1.032054 |  |  | 96 | 0.279103 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | current_auto | 1.083313 |  |  | 96 | 0.285334 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | manual_default | 1.078950 |  |  | 300 | 0.849594 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | no_gain_floor | 1.083313 |  |  | 96 | 0.283954 |  |
| small-wide-1023x256-regression | small-wide | regression | 13 | quality_first | 1.083313 |  |  | 96 | 0.284875 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | current_auto | 1.109971 |  |  | 96 | 0.283456 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | manual_default | 1.115278 |  |  | 300 | 0.838635 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | no_gain_floor | 1.109971 |  |  | 96 | 0.283582 |  |
| small-wide-1023x256-regression | small-wide | regression | 29 | quality_first | 1.109971 |  |  | 96 | 0.284017 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | current_auto | 0.955110 |  |  | 96 | 0.090133 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | manual_default | 0.988537 |  |  | 300 | 0.271381 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | no_gain_floor | 0.955110 |  |  | 96 | 0.088741 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 7 | quality_first | 0.955110 |  |  | 96 | 0.089952 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | current_auto | 1.020762 |  |  | 96 | 0.090080 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | manual_default | 1.047432 |  |  | 300 | 0.268018 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | no_gain_floor | 1.020762 |  |  | 96 | 0.091314 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 13 | quality_first | 1.020762 |  |  | 96 | 0.091286 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | current_auto | 1.243298 |  |  | 96 | 0.091119 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | manual_default | 1.272168 |  |  | 300 | 0.275574 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | no_gain_floor | 1.243298 |  |  | 96 | 0.091528 |  |
| small-wide-1023x256-sparse_regression | small-wide | sparse_regression | 29 | quality_first | 1.243298 |  |  | 96 | 0.091067 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | current_auto | 1.115527 | 0.713725 |  | 300 | 0.869577 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | manual_default | 1.115527 | 0.713725 |  | 300 | 0.899125 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | no_gain_floor | 1.115527 | 0.713725 |  | 300 | 0.861612 |  |
| small-wide-1023x256-binary | small-wide | binary | 7 | quality_first | 1.115527 | 0.713725 |  | 300 | 0.867327 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | current_auto | 0.863139 | 0.737255 |  | 300 | 0.867424 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | manual_default | 0.863139 | 0.737255 |  | 300 | 0.868351 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | no_gain_floor | 0.863139 | 0.737255 |  | 300 | 0.869291 |  |
| small-wide-1023x256-binary | small-wide | binary | 13 | quality_first | 0.863139 | 0.737255 |  | 300 | 0.869294 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | current_auto | 0.730849 | 0.725490 |  | 300 | 0.855364 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | manual_default | 0.730849 | 0.725490 |  | 300 | 0.855847 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | no_gain_floor | 0.730849 | 0.725490 |  | 300 | 0.854523 |  |
| small-wide-1023x256-binary | small-wide | binary | 29 | quality_first | 0.730849 | 0.725490 |  | 300 | 0.857057 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | current_auto | 1.240497 | 0.592157 |  | 96 | 1.100531 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | manual_default | 2.091386 | 0.588235 |  | 300 | 3.385035 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | no_gain_floor | 1.240497 | 0.592157 |  | 96 | 1.106245 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 7 | quality_first | 1.240497 | 0.592157 |  | 96 | 1.135309 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | current_auto | 1.235065 | 0.588235 |  | 96 | 1.143524 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | manual_default | 2.177473 | 0.592157 |  | 300 | 3.470898 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | no_gain_floor | 1.235065 | 0.588235 |  | 96 | 1.131365 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 13 | quality_first | 1.235065 | 0.588235 |  | 96 | 1.128094 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | current_auto | 1.214883 | 0.568627 |  | 96 | 1.101777 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | manual_default | 2.209509 | 0.568627 |  | 300 | 3.388420 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | no_gain_floor | 1.214883 | 0.568627 |  | 96 | 1.097808 |  |
| small-wide-1023x256-multiclass | small-wide | multiclass | 29 | quality_first | 1.214883 | 0.568627 |  | 96 | 1.098669 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | current_auto | 0.046265 |  | 0.953735 | 96 | 0.307706 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | manual_default | 0.038047 |  | 0.961953 | 300 | 0.969020 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | no_gain_floor | 0.046265 |  | 0.953735 | 96 | 0.306867 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 7 | quality_first | 0.046265 |  | 0.953735 | 96 | 0.313678 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | current_auto | 0.045150 |  | 0.954850 | 96 | 0.313098 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | manual_default | 0.034720 |  | 0.965280 | 300 | 0.981643 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | no_gain_floor | 0.045150 |  | 0.954850 | 96 | 0.308759 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 13 | quality_first | 0.045150 |  | 0.954850 | 96 | 0.307973 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | current_auto | 0.040948 |  | 0.959052 | 96 | 0.326206 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | manual_default | 0.039767 |  | 0.960233 | 300 | 0.974976 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | no_gain_floor | 0.040948 |  | 0.959052 | 96 | 0.303136 |  |
| small-wide-1023x256-ranking | small-wide | ranking | 29 | quality_first | 0.040948 |  | 0.959052 | 96 | 0.306047 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | current_auto | 0.936773 |  |  | 40 | 0.026260 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | manual_default | 0.959450 |  |  | 40 | 0.026689 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | no_gain_floor | 0.936773 |  |  | 40 | 0.026466 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 7 | quality_first | 0.954618 |  |  | 40 | 0.026533 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | current_auto | 0.925707 |  |  | 40 | 0.026723 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | manual_default | 0.962787 |  |  | 40 | 0.026266 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | no_gain_floor | 0.925707 |  |  | 40 | 0.026611 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 13 | quality_first | 0.955503 |  |  | 40 | 0.025888 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | current_auto | 0.991331 |  |  | 40 | 0.026528 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | manual_default | 1.010573 |  |  | 40 | 0.026669 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | no_gain_floor | 0.991331 |  |  | 40 | 0.025499 |  |
| medium-narrow-2048x16-regression | medium-narrow | regression | 29 | quality_first | 1.009384 |  |  | 40 | 0.026226 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | current_auto | 1.165999 |  |  | 40 | 0.017535 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | manual_default | 1.143000 |  |  | 40 | 0.017980 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | no_gain_floor | 1.165999 |  |  | 40 | 0.017432 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 7 | quality_first | 1.209558 |  |  | 40 | 0.017542 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | current_auto | 1.168801 |  |  | 40 | 0.016715 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | manual_default | 1.161288 |  |  | 40 | 0.017071 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | no_gain_floor | 1.168801 |  |  | 40 | 0.017341 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 13 | quality_first | 1.156744 |  |  | 40 | 0.017584 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | current_auto | 1.213547 |  |  | 40 | 0.017320 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | manual_default | 1.252843 |  |  | 40 | 0.017783 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | no_gain_floor | 1.213547 |  |  | 40 | 0.016765 |  |
| medium-narrow-2048x16-sparse_regression | medium-narrow | sparse_regression | 29 | quality_first | 1.183996 |  |  | 40 | 0.017851 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | current_auto | 0.470797 | 0.767578 |  | 40 | 0.025946 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | manual_default | 0.470915 | 0.753906 |  | 40 | 0.026240 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | no_gain_floor | 0.470797 | 0.767578 |  | 40 | 0.025838 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 7 | quality_first | 0.467156 | 0.769531 |  | 40 | 0.025827 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | current_auto | 0.488263 | 0.746094 |  | 40 | 0.026029 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | manual_default | 0.499859 | 0.740234 |  | 40 | 0.026367 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | no_gain_floor | 0.488263 | 0.746094 |  | 40 | 0.026535 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 13 | quality_first | 0.491261 | 0.748047 |  | 40 | 0.025907 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | current_auto | 0.487371 | 0.744141 |  | 40 | 0.026551 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | manual_default | 0.487910 | 0.757812 |  | 40 | 0.025781 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | no_gain_floor | 0.487371 | 0.744141 |  | 40 | 0.026436 |  |
| medium-narrow-2048x16-binary | medium-narrow | binary | 29 | quality_first | 0.488356 | 0.750000 |  | 40 | 0.025476 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | current_auto | 0.888976 | 0.617188 |  | 40 | 0.099254 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | manual_default | 0.916457 | 0.621094 |  | 40 | 0.102473 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | no_gain_floor | 0.888976 | 0.617188 |  | 40 | 0.097203 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 7 | quality_first | 0.891970 | 0.623047 |  | 40 | 0.101305 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | current_auto | 0.837018 | 0.652344 |  | 40 | 0.097506 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | manual_default | 0.833373 | 0.658203 |  | 40 | 0.099546 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | no_gain_floor | 0.837018 | 0.652344 |  | 40 | 0.098257 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 13 | quality_first | 0.830199 | 0.656250 |  | 40 | 0.098602 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | current_auto | 0.952523 | 0.591797 |  | 40 | 0.098487 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | manual_default | 0.983401 | 0.566406 |  | 40 | 0.103303 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | no_gain_floor | 0.952523 | 0.591797 |  | 40 | 0.099676 |  |
| medium-narrow-2048x16-multiclass | medium-narrow | multiclass | 29 | quality_first | 0.982138 | 0.568359 |  | 40 | 0.100140 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | current_auto | 0.056037 |  | 0.943963 | 40 | 0.032622 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | manual_default | 0.049562 |  | 0.950438 | 40 | 0.031427 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | no_gain_floor | 0.056037 |  | 0.943963 | 40 | 0.032021 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 7 | quality_first | 0.055997 |  | 0.944003 | 40 | 0.032714 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | current_auto | 0.054065 |  | 0.945935 | 40 | 0.032151 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | manual_default | 0.061249 |  | 0.938751 | 40 | 0.031616 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | no_gain_floor | 0.054065 |  | 0.945935 | 40 | 0.032660 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 13 | quality_first | 0.059670 |  | 0.940330 | 40 | 0.032510 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | current_auto | 0.043007 |  | 0.956993 | 40 | 0.032235 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | manual_default | 0.059333 |  | 0.940667 | 40 | 0.031925 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | no_gain_floor | 0.043007 |  | 0.956993 | 40 | 0.032497 |  |
| medium-narrow-2048x16-ranking | medium-narrow | ranking | 29 | quality_first | 0.044501 |  | 0.955499 | 40 | 0.032167 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | current_auto | 0.907927 |  |  | 40 | 0.055857 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | manual_default | 0.910924 |  |  | 40 | 0.054322 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | no_gain_floor | 0.907927 |  |  | 40 | 0.055068 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 7 | quality_first | 0.910832 |  |  | 40 | 0.053606 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | current_auto | 0.950398 |  |  | 40 | 0.053979 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | manual_default | 0.956071 |  |  | 40 | 0.053639 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | no_gain_floor | 0.950398 |  |  | 40 | 0.054100 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 13 | quality_first | 0.951897 |  |  | 40 | 0.053429 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | current_auto | 0.928650 |  |  | 40 | 0.055393 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | manual_default | 0.939883 |  |  | 40 | 0.053814 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | no_gain_floor | 0.928650 |  |  | 40 | 0.054696 |  |
| medium-narrow-8192x16-regression | medium-narrow | regression | 29 | quality_first | 0.932671 |  |  | 40 | 0.054787 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | current_auto | 1.070219 |  |  | 40 | 0.045459 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | manual_default | 1.092623 |  |  | 40 | 0.044933 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | no_gain_floor | 1.070219 |  |  | 40 | 0.045320 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 7 | quality_first | 1.066447 |  |  | 40 | 0.044888 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | current_auto | 1.121034 |  |  | 40 | 0.045415 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | manual_default | 1.125220 |  |  | 40 | 0.044779 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | no_gain_floor | 1.121034 |  |  | 40 | 0.045227 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 13 | quality_first | 1.132881 |  |  | 40 | 0.044452 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | current_auto | 0.976190 |  |  | 40 | 0.045615 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | manual_default | 0.964284 |  |  | 40 | 0.045363 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | no_gain_floor | 0.976190 |  |  | 40 | 0.044737 |  |
| medium-narrow-8192x16-sparse_regression | medium-narrow | sparse_regression | 29 | quality_first | 0.971045 |  |  | 40 | 0.044824 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | current_auto | 0.438248 | 0.794922 |  | 40 | 0.058416 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | manual_default | 0.440579 | 0.803711 |  | 40 | 0.055634 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | no_gain_floor | 0.438248 | 0.794922 |  | 40 | 0.058017 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 7 | quality_first | 0.440884 | 0.791992 |  | 40 | 0.056149 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | current_auto | 0.448574 | 0.783203 |  | 40 | 0.057729 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | manual_default | 0.446551 | 0.790039 |  | 40 | 0.055808 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | no_gain_floor | 0.448574 | 0.783203 |  | 40 | 0.057786 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 13 | quality_first | 0.448503 | 0.777344 |  | 40 | 0.055607 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | current_auto | 0.467210 | 0.782227 |  | 40 | 0.057438 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | manual_default | 0.468290 | 0.773438 |  | 40 | 0.056552 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | no_gain_floor | 0.467210 | 0.782227 |  | 40 | 0.056361 |  |
| medium-narrow-8192x16-binary | medium-narrow | binary | 29 | quality_first | 0.469317 | 0.775391 |  | 40 | 0.055094 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | current_auto | 0.873238 | 0.636719 |  | 40 | 0.215105 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | manual_default | 0.873152 | 0.643555 |  | 40 | 0.219806 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | no_gain_floor | 0.873240 | 0.636719 |  | 40 | 0.215576 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 7 | quality_first | 0.875250 | 0.634766 |  | 40 | 0.218475 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | current_auto | 0.883946 | 0.636719 |  | 40 | 0.215723 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | manual_default | 0.884437 | 0.643555 |  | 40 | 0.219493 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | no_gain_floor | 0.883945 | 0.636719 |  | 40 | 0.215142 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 13 | quality_first | 0.885673 | 0.643555 |  | 40 | 0.217914 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | current_auto | 0.871736 | 0.652344 |  | 40 | 0.214830 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | manual_default | 0.879004 | 0.650391 |  | 40 | 0.220332 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | no_gain_floor | 0.871737 | 0.652344 |  | 40 | 0.214954 |  |
| medium-narrow-8192x16-multiclass | medium-narrow | multiclass | 29 | quality_first | 0.876522 | 0.651367 |  | 40 | 0.216695 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | current_auto | 0.036837 |  | 0.963163 | 40 | 0.070733 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | manual_default | 0.036345 |  | 0.963655 | 40 | 0.069733 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | no_gain_floor | 0.036837 |  | 0.963163 | 40 | 0.070991 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 7 | quality_first | 0.038254 |  | 0.961746 | 40 | 0.070696 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | current_auto | 0.040332 |  | 0.959668 | 40 | 0.075005 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | manual_default | 0.047581 |  | 0.952419 | 40 | 0.071919 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | no_gain_floor | 0.040332 |  | 0.959668 | 40 | 0.072751 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 13 | quality_first | 0.040932 |  | 0.959068 | 40 | 0.072048 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | current_auto | 0.032483 |  | 0.967517 | 40 | 0.072043 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | manual_default | 0.029343 |  | 0.970657 | 40 | 0.071064 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | no_gain_floor | 0.032483 |  | 0.967517 | 40 | 0.072786 |  |
| medium-narrow-8192x16-ranking | medium-narrow | ranking | 29 | quality_first | 0.026819 |  | 0.973181 | 40 | 0.071151 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | current_auto | 1.139516 |  |  | 40 | 0.067410 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | manual_default | 1.001547 |  |  | 40 | 0.083979 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | no_gain_floor | 1.139516 |  |  | 40 | 0.067287 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 7 | quality_first | 0.991976 |  |  | 40 | 0.083158 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | current_auto | 1.258807 |  |  | 40 | 0.067933 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | manual_default | 1.088536 |  |  | 40 | 0.082368 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | no_gain_floor | 1.258807 |  |  | 40 | 0.066948 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 13 | quality_first | 1.094273 |  |  | 40 | 0.081969 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | current_auto | 1.330950 |  |  | 40 | 0.069844 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | manual_default | 1.094226 |  |  | 40 | 0.083387 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | no_gain_floor | 1.330950 |  |  | 40 | 0.069409 |  |
| medium-wide-2048x128-regression | medium-wide | regression | 29 | quality_first | 1.085990 |  |  | 40 | 0.083337 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | current_auto | 0.967074 |  |  | 40 | 0.029903 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | manual_default | 1.006154 |  |  | 40 | 0.035450 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | no_gain_floor | 0.967074 |  |  | 40 | 0.029714 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 7 | quality_first | 0.985487 |  |  | 40 | 0.033450 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | current_auto | 0.965502 |  |  | 40 | 0.030338 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | manual_default | 1.038211 |  |  | 40 | 0.034809 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | no_gain_floor | 0.965502 |  |  | 40 | 0.029804 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 13 | quality_first | 0.980549 |  |  | 40 | 0.033697 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | current_auto | 1.009232 |  |  | 40 | 0.029954 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | manual_default | 0.978235 |  |  | 40 | 0.034140 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | no_gain_floor | 1.009232 |  |  | 40 | 0.028936 |  |
| medium-wide-2048x128-sparse_regression | medium-wide | sparse_regression | 29 | quality_first | 0.893986 |  |  | 40 | 0.032704 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | current_auto | 0.479360 | 0.769531 |  | 40 | 0.069888 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | manual_default | 0.474076 | 0.769531 |  | 40 | 0.086230 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | no_gain_floor | 0.479360 | 0.769531 |  | 40 | 0.068823 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 7 | quality_first | 0.465135 | 0.775391 |  | 40 | 0.082215 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | current_auto | 0.467152 | 0.781250 |  | 40 | 0.067530 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | manual_default | 0.456495 | 0.791016 |  | 40 | 0.084326 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | no_gain_floor | 0.467152 | 0.781250 |  | 40 | 0.067339 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 13 | quality_first | 0.456160 | 0.800781 |  | 40 | 0.082912 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | current_auto | 0.462049 | 0.787109 |  | 40 | 0.068410 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | manual_default | 0.440985 | 0.802734 |  | 40 | 0.083149 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | no_gain_floor | 0.462049 | 0.787109 |  | 40 | 0.067442 |  |
| medium-wide-2048x128-binary | medium-wide | binary | 29 | quality_first | 0.434184 | 0.802734 |  | 40 | 0.083400 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | current_auto | 0.906088 | 0.613281 |  | 40 | 0.253584 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | manual_default | 0.925677 | 0.609375 |  | 40 | 0.322206 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | no_gain_floor | 0.906435 | 0.613281 |  | 40 | 0.252490 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 7 | quality_first | 0.915058 | 0.623047 |  | 40 | 0.317687 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | current_auto | 0.932231 | 0.619141 |  | 40 | 0.252387 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | manual_default | 0.943052 | 0.601562 |  | 40 | 0.323335 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | no_gain_floor | 0.931674 | 0.621094 |  | 40 | 0.254798 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 13 | quality_first | 0.928870 | 0.605469 |  | 40 | 0.316441 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | current_auto | 0.903675 | 0.650391 |  | 40 | 0.252782 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | manual_default | 0.901041 | 0.634766 |  | 40 | 0.322563 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | no_gain_floor | 0.903674 | 0.650391 |  | 40 | 0.254703 |  |
| medium-wide-2048x128-multiclass | medium-wide | multiclass | 29 | quality_first | 0.878114 | 0.654297 |  | 40 | 0.308654 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | current_auto | 0.052335 |  | 0.947665 | 40 | 0.073922 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | manual_default | 0.040324 |  | 0.959676 | 40 | 0.088956 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | no_gain_floor | 0.052335 |  | 0.947665 | 40 | 0.074820 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 7 | quality_first | 0.043374 |  | 0.956626 | 40 | 0.089655 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | current_auto | 0.054497 |  | 0.945503 | 40 | 0.074673 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | manual_default | 0.054565 |  | 0.945435 | 40 | 0.089569 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | no_gain_floor | 0.054497 |  | 0.945503 | 40 | 0.073202 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 13 | quality_first | 0.055063 |  | 0.944937 | 40 | 0.088989 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | current_auto | 0.054830 |  | 0.945170 | 40 | 0.074195 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | manual_default | 0.044983 |  | 0.955017 | 40 | 0.110482 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | no_gain_floor | 0.054830 |  | 0.945170 | 40 | 0.076230 |  |
| medium-wide-2048x128-ranking | medium-wide | ranking | 29 | quality_first | 0.043895 |  | 0.956105 | 40 | 0.089316 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | current_auto | 1.404662 |  |  | 40 | 0.141571 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | manual_default | 0.988130 |  |  | 40 | 0.206178 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | no_gain_floor | 1.404662 |  |  | 40 | 0.137225 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 7 | quality_first | 0.992830 |  |  | 40 | 0.206364 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | current_auto | 1.397172 |  |  | 40 | 0.144387 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | manual_default | 0.973534 |  |  | 40 | 0.205994 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | no_gain_floor | 1.397172 |  |  | 40 | 0.140920 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 13 | quality_first | 0.974189 |  |  | 40 | 0.224675 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | current_auto | 1.319181 |  |  | 40 | 0.160814 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | manual_default | 0.928280 |  |  | 40 | 0.226002 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | no_gain_floor | 1.319181 |  |  | 40 | 0.154790 |  |
| medium-wide-8192x256-regression | medium-wide | regression | 29 | quality_first | 0.920007 |  |  | 40 | 0.209807 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | current_auto | 0.975513 |  |  | 40 | 0.076737 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | manual_default | 0.917171 |  |  | 40 | 0.107510 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | no_gain_floor | 0.975513 |  |  | 40 | 0.070300 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 7 | quality_first | 0.934531 |  |  | 40 | 0.104396 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | current_auto | 1.018249 |  |  | 40 | 0.083526 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | manual_default | 0.957270 |  |  | 40 | 0.117448 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | no_gain_floor | 1.018249 |  |  | 40 | 0.075159 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 13 | quality_first | 0.920485 |  |  | 40 | 0.107919 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | current_auto | 1.049444 |  |  | 40 | 0.083220 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | manual_default | 0.950134 |  |  | 40 | 0.113772 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | no_gain_floor | 1.049444 |  |  | 40 | 0.074982 |  |
| medium-wide-8192x256-sparse_regression | medium-wide | sparse_regression | 29 | quality_first | 0.953436 |  |  | 40 | 0.107770 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | current_auto | 0.477624 | 0.765625 |  | 40 | 0.144298 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | manual_default | 0.454305 | 0.772461 |  | 40 | 0.205237 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | no_gain_floor | 0.477624 | 0.765625 |  | 40 | 0.138386 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 7 | quality_first | 0.455709 | 0.785156 |  | 40 | 0.208986 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | current_auto | 0.487223 | 0.770508 |  | 40 | 0.144607 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | manual_default | 0.463858 | 0.785156 |  | 40 | 0.207614 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | no_gain_floor | 0.487223 | 0.770508 |  | 40 | 0.136933 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 13 | quality_first | 0.463091 | 0.782227 |  | 40 | 0.207240 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | current_auto | 0.501657 | 0.762695 |  | 40 | 0.142873 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | manual_default | 0.482422 | 0.762695 |  | 40 | 0.209073 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | no_gain_floor | 0.501657 | 0.762695 |  | 40 | 0.137765 |  |
| medium-wide-8192x256-binary | medium-wide | binary | 29 | quality_first | 0.482142 | 0.759766 |  | 40 | 0.206859 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | current_auto | 0.914937 | 0.622070 |  | 40 | 0.595706 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | manual_default | 0.900804 | 0.625000 |  | 40 | 0.814958 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | no_gain_floor | 0.914937 | 0.622070 |  | 40 | 0.503038 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 7 | quality_first | 0.901093 | 0.630859 |  | 40 | 0.796668 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | current_auto | 0.902941 | 0.642578 |  | 40 | 0.503473 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | manual_default | 0.884845 | 0.639648 |  | 40 | 0.803998 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | no_gain_floor | 0.902941 | 0.642578 |  | 40 | 0.502429 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 13 | quality_first | 0.877802 | 0.645508 |  | 40 | 0.784272 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | current_auto | 0.932150 | 0.608398 |  | 40 | 0.518075 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | manual_default | 0.925418 | 0.602539 |  | 40 | 0.806971 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | no_gain_floor | 0.932150 | 0.608398 |  | 40 | 0.517192 |  |
| medium-wide-8192x256-multiclass | medium-wide | multiclass | 29 | quality_first | 0.924685 | 0.612305 |  | 40 | 0.781142 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | current_auto | 0.038051 |  | 0.961949 | 40 | 0.161770 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | manual_default | 0.030179 |  | 0.969821 | 40 | 0.224405 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | no_gain_floor | 0.038051 |  | 0.961949 | 40 | 0.156310 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 7 | quality_first | 0.028134 |  | 0.971866 | 40 | 0.239818 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | current_auto | 0.046119 |  | 0.953881 | 40 | 0.157606 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | manual_default | 0.045575 |  | 0.954425 | 40 | 0.226151 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | no_gain_floor | 0.046119 |  | 0.953881 | 40 | 0.152639 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 13 | quality_first | 0.044349 |  | 0.955651 | 40 | 0.223945 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | current_auto | 0.041931 |  | 0.958069 | 40 | 0.160389 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | manual_default | 0.031306 |  | 0.968694 | 40 | 0.224446 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | no_gain_floor | 0.041931 |  | 0.958069 | 40 | 0.160622 |  |
| medium-wide-8192x256-ranking | medium-wide | ranking | 29 | quality_first | 0.034885 |  | 0.965115 | 40 | 0.228408 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | current_auto | 0.911667 |  |  | 40 | 0.092374 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | manual_default | 0.899458 |  |  | 40 | 0.087501 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | no_gain_floor | 0.911667 |  |  | 40 | 0.087265 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 7 | quality_first | 0.903352 |  |  | 40 | 0.087958 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | current_auto | 0.888338 |  |  | 40 | 0.087457 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | manual_default | 0.907969 |  |  | 40 | 0.086757 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | no_gain_floor | 0.888338 |  |  | 40 | 0.086777 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 13 | quality_first | 0.902927 |  |  | 40 | 0.087588 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | current_auto | 0.926783 |  |  | 40 | 0.088148 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | manual_default | 0.938812 |  |  | 40 | 0.086988 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | no_gain_floor | 0.926783 |  |  | 40 | 0.087814 |  |
| large-narrow-16384x16-regression | large-narrow | regression | 29 | quality_first | 0.934912 |  |  | 40 | 0.087842 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | current_auto | 1.020661 |  |  | 40 | 0.078445 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | manual_default | 1.015931 |  |  | 40 | 0.083500 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | no_gain_floor | 1.020661 |  |  | 40 | 0.077657 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 7 | quality_first | 1.015392 |  |  | 40 | 0.083014 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | current_auto | 0.955216 |  |  | 40 | 0.078193 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | manual_default | 0.928215 |  |  | 40 | 0.083067 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | no_gain_floor | 0.955216 |  |  | 40 | 0.076767 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 13 | quality_first | 0.944869 |  |  | 40 | 0.081940 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | current_auto | 1.061646 |  |  | 40 | 0.078655 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | manual_default | 1.005228 |  |  | 40 | 0.082989 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | no_gain_floor | 1.061646 |  |  | 40 | 0.077632 |  |
| large-narrow-16384x16-sparse_regression | large-narrow | sparse_regression | 29 | quality_first | 1.064457 |  |  | 40 | 0.082551 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | current_auto | 0.442674 | 0.795898 |  | 40 | 0.094750 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | manual_default | 0.440494 | 0.800781 |  | 40 | 0.095375 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | no_gain_floor | 0.442674 | 0.795898 |  | 40 | 0.093776 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 7 | quality_first | 0.443080 | 0.799805 |  | 40 | 0.095546 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | current_auto | 0.450509 | 0.787109 |  | 40 | 0.095725 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | manual_default | 0.447729 | 0.788086 |  | 40 | 0.095675 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | no_gain_floor | 0.450509 | 0.787109 |  | 40 | 0.094736 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 13 | quality_first | 0.450698 | 0.784180 |  | 40 | 0.094761 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | current_auto | 0.449667 | 0.777344 |  | 40 | 0.094844 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | manual_default | 0.451824 | 0.778320 |  | 40 | 0.094515 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | no_gain_floor | 0.449667 | 0.777344 |  | 40 | 0.094307 |  |
| large-narrow-16384x16-binary | large-narrow | binary | 29 | quality_first | 0.452321 | 0.782227 |  | 40 | 0.094876 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | current_auto | 0.844281 | 0.646484 |  | 40 | 0.353491 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | manual_default | 0.843943 | 0.653320 |  | 40 | 0.375272 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | no_gain_floor | 0.844281 | 0.646484 |  | 40 | 0.352215 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 7 | quality_first | 0.840072 | 0.654297 |  | 40 | 0.373236 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | current_auto | 0.847900 | 0.631836 |  | 40 | 0.354883 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | manual_default | 0.848186 | 0.639648 |  | 40 | 0.373484 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | no_gain_floor | 0.847900 | 0.631836 |  | 40 | 0.352771 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 13 | quality_first | 0.842316 | 0.638672 |  | 40 | 0.373733 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | current_auto | 0.855436 | 0.656250 |  | 40 | 0.359609 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | manual_default | 0.854029 | 0.657227 |  | 40 | 0.369966 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | no_gain_floor | 0.855436 | 0.656250 |  | 40 | 0.355995 |  |
| large-narrow-16384x16-multiclass | large-narrow | multiclass | 29 | quality_first | 0.853687 | 0.659180 |  | 40 | 0.370028 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | current_auto | 0.028092 |  | 0.971908 | 40 | 0.118708 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | manual_default | 0.026607 |  | 0.973393 | 40 | 0.117949 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | no_gain_floor | 0.028092 |  | 0.971908 | 40 | 0.116789 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 7 | quality_first | 0.027389 |  | 0.972611 | 40 | 0.121769 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | current_auto | 0.030977 |  | 0.969023 | 40 | 0.122241 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | manual_default | 0.030402 |  | 0.969598 | 40 | 0.121952 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | no_gain_floor | 0.030977 |  | 0.969023 | 40 | 0.121841 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 13 | quality_first | 0.032129 |  | 0.967871 | 40 | 0.121967 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | current_auto | 0.028123 |  | 0.971877 | 40 | 0.121952 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | manual_default | 0.029712 |  | 0.970288 | 40 | 0.120518 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | no_gain_floor | 0.028123 |  | 0.971877 | 40 | 0.120517 |  |
| large-narrow-16384x16-ranking | large-narrow | ranking | 29 | quality_first | 0.029030 |  | 0.970970 | 40 | 0.121141 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | current_auto | 1.254501 |  |  | 40 | 0.209307 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | manual_default | 0.876513 |  |  | 40 | 0.316706 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | no_gain_floor | 1.254501 |  |  | 40 | 0.198876 |  |
| large-wide-16384x256-regression | large-wide | regression | 7 | quality_first | 0.884073 |  |  | 40 | 0.313892 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | current_auto | 1.367765 |  |  | 40 | 0.210742 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | manual_default | 0.893836 |  |  | 40 | 0.313605 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | no_gain_floor | 1.367765 |  |  | 40 | 0.199191 |  |
| large-wide-16384x256-regression | large-wide | regression | 13 | quality_first | 0.897455 |  |  | 40 | 0.321524 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | current_auto | 1.206074 |  |  | 40 | 0.213717 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | manual_default | 0.834827 |  |  | 40 | 0.320029 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | no_gain_floor | 1.206074 |  |  | 40 | 0.202381 |  |
| large-wide-16384x256-regression | large-wide | regression | 29 | quality_first | 0.832478 |  |  | 40 | 0.319434 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | current_auto | 1.123433 |  |  | 40 | 0.129320 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | manual_default | 0.984147 |  |  | 40 | 0.191589 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | no_gain_floor | 1.123433 |  |  | 40 | 0.111135 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 7 | quality_first | 1.003638 |  |  | 40 | 0.184271 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | current_auto | 1.064186 |  |  | 40 | 0.130232 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | manual_default | 0.953405 |  |  | 40 | 0.195085 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | no_gain_floor | 1.064186 |  |  | 40 | 0.110588 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 13 | quality_first | 1.022102 |  |  | 40 | 0.183521 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | current_auto | 1.118783 |  |  | 40 | 0.121062 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | manual_default | 0.975560 |  |  | 40 | 0.186727 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | no_gain_floor | 1.118783 |  |  | 40 | 0.111086 |  |
| large-wide-16384x256-sparse_regression | large-wide | sparse_regression | 29 | quality_first | 1.014636 |  |  | 40 | 0.186221 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | current_auto | 0.487697 | 0.765625 |  | 40 | 0.217169 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | manual_default | 0.467187 | 0.772461 |  | 40 | 0.326369 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | no_gain_floor | 0.487697 | 0.765625 |  | 40 | 0.206041 |  |
| large-wide-16384x256-binary | large-wide | binary | 7 | quality_first | 0.467222 | 0.774414 |  | 40 | 0.326032 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | current_auto | 0.489207 | 0.773438 |  | 40 | 0.225656 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | manual_default | 0.465047 | 0.777344 |  | 40 | 0.333245 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | no_gain_floor | 0.489207 | 0.773438 |  | 40 | 0.204474 |  |
| large-wide-16384x256-binary | large-wide | binary | 13 | quality_first | 0.466821 | 0.775391 |  | 40 | 0.325380 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | current_auto | 0.502915 | 0.763672 |  | 40 | 0.216343 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | manual_default | 0.472571 | 0.777344 |  | 40 | 0.326022 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | no_gain_floor | 0.502915 | 0.763672 |  | 40 | 0.203329 |  |
| large-wide-16384x256-binary | large-wide | binary | 29 | quality_first | 0.471856 | 0.768555 |  | 40 | 0.333447 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | current_auto | 0.883432 | 0.633789 |  | 40 | 0.745265 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | manual_default | 0.878500 | 0.627930 |  | 40 | 1.242145 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | no_gain_floor | 0.883432 | 0.633789 |  | 40 | 0.733378 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 7 | quality_first | 0.875623 | 0.624023 |  | 40 | 1.234555 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | current_auto | 0.902899 | 0.623047 |  | 40 | 0.757881 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | manual_default | 0.880031 | 0.627930 |  | 40 | 1.233165 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | no_gain_floor | 0.902899 | 0.623047 |  | 40 | 0.730996 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 13 | quality_first | 0.879740 | 0.618164 |  | 40 | 1.232953 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | current_auto | 0.899215 | 0.644531 |  | 40 | 0.741154 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | manual_default | 0.878460 | 0.635742 |  | 40 | 1.215425 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | no_gain_floor | 0.899215 | 0.644531 |  | 40 | 0.727108 |  |
| large-wide-16384x256-multiclass | large-wide | multiclass | 29 | quality_first | 0.880410 | 0.637695 |  | 40 | 1.213544 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | current_auto | 0.043758 |  | 0.956242 | 40 | 0.249241 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | manual_default | 0.031206 |  | 0.968794 | 40 | 0.354654 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | no_gain_floor | 0.043758 |  | 0.956242 | 40 | 0.234105 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 7 | quality_first | 0.034106 |  | 0.965894 | 40 | 0.374669 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | current_auto | 0.032260 |  | 0.967740 | 40 | 0.242351 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | manual_default | 0.027507 |  | 0.972493 | 40 | 0.361846 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | no_gain_floor | 0.032260 |  | 0.967740 | 40 | 0.229382 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 13 | quality_first | 0.029329 |  | 0.970671 | 40 | 0.358844 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | current_auto | 0.040766 |  | 0.959234 | 40 | 0.250497 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | manual_default | 0.030942 |  | 0.969058 | 40 | 0.356368 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | no_gain_floor | 0.040766 |  | 0.959234 | 40 | 0.233680 |  |
| large-wide-16384x256-ranking | large-wide | ranking | 29 | quality_first | 0.031514 |  | 0.968486 | 40 | 0.377173 |  |
