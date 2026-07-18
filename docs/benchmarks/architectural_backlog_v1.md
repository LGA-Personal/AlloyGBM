# Architectural Backlog Benchmark v1

This report records the implementation baseline for the six architecture
projects identified by the July 2026 core review. It is an offline synthetic
benchmark and acceptance contract, not a cross-library comparison.

## Baseline

- Production baseline: `8bec92c` (main after PR #113)
- Benchmark harness: `c0c22c1`
- AlloyGBM: `0.12.10`
- Host: `Mac16,12`, arm64, 10 logical CPUs, macOS `26.5.2`
- Python: `3.13.5`; NumPy: `2.5.0`
- Rust: `1.92.0`
- Repetitions: three fresh subprocesses per case; tables show medians

The run was captured from a clean worktree. Timing and RSS are descriptive for
this host. Candidate performance gates require a new baseline on the same host
and software environment; the committed values should not be used as
cross-machine thresholds.

Command:

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode baseline \
  --output benchmarks/results/architectural_backlog_baseline.json \
  --gate
```

Raw JSON under `benchmarks/results/` is intentionally git-ignored because it
contains machine-specific paths and measurements. Implementation branches
should retain their same-host baseline and candidate JSON as PR or CI artifacts.

## SoA Histograms

| Case | Fit (s) | Native train (s) | RSS delta (MiB) | Stumps | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_wide` | 1.358 | 1.286 | 214.23 | 3,150 | 1.93602 |
| `standard_deep` | 4.608 | 4.567 | 131.56 | 55,360 | 0.39240 |
| `dro_wide` | 0.627 | 0.606 | 66.84 | 2,520 | 1.05731 |
| `linear_leaf` | 0.239 | 0.232 | 15.47 | 1,500 | 0.28369 |

`standard_deep` is the dominant training arm, while `standard_wide` carries
the largest memory delta. DRO and linear-leaf arms establish special-mode
parity guards rather than requiring speedups.

## Node-Level Parallelism

| Threads | Fit (s) | Native train (s) | RSS delta (MiB) | Stumps | RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.643 | 0.531 | 272.62 | 4,002 | 0.52254 |
| 8 | 0.644 | 0.530 | 272.88 | 4,002 | 0.52254 |

The fixture reaches 4,002 splits, including 1,955 active depth-11 nodes. The
current serial active-node loop shows effectively no 1-to-8-thread improvement,
which confirms that the candidate benchmark measures the reviewed bottleneck.

## Duplicate Bin Storage

| Case | Fit (s) | Bridge prepare (s) | Native train (s) | RSS delta (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `wide_shallow_u8` | 0.096 | 0.077 | 0.009 | 244.48 |
| `wide_shallow_u16` | 0.117 | 0.094 | 0.009 | 281.08 |
| `tall_narrow_u8` | 0.097 | 0.050 | 0.021 | 210.50 |
| `tall_narrow_u16` | 0.100 | 0.053 | 0.020 | 229.78 |

The u16 arms consume more memory and bridge-preparation time than their u8
counterparts. Candidate gates require exact prediction parity, at least a 20%
RSS reduction, and no native-training regression.

## Compact Predictor Nodes

| Case | Load (s) | RSS delta (MiB) | Predict (ns/row) | Artifact bytes |
| --- | ---: | ---: | ---: | ---: |
| `sparse_spines` | 0.014742 | 126.95 | 54.50 | 4,274 |
| `shallow_control` | 0.000082 | 0.00 | 13.25 | 1,970 |

The sparse artifact contains only 128 stumps and is 4.2 KiB, but heap-indexed
runtime slots raise the loaded-process high-water mark by about 127 MiB. This
large signal supports a strict memory gate while the balanced shallow artifact
guards ordinary prediction throughput.

## Exclusive Feature Bundling

| Case | Fit (s) | Native train (s) | RSS delta (MiB) | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `exclusive_one_hot` | 0.725 | 0.605 | 476.23 | 1.53950 |
| `controlled_conflict` | 0.794 | 0.677 | 476.58 | 1.52217 |
| `dense_control` | 1.128 | 0.902 | 650.42 | 5.28598 |

The exact one-hot case provides a substantial memory and training-time target.
The conflict arm must fall back unchanged, and the non-bundleable dense arm
prevents detection overhead from becoming a general regression.

## Approximate Quantile Sketches

| Rows x features | Fit (s) | Bridge prepare (s) | Native train (s) | RSS delta (MiB) | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 x 16 | 0.625 | 0.076 | 0.506 | 213.00 | 0.68410 |

The exact baseline has zero interval rank error by construction. Candidate
sketches must meet the mean/p99/max error budgets of `0.0025`, `0.0075`, and
`0.01`, reduce bridge preparation to at most 60% of baseline, and reduce RSS by
at least 10% and 32 MiB without materially changing held-out quality.

## Result

All baseline schema, finite-value, fixture-depth, and deterministic-fixture
gates passed. Production implementation remains open for all six projects; the
independent plans in this directory define code changes, regression tests,
commit boundaries, and candidate commands.
