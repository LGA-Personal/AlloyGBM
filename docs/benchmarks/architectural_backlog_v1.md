# Architectural Backlog Benchmark v1

This report records the implementation baseline for the six architecture
projects identified by the July 2026 core review. It is an offline synthetic
benchmark and acceptance contract, not a cross-library comparison.

## Baseline

- Production baseline: `8bec92c` (main after PR #113)
- Benchmark harness: `df872bc`
- AlloyGBM: `0.12.10`
- Host: `Mac16,12`, arm64, 10 logical CPUs, macOS `26.5.2`
- Python: `3.13.5`; NumPy: `2.5.0`
- Rust: `1.92.0`
- Repetitions: three fresh subprocesses per case; tables show medians

The run was captured from a clean worktree. Memory is the process high-water
mark after fit/load minus live RSS immediately before that phase. Timing and RSS
are descriptive for this host. Candidate performance gates require a new
baseline on the same host and software environment; the committed values should
not be used as cross-machine thresholds.

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

| Case | Fit (s) | Native train (s) | Incremental peak RSS (MiB) | Stumps | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard_wide` | 1.363 | 1.290 | 214.88 | 3,150 | 2.11583 |
| `standard_deep` | 4.626 | 4.586 | 135.30 | 55,594 | 0.35581 |
| `dro_wide` | 0.611 | 0.589 | 68.44 | 2,517 | 0.91528 |
| `linear_leaf` | 0.239 | 0.232 | 17.47 | 1,504 | 0.26825 |

`standard_deep` is the dominant training arm, while `standard_wide` carries
the largest memory delta. DRO and linear-leaf arms establish special-mode
parity guards rather than requiring speedups.

### SoA histogram candidate

The SoA implementation was measured on 2026-07-18 from the same host with a
fresh baseline at `e9450c2` and candidate implementation commit `7c4206a`.
Both runs used three fresh subprocesses per case; the table reports medians.

| Case | Variant | Fit (s) | Native train (s) | Incremental peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| `standard_wide` | baseline | 1.916 | 1.826 | 213.28 |
| `standard_wide` | SoA | 1.311 | 1.242 | 212.06 |
| `standard_deep` | baseline | 6.943 | 6.890 | 130.42 |
| `standard_deep` | SoA | 4.487 | 4.448 | 126.16 |
| `dro_wide` | baseline | 0.957 | 0.929 | 67.00 |
| `dro_wide` | SoA | 0.646 | 0.624 | 67.23 |
| `linear_leaf` | baseline | 0.348 | 0.339 | 17.06 |
| `linear_leaf` | SoA | 0.236 | 0.229 | 17.08 |

Median fit time improved by 31.6% for `standard_wide`, 35.4% for
`standard_deep`, 32.5% for `dro_wide`, and 32.1% for `linear_leaf`. Median
incremental fit RSS changed by -1.22 MiB, -4.27 MiB, +0.23 MiB, and +0.02 MiB,
respectively. Artifact and prediction digests matched exactly in every
repetition, so all SoA candidate gates passed.

## Node-Level Parallelism

| Threads | Fit (s) | Native train (s) | Incremental peak RSS (MiB) | Stumps | RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.642 | 0.531 | 274.28 | 4,002 | 0.52254 |
| 8 | 0.633 | 0.524 | 272.78 | 4,002 | 0.52254 |

The fixture reaches 4,002 splits, including 1,955 active depth-11 nodes. The
current serial active-node loop shows effectively no 1-to-8-thread improvement,
which confirms that the candidate benchmark measures the reviewed bottleneck.

### Node-parallelism candidate

The node-parallel implementation was measured on 2026-07-19 from the same host with a fresh
baseline at `519a9d4` and candidate implementation commit `29241f1`. Both runs used three fresh
subprocesses per thread count; the table reports medians.

| Threads | Variant | Fit (s) | Native train (s) | Incremental peak RSS (MiB) |
| ---: | --- | ---: | ---: | ---: |
| 1 | baseline | 0.592 | 0.482 | 273.23 |
| 1 | node parallel | 0.575 | 0.467 | 277.17 |
| 8 | baseline | 0.589 | 0.479 | 271.67 |
| 8 | node parallel | 0.392 | 0.285 | 286.72 |

Median eight-thread native training improved by 40.5%, candidate one-to-eight-thread scaling was
1.64x, and single-thread native time improved by 3.0%. Prediction digests, RMSE, artifact size, and
the 4,002-stump structure matched across all runs; artifact bytes were stable across repeated runs
at each thread count. Incremental fit RSS increased by 3.94 MiB at one thread and 15.05 MiB at
eight threads because active proposals own their child work concurrently. All timing and
determinism gates passed; the memory increase is the accepted throughput tradeoff for this project.

## Duplicate Bin Storage

| Case | Fit (s) | Bridge prepare (s) | Native train (s) | Incremental peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `wide_shallow_u8` | 0.093 | 0.075 | 0.009 | 246.02 |
| `wide_shallow_u16` | 0.112 | 0.090 | 0.008 | 282.77 |
| `tall_narrow_u8` | 0.092 | 0.046 | 0.020 | 210.36 |
| `tall_narrow_u16` | 0.099 | 0.053 | 0.020 | 229.78 |

The u16 arms consume more memory and bridge-preparation time than their u8
counterparts. Candidate gates require exact prediction parity, at least a 20%
RSS reduction, and no native-training regression.

## Compact Predictor Nodes

| Case | Load (s) | Incremental peak RSS (MiB) | Predict (ns/row) | Artifact bytes |
| --- | ---: | ---: | ---: | ---: |
| `sparse_spines` | 0.012143 | 128.53 | 53.45 | 4,274 |
| `shallow_control` | 0.000095 | 1.58 | 11.22 | 1,970 |

The sparse artifact contains only 128 stumps and is 4.2 KiB, but heap-indexed
runtime slots add about 129 MiB at load. This
large signal supports a strict memory gate while the balanced shallow artifact
guards ordinary prediction throughput.

### Compact-node candidate

The compact runtime implementation was measured on 2026-07-18 from the same
host with a fresh baseline at `bdee724` and candidate commit `6a58019`. Both
runs used three fresh subprocesses per case.

| Case | Variant | Load (s) | Incremental peak RSS (MiB) | Predict (ns/row) |
| --- | --- | ---: | ---: | ---: |
| `sparse_spines` | baseline | 0.014867 | 128.52 | 49.96 |
| `sparse_spines` | compact | 0.000110 | 1.58 | 35.59 |
| `shallow_control` | baseline | 0.000098 | 1.58 | 10.81 |
| `shallow_control` | compact | 0.000087 | 1.59 | 10.83 |

The compact loader reduced sparse-spine incremental load RSS by 98.8%, load
time by 99.3%, and prediction time per row by 28.8%. The shallow control moved
by less than 1% in RSS and prediction throughput. Artifact and prediction
digests matched exactly in every repetition, so all compact-node candidate
gates passed.

## Exclusive Feature Bundling

| Case | Fit (s) | Native train (s) | Incremental peak RSS (MiB) | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `exclusive_one_hot` | 0.746 | 0.630 | 476.08 | 1.53950 |
| `controlled_conflict` | 0.806 | 0.690 | 477.02 | 1.52217 |
| `dense_control` | 1.128 | 0.895 | 651.03 | 5.31900 |

The exact one-hot case provides a substantial memory and training-time target.
The conflict arm must fall back unchanged, and the non-bundleable dense arm
prevents detection overhead from becoming a general regression.

## Approximate Quantile Sketches

| Rows x features | Fit (s) | Bridge prepare (s) | Native train (s) | Incremental peak RSS (MiB) | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 x 16 | 0.627 | 0.078 | 0.505 | 212.78 | 0.68410 |

The exact baseline has zero interval rank error by construction. Candidate
sketches must meet the mean/p99/max error budgets of `0.0025`, `0.0075`, and
`0.01`, reduce bridge preparation to at most 60% of baseline, and reduce RSS by
at least 10% and 32 MiB without materially changing held-out quality.

## Result

All baseline schema, finite-value, fixture-depth, and deterministic-fixture
gates passed. SoA histograms, node-level parallelism, and compact predictor nodes have passed their
production candidate gates; implementation remains open for the other three projects. The independent
plans in this directory define their code changes, regression tests, commit
boundaries, and candidate commands.
