# DART Aggregate-Contribution Evidence (v1)

## Scope

This report compares the recorded pre-optimization DART profile with the
aggregate-contribution implementation. The implementation keeps one required
dropout traversal for each selected tree, accumulates its weighted
contribution in reusable `O(rows)` scratch storage, and restores the
normalized aggregate without a repeat tree walk. It does not change DART
configuration, dropout selection, normalization policy, or any quality gate.

## Commands and Environment

Recorded baseline source: `2bbab7ef0f522ceff63ea25e244797fbcbe5405e`.
Optimized source: `e0c1d7f95aafe255ab7357ad13df9043e7525602`.
Final-review source: `136bbd7`.

The baseline was generated with the same full DART command and recorded in
`/tmp/alloygbm-dart-baseline.md` before this branch's changes. That report did
not serialize host or tool-version metadata, so the optimized-run environment
below is not asserted as a contemporaneous baseline capture.

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/review_guardrails.py \
  --section dart --gate --output /tmp/alloygbm-dart-baseline.md
```

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/review_guardrails.py \
  --section dart --gate --output /tmp/alloygbm-dart-optimized.md
```

Optimized run captured on 2026-07-25 11:35:21 PDT:

- macOS 26.5.2 (build 25F84), Apple M4, 24 GiB memory;
- Rust 1.92.0, Cargo 1.92.0, Python 3.13.5, and maturin 1.12.6; and
- full profile: seeds 7, 13, and 29; 512 training and 256 held-out rows;
  depth 4; learning rate 0.06; `lambda_l2=1.0`; manual policy; deterministic
  quantile binning.

## Full Profile

Each figure is the rendered median from its report. Standard-time ratios use
unrounded median fit times. A dash means that a standard control has no
matched-standard ratio.

| Arm | Profile | Baseline fit s | Optimized fit s | Baseline ratio | Optimized ratio | Median RMSE (baseline -> optimized) | Dropout pressure |
|---|---|---:|---:|---:|---:|---:|---:|
| `dart_100_0.10_20` | default_like | 0.0322 | 0.0232 | 2.694 | 1.947 | 0.972156 -> 0.972166 | 499.50 |
| `dart_100_0.10_5` | default_like | 0.0261 | 0.0200 | 2.182 | 1.684 | 0.946943 -> 0.946906 | 377.00 |
| `dart_100_0.10_50` | default_like | 0.0320 | 0.0230 | 2.673 | 1.934 | 0.972156 -> 0.972166 | 499.50 |
| `dart_200_0.20_50` | stress_profile | 0.1648 | 0.0964 | 7.342 | 4.171 | 1.046657 -> 1.046686 | 3982.00 |
| `dart_50_0.05_50` | default_like | 0.0095 | 0.0082 | 1.492 | 1.255 | 0.941069 -> 0.941141 | 70.75 |
| `standard_100` | standard_control | 0.0120 | 0.0119 | - | - | 0.671367 -> 0.671367 | - |
| `standard_200` | standard_control | 0.0225 | 0.0231 | - | - | 0.571675 -> 0.571675 | - |
| `standard_50` | standard_control | 0.0064 | 0.0065 | - | - | 0.746986 -> 0.746986 | - |

The stress profile improved from `0.1648s` to `0.0964s` (a 41.5% reduction in
rendered median DART fit time) and from `7.342x` to `4.171x` matched-standard
time (a 43.2% reduction in the ratio). These are descriptive,
machine-dependent timings, not a CI performance contract; no wall-clock gate
was added.

## Final-Review Rerun

The full DART section was rerun after the artifact, multiclass scratch,
warm-start validation, and conditional scalar-allocation corrections. The
scalar allocation change touches the standard controls in this report, so the
saved evidence was not reused unchanged.

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/review_guardrails.py \
  --section dart --gate --output /tmp/alloygbm-dart-final-review.md
```

Captured on 2026-07-25 on the same Apple M4 development machine:

| Arm | Profile | Final fit s | Final standard ratio | Median RMSE |
|---|---|---:|---:|---:|
| `dart_100_0.10_20` | default_like | 0.0235 | 2.012 | 0.972166 |
| `dart_100_0.10_5` | default_like | 0.0203 | 1.741 | 0.946906 |
| `dart_100_0.10_50` | default_like | 0.0240 | 2.057 | 0.972166 |
| `dart_200_0.20_50` | stress_profile | 0.1112 | 4.908 | 1.046686 |
| `dart_50_0.05_50` | default_like | 0.0079 | 1.285 | 0.941141 |
| `standard_100` | standard_control | 0.0117 | - | 0.671367 |
| `standard_200` | standard_control | 0.0227 | - | 0.571675 |
| `standard_50` | standard_control | 0.0062 | - | 0.746986 |

The final-source stress median remains below the recorded baseline
(`0.1112s` versus `0.1648s`; `4.908x` versus `7.342x`) but is slower than the
earlier optimized capture. These short timings are machine- and run-dependent;
the rerun is compatibility evidence, not a stable estimate of the exact speed
change. All three DART gates passed again, and no wall-clock gate was added.

## Gates and Remaining Work

Both full reports passed the same DART gates:

| Gate | Result |
|---|---|
| Contract | Pass: requested seed/config/profile/control matrix, finite metrics, unique rows, and pressure |
| Completion | Pass: every DART fit completed its requested rounds |
| Quality | Pass: maximum default-like DART/standard RMSE ratio `1.448` (limit `1.500`; stress excluded) |

The high-pressure stress arm remains visible and contract-checked but
non-blocking for quality. This implementation does not calibrate an expected
drop cap or select a new DART default, so expected-drop calibration remains a
separate follow-up.
