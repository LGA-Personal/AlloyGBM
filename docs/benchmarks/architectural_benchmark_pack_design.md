# Architectural Benchmark Pack Design

## Purpose

The July 2026 review left six substantial architecture projects in the roadmap:
SoA histograms, node-level parallelism, duplicate row-major bin storage, compact
predictor nodes, Exclusive Feature Bundling (EFB), and approximate quantile
sketches. This benchmark pack defines the evidence each project must produce
before and after implementation. It does not implement any of the six projects.

The pack must run on the current codebase before candidate implementations
exist. A baseline report is captured from an unmodified checkout; a candidate
report is captured from the implementation branch on the same machine. The
comparator evaluates correctness and quality first, then performance.

## Repository Layout

The benchmark code lives in a focused Python package:

```text
benchmarks/architectural_backlog/
  __init__.py
  common.py
  fixtures.py
  scenarios.py
  run.py
benchmarks/tests/test_architectural_backlog.py
crates/engine/examples/sparse_predictor_fixture.rs
docs/benchmarks/architectural_backlog_v1.md
docs/benchmarks/architectural_backlog_*_implementation.md
```

`common.py` owns the JSON schema, environment manifest, subprocess execution,
RSS normalization, aggregation, and comparison. `fixtures.py` owns deterministic
data generation. `scenarios.py` owns the six scenario definitions and worker
functions. `run.py` is the only command-line entry point.

Keeping the scenarios in one module is deliberate: they share estimator setup,
result fields, and subprocess lifecycle, while remaining individually selectable
by name. The module should stay below roughly 800 lines; split it by scenario
only if implementation pushes it past that boundary.

## Command Contract

Baseline capture:

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode baseline \
  --output benchmarks/results/architectural_backlog_baseline.json
```

Candidate comparison:

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate \
  --baseline benchmarks/results/architectural_backlog_baseline.json \
  --output benchmarks/results/architectural_backlog_candidate.json \
  --gate
```

Development smoke run:

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile quick --mode baseline --gate
```

The CLI also accepts repeated `--scenario` arguments. Valid names are
`soa_histograms`, `node_parallelism`, `duplicate_bins`, `compact_nodes`, `efb`,
and `quantile_sketches`.

## Isolation And Measurement

Every case and repetition runs in a fresh subprocess. This is required because
Python, NumPy, the native predictor, and Rust allocations otherwise leave high
water marks that make later RSS measurements meaningless. Full runs use three
repetitions and aggregate medians; quick runs use one repetition.

The worker records:

- elapsed wall time from `time.perf_counter()`;
- `fit_timing_` components when available;
- live RSS immediately before fit/load plus process peak RSS afterward,
  normalized to MiB; their difference is the incremental fit/load peak;
- dimensions, seed, estimator parameters, and active thread count;
- held-out RMSE or rank error where quality can change;
- a SHA-256 digest of canonical `float32` predictions where behavior must be
  invariant.

Live RSS uses `/proc/self/statm` on Linux and `ps` on macOS; peak RSS uses
`resource.getrusage`. Fixtures avoid whole-matrix float64 temporaries so a
pre-fit allocation does not establish an unrelated high-water mark. RSS fields
are nullable on unsupported platforms. A missing RSS value skips only memory
gates; zero remains a measured value and does not remove a gate.

The environment manifest records git SHA, dirty state, AlloyGBM version and
extension path, Python and NumPy versions, platform, machine architecture,
logical CPU count, and `RAYON_NUM_THREADS`. Performance gates require matching
platform, architecture, logical CPU count, Python major/minor, profile, and
thread count between baseline and candidate reports.

## Result Schema

The JSON document uses `schema_version = 1` and contains:

```json
{
  "schema_version": 1,
  "profile": "full",
  "mode": "baseline",
  "environment": {},
  "results": [
    {
      "scenario": "duplicate_bins",
      "case": "wide_shallow",
      "repetition": 0,
      "metrics": {},
      "dimensions": {},
      "parameters": {}
    }
  ]
}
```

Numeric metrics must be finite and non-negative unless the metric's definition
explicitly permits signed values. Reports with missing cases, duplicate
`(scenario, case, repetition)` keys, unknown schema versions, or malformed
metrics fail before comparison.

## Scenario Contracts

### 1. SoA Histograms

The fixture stresses histogram accumulation and materialization across three
gain paths:

| Case | Full shape | Trees | Special setting |
| --- | ---: | ---: | --- |
| `standard_wide` | 100,000 x 128 | 50 | scalar leaves, 64 bins |
| `standard_deep` | 200,000 x 24 | 60 | scalar leaves, depth 10 |
| `dro_wide` | 75,000 x 48 | 40 | DRO leaf robustness |
| `linear_leaf` | 30,000 x 16 | 24 | `leaf_model="linear"` |

Quick shapes are 8,000 x 16, 6,000 x 12, and 2,000 x 8 with 4 trees. All use
deterministic quantile binning, manual policy, depth 6, and a fixed held-out
split. The worker reports fit time, native training time, incremental peak RSS, RMSE, and
prediction digest.

Candidate gates on a full same-host run:

- all prediction digests exactly match baseline;
- each RMSE matches baseline within `1e-7` absolute;
- no case exceeds `1.05x` baseline fit time or RSS;
- `standard_wide` median native training time is at most `0.90x` baseline.

The special-mode cases are regression guards, not required speed wins.

### 2. Node-Level Parallelism

The full fixture is a Boolean hypercube with `2^20` rows, 14 features, 12
informative binary bits, distinct power-of-two target weights, and one manual
level-wise depth-12 tree. It creates up to 2,048 small nodes at a level while
keeping per-node feature parallelism disabled. Quick mode uses `2^14` rows,
10 features, and depth 8.

The parent launches identical workers with `RAYON_NUM_THREADS=1` and
`RAYON_NUM_THREADS=8`; Rayon is configured before importing AlloyGBM. The report
includes fit time, native training time, RMSE, prediction digest, and emitted
stump count. Full runs require at least 3,500 stumps so lower-level pressure
cannot silently disappear through regularization or fixture drift.

Candidate gates:

- each thread-count prediction digest equals its corresponding baseline;
- repeated runs at one thread and repeated runs at eight threads are internally
  deterministic; cross-thread artifacts are compared structurally/tolerantly
  because existing parallel partition reductions may change f32 order;
- one-thread fit time is no worse than `1.05x` baseline;
- eight-thread native training time is at most `0.85x` baseline;
- eight-thread time is at most `0.80x` the candidate's one-thread time;
- peak RSS is no more than `1.25x` baseline.

The last gate proves useful scaling rather than merely moving work between
threads.

### 3. Duplicate Bin Storage

Two one-tree, shallow fixtures isolate binned-matrix construction and storage:

| Case | Full shape | Purpose |
| --- | ---: | --- |
| `wide_shallow_u8` / `wide_shallow_u16` | 30,000 x 512 | feature-dominant allocation |
| `tall_narrow_u8` / `tall_narrow_u16` | 600,000 x 16 | row-dominant allocation |

The u8 arms use 64 bins; the u16 arms use 256 bins. This matters because the
current u8 constructor retains four bytes per cell across legacy/adaptive
row/column mirrors, while u16 retains six bytes per cell. Quick shapes are
2,000 x 64 and 20,000 x 8. Live RSS is sampled after the input array
and target exist, so the incremental peak focuses on adaptation, binning, and
native training allocations rather than fixture storage. The report also
records input bytes, fit time, input-adaptation time, and prediction digest.

Candidate gates:

- prediction digests exactly match baseline;
- native training time is no worse than `1.03x` baseline;
- native bridge preparation time is at most `0.95x` baseline;
- incremental peak RSS is at most `0.80x` baseline in both full cases.

The memory gate is intentionally below the theoretical 50% binned-storage cut
because raw inputs and unrelated fit buffers remain live.

### 4. Compact Predictor Nodes

The orchestrator uses a Rust example to construct a deterministic artifact with
eight 16-node right-spine trees. Each spine ends at heap-local node id `65,534`,
so the current loader allocates 65,535 slots for only 16 populated nodes. The
artifact contains 128 stumps and remains only a few KiB; it does not rely on
best-split ordering to induce a sparse shape. Quick mode uses one right-spine
tree.

Each fresh worker records live RSS before and peak RSS after loading the native artifact, warms
the native predictor, then measures repeated prediction over a fixed 100,000-row
batch (5,000 rows in quick mode). It reports load time, incremental load peak,
prediction time per row, artifact bytes, and prediction digest. Artifact size is
descriptive because the compact representation is load-time-only.

Candidate gates:

- artifact bytes and prediction digest exactly match baseline;
- load time is no worse than `1.10x` baseline;
- incremental load peak is at most `0.25x` baseline;
- deep-spine prediction time per row is at most `0.85x` baseline;
- a balanced shallow control is no slower than `1.05x` baseline.

Scalar parity is the performance gate. The implementation plan separately
requires unit parity for linear, categorical, DART, multiclass, and SHAP paths.

### 5. Exclusive Feature Bundling

The fixture creates grouped one-hot features with a nonlinear held-out target:

| Case | Full shape | Conflict rate |
| --- | ---: | ---: |
| `exclusive_one_hot` | 80,000 x 512 | 0.0 |
| `controlled_conflict` | 80,000 x 512 | 0.02 |

There are 32 groups of 16 features. Quick mode uses 4,000 rows and 64 features.
Baseline mode fits the current unbundled estimator. Candidate mode requires the
future `feature_bundling="exact"` estimator parameter and records that the
candidate path was activated.

The report includes fit time, incremental peak RSS, RMSE, prediction digest, original and
effective feature counts when exposed, and conflict diagnostics when exposed.

Candidate gates:

- candidate activation is confirmed;
- exclusive-case RMSE is within `1e-6` absolute of baseline;
- controlled-conflict input is deterministically refused for bundling and
  follows the unbundled path with identical artifacts;
- exclusive-case total fit time improves by at least 15% or RSS by at least 20%;
- a dense non-bundleable control does not slow down by more than 3%.

The controlled-conflict case may reject unsafe bundles; correctness takes
priority over forcing a performance win.

### 6. Approximate Quantile Sketches

The fixture uses independent lognormal, exponential, heavy-tailed Student-t,
mixture, duplicated, and missing-value columns. Full mode uses 1,000,000 rows
and 16 features; quick mode uses 25,000 rows and 8 features. Baseline mode uses
the exact quantile path. Candidate mode requires the future
`quantile_sketch_max_rows=65536` parameter.

The worker measures fit/native-bridge preparation time and RSS, reads the fitted quantile
cuts, and evaluates each cut against the empirical CDF of the source column. It
reports mean, p99, and maximum interval rank error on continuous columns plus
held-out RMSE from a shallow model. Duplicate and constant columns are checked
separately for finite, strictly increasing cuts because deduplication removes the
requested-quantile identity needed for a meaningful rank-error score.

Candidate gates:

- candidate activation is confirmed;
- mean absolute rank error is at most `0.0025`;
- p99 absolute rank error is at most `0.0075`;
- maximum absolute rank error is at most `0.01`;
- held-out RMSE is at most `1.01x` baseline;
- native bridge preparation time is at most `0.60x` baseline;
- total fit time is no worse than `1.05x` baseline;
- incremental peak RSS is at most `0.90x` baseline and falls by at least
  32 MiB in the full case.

Repeated values and NaNs must preserve strictly increasing, finite cut arrays
and existing missing-bin behavior. The implementation plan must first pin the
current Rust/Python upper-tail bin contract and must return native cut metadata
to joint mode instead of re-deriving prediction cuts in Python.

## Quick Versus Full Gates

Quick mode gates schema validity, finite metrics, fixture determinism, candidate
activation, and quality/parity. It never rejects a change for timing or RSS.
Full mode enables performance gates only when a baseline report is supplied and
the environment compatibility check passes.

Timing and RSS are always shown even when not gated. A failed compatibility
check produces a clear nonzero result under `--gate`; it never silently treats
cross-machine numbers as comparable.

## Tests

Contract tests cover:

- deterministic fixtures and expected shapes/domains;
- schema round-trip and malformed-report rejection;
- platform-specific RSS unit normalization;
- median aggregation and ratio calculations;
- environment compatibility checks;
- one passing and one failing comparison per scenario;
- CLI quick-mode execution for a selected lightweight scenario;
- report rendering includes all six scenario names and gate outcomes.

Tests use synthetic `CaseResult` values for comparator branches. Only one small
real-model smoke test is needed; the full benchmark is not part of pytest.

## Documentation And Handoff

`docs/benchmarks/architectural_backlog_v1.md` records the baseline commit,
machine metadata, commands, results, and interpretation. Absolute performance
numbers are descriptive. Future implementation PRs attach same-host baseline
and candidate JSON and quote the comparator result.

Six independent implementation plans live under `docs/benchmarks/`. Each plan names
its production files, required behavior tests, benchmark command, acceptance
gates, and commit boundaries. The plans do not combine projects: SoA layout and
node parallelism remain sequenced because the latter depends on stable histogram
and partition ownership, but each produces reviewable software independently.
