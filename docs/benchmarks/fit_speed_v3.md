# Fit Speed v3: Per-Round Costs on Dense Data and Multiclass

- Harness: fixed-round timing on the synthetic shapes from
  [`data/fit_speed_v2/shapes.py`](data/fit_speed_v2/shapes.py), run by
  [`data/fit_speed_v3/time_interleaved.sh`](data/fit_speed_v3/time_interleaved.sh),
  plus a 29-config model-hash battery
  ([`data/fit_speed_v3/hash_battery.py`](data/fit_speed_v3/hash_battery.py)).
  No OpenML data was used; the OpenML suites are held out for final
  confirmation.
- Base: `main` at `8667813` (after PRs #154 and #156) for timing; hashes
  re-checked against `22232d8` (after #157). Candidate: this branch.
- Environment: Linux x86_64 container (4 cores), Python 3.13,
  `OPENBLAS_NUM_THREADS=1`. Main and branch runs are interleaved and the
  median of 5 is reported.
- Raw records: [`data/fit_speed_v3/`](data/fit_speed_v3/).

## Result

Every model is bit-identical to `main`. The 29-config battery (regression,
binary, multiclass, ranking, joint, plus leaf-wise, monotone, interaction
constraints, native and target-encoded categoricals, linear leaves, DRO,
MorphBoost, GOSS, DART, quantile, Poisson, feature bundling) hashes the same
artifact bytes and predictions at 1 and 4 threads, and at 60,000 rows. The
battery is sensitive to real changes: it flags the two linear-leaf configs
that #156 changed and the four binary and quantile configs that #157 changed.

Fit time at learning rate 0.05, depth 6:

| Data | Rounds | Threads | main (s) | branch (s) | Ratio |
|---|---:|---:|---:|---:|---:|
| 60,000 × 18, continuous regression | 300 | 1 | 3.75 | 2.95 | **0.79** |
| 60,000 × 18, continuous regression | 300 | 4 | 2.64 | 1.94 | **0.73** |
| 40,000 × 8, continuous binary | 300 | 1 | 1.92 | 1.51 | **0.78** |
| 40,000 × 8, continuous binary | 300 | 4 | 1.64 | 1.19 | **0.73** |
| 1,800 × 216, 10 classes | 100 | 1 | 27.4 | 23.0 | **0.84** |
| 1,800 × 216, 10 classes | 100 | 4 | 7.21 | 6.08 | **0.84** |
| 7,000 × 300, 26 classes | 30 | 1 | 28.3 | 22.2 | **0.78** |
| 7,000 × 300, 26 classes | 30 | 4 | 8.17 | 5.85 | **0.72** |
| 900 × 20, 12 low-cardinality features | 1,000 | 1 | 0.70 | 0.57 | 0.81 |
| 900 × 20, 12 low-cardinality features | 1,000 | 4 | 0.79 | 0.71 | 0.90 |
| 800 × 17, continuous | 300 | 1 | 0.222 | 0.197 | 0.89 |
| 800 × 17, continuous | 300 | 4 | 0.154 | 0.152 | 0.99 |
| 3,000 × 1,000, mostly 0/1 features | 300 | 1 | 3.25 | 3.19 | 0.98 |
| 3,000 × 1,000, mostly 0/1 features | 300 | 4 | 1.54 | 1.51 | 0.98 |

With many classes the softmax change dominates. On the dev-suite task
helena (100 classes, 15k rows; dev suite only, not an OpenML benchmark
task), 100 rounds at 4 threads went from 108 s to 15.6 s, against 17.9 s
for LightGBM at matched settings.

## What changed

Profiling dense fits on `main` put the time in histogram building (~20%),
the split scan (~19%), row partitioning (~15%), the out-of-bag prediction
replay used with `row_subsample` (~9%) and row sampling (~3%). Multiclass
fits spent most of their time in the softmax gradient. Each change below
keeps the same arithmetic in the same order:

- **Multiclass softmax once per row.** The trainer asked the objective for
  each class's gradients in turn, and every call recomputed the full
  normalizer, so a round cost K² exponentials per row. The new
  `compute_gradients_all_classes` computes each row's normalizer once.
  Pinned by `all_classes_matches_per_class_bit_for_bit`.
- **Out-of-bag replay through an index tree.** Replay looked up each node's
  stump in a hash map at every level of every row; children are now
  resolved to array indices once per round. Replay also goes parallel from
  4,096 rows instead of 32,768 (per-row scatter, no reduction).
- **Table-driven, branch-free row partition.** For a u8 bin column the
  direction of all 256 bins is computed once per split, so each row costs
  one table load. Rows are written to both sides and only the matching
  cursor advances, which removes a mispredicted branch per row. Child
  statistics still sum each side in row order.
- **Split-scan quick reject.** A chunk of eight thresholds whose best gain
  does not exceed the running best cannot change the argmax, so its edge
  filter and tolerance compare are skipped.
- **Row-sample pivot by bucketing.** The sampled-row pivot is found by
  counting hashes into 4,096 buckets and selecting only inside the bucket
  holding the target rank. Keys are unique, so it is exactly the pivot the
  full selection found (pinned by `bucketed_pivot_matches_full_selection`).
- **Parallel row-wise gradients and binary log loss.** Squared error,
  binary cross-entropy and quantile gradients depend only on their own row
  and are filled in parallel chunks. Binary log-loss row terms are computed
  in parallel and summed serially in index order. Objective formulas are
  untouched.

## Where the time goes now

On one thread AlloyGBM is now 1.3 to 1.5 times LightGBM per round on dense
data; at 4 threads it is 2 to 2.5 times. Most of the remaining gap is thread
scaling: the serial partition near the root and parallel tasks too small to
amortize. Wide sparse data (the 3,000 × 1,000 row above: 3.2 s against 0.65 s for
LightGBM on one thread) is the next change.

## Reproduce

```bash
# main and branch builds in separate virtual environments, then:
docs/benchmarks/data/fit_speed_v3/time_interleaved.sh main/.venv/bin/python branch/.venv/bin/python
GOLDEN_JOBS=4 python docs/benchmarks/data/fit_speed_v3/hash_battery.py hashes.json
```
