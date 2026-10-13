# Fit Speed v4: Skip Each Sparse Feature's Dominant Bin

- Harness: default fits on the 14 dev-suite tasks with the most features
  dominated by a single value (3 splits each, 4 threads,
  [`data/fit_speed_v4/dev_fit.py`](data/fit_speed_v4/dev_fit.py)), repeat
  splits on the two tasks that moved most, and fixed-round timing on the
  synthetic shapes from [`data/fit_speed_v2/shapes.py`](data/fit_speed_v2/shapes.py)
  ([`data/fit_speed_v3/time_interleaved.sh`](data/fit_speed_v3/time_interleaved.sh)).
  The dev suite is disjoint from the OpenML benchmark suites, which stay held
  out for final confirmation.
- Base: PR #158 (`claude/fit-speed-0qeqb9`, on `main` 22232d8). Candidate:
  this branch.
- Environment: Linux x86_64 container (4 cores), Python 3.13,
  `OPENBLAS_NUM_THREADS=1`. Base and candidate runs are interleaved.
- Raw records: [`data/fit_speed_v4/`](data/fit_speed_v4/).

## Result

Wide data with mostly-constant columns fits about twice as fast. Accuracy is
unchanged within noise. Models are not bit-identical: a histogram slot that
used to be a sum of rows is now the node total minus the other slots, so
float rounding differs.

Dev-suite default fits (auto rounds, 3 splits per task):

| Task | Loss ratio | base (s) | candidate (s) | Time ratio |
|---|---:|---:|---:|---:|
| micro-mass | 1.0110 | 384.8 | 287.3 | 0.75 |
| fabert | 1.0000 | 86.0 | 37.2 | **0.43** |
| QSAR-TID-11 | 1.0028 | 23.0 | 11.3 | **0.49** |
| QSAR-TID-10980 | 0.9957 | 22.6 | 12.1 | **0.54** |
| Allstate Claims Severity | 1.0000 | 17.9 | 16.0 | 0.90 |
| KDDCup99 | 1.0039 | 16.6 | 13.3 | 0.80 |
| KDDCup09 appetency | 1.0000 | 8.5 | 8.5 | 1.00 |
| Mercedes-Benz | 0.9995 | 6.9 | 5.8 | 0.85 |
| pol | 0.9944 | 5.7 | 6.0 | 1.05 |
| porto-seguro | 1.0000 | 3.6 | 3.6 | 0.99 |
| jasmine | 1.0000 | 3.3 | 2.9 | 0.88 |
| house prices (nominal) | 1.0001 | 1.3 | 1.3 | 1.00 |
| analcatdata supreme | 1.0000 | 1.1 | 1.1 | 0.94 |
| compas two-years | 1.0000 | 0.8 | 0.8 | 0.95 |
| **All 14** | **1.0005** (geomean) | **582** | **407** | **0.70** |

The two largest loss moves are noise. Rerun with more splits
([`dev_fit_n_splits.py`](data/fit_speed_v4/dev_fit_n_splits.py)):
micro-mass over 8 splits is +0.11% (sd 2.05%, standard error 0.73%), and pol
over 10 splits is −0.02% (sd 1.42%). Those reruns used earlier builds of the
same two changes; they produce the same models as this branch on these tasks.

Fixed-round timing (300 rounds, lr 0.05, depth 6, median of 5):

| Data | Threads | base (s) | candidate (s) | Ratio |
|---|---:|---:|---:|---:|
| 3,000 × 1,000, mostly 0/1 features | 1 | 3.34 | 1.45 | **0.43** |
| 3,000 × 1,000, mostly 0/1 features | 4 | 1.72 | 0.94 | **0.55** |
| 900 × 20, 12 low-cardinality (1,000 rounds) | 1 | 1.27 | 1.18 | 0.93 |
| 1,800 × 216, 10 classes (100 rounds) | 1 | 23.2 | 18.3 | 0.79 |
| 60,000 × 18, continuous | 4 | 1.79 | 1.74 | 0.97 |
| 40,000 × 8, continuous | 4 | 1.52 | 1.48 | 0.98 |

Dense continuous data has no dominant bins, so it takes the old path; it is
unchanged or slightly faster from the allocation change. On the sparse
3,000 × 1,000 shape LightGBM takes 0.65 s on one thread, so a 2.2× gap remains
there (5× before this PR and #154).

## What changed

- **Dominant-bin skip.** When one bin holds at least 80% of a feature's rows
  (`DOMINANT_BIN_MIN_SHARE`), the histogram kernel does not add rows into it.
  After the other bins are summed, that slot is filled as the node's totals
  minus the other bins. Counts are exact; an empty slot is set to exactly
  zero, and hessians are clamped at zero against rounding. Per-feature
  dominant bins are computed once per binned matrix and cached.
- **Sparse row walk.** For each feature with a dominant bin, the matrix
  keeps the ascending list of rows outside it. When that list is much
  shorter than the node (`3 × list < node rows`), the kernel walks the list
  and keeps only rows in the node, using a per-thread stamp map of node
  positions. Node rows are ascending, so the walk adds rows in the same
  order as the skip path and gives identical sums.
- **One allocation per node bundle.** Each node's histogram bundle is sized
  once and every tile appends into it, instead of cloning and growing per
  tile. This change alone is bit-identical.

## Tests

- `dominant_bin_histograms_match_direct_sums`: on a sparse column, a dense
  column and a mostly-missing column, histograms built with the row walk
  (large nodes) and with the plain skip (small nodes) match direct per-bin
  sums.
- `append_parts_matches_append`: the sized append matches the old append.
- `test_exact_bundling_preserves_artifact_and_predictions` now checks
  bit-identical predictions (on the training rows and a random probe set)
  instead of identical artifact bytes. Unbundled one-hot columns fill their
  dominant bin by subtraction while the bundle sums it directly, so stored
  split statistics can differ in the last bit; trees and leaves are the same.
- The 29-config hash battery changes on 27 configs; the two that stay the
  same (`reg_wide`, `joint`) have no dominant bins on the trained path.
  This is expected: the battery data includes 20 sparse 0/1 columns.

## Reproduce

```bash
# base and candidate builds in separate virtual environments, then:
docs/benchmarks/data/fit_speed_v3/time_interleaved.sh base/.venv/bin/python cand/.venv/bin/python
# dev suite (harness from benchmarks/dev_suite.py):
python docs/benchmarks/data/fit_speed_v4/dev_fit.py out.json 168910 168911 359951 359953 \
  360112 360113 360932 360933 361062 361093 361097 361286 361292 3945
```
