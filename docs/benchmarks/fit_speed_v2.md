# Fit Speed v2: Per-Feature Histogram Sizes

- Harness: `benchmarks/default_quality.py` (19 datasets, 5 seeds) for
  accuracy and suite time, plus fixed-round timing on synthetic shapes
  ([`data/fit_speed_v2/shapes.py`](data/fit_speed_v2/shapes.py),
  [`time_fit.py`](data/fit_speed_v2/time_fit.py)). No OpenML data was used:
  the OpenML suites are held out for final confirmation.
- Base: `main` at `b6f7262` (after PRs #150 and #152). Candidate: this branch.
- Environment: Linux x86_64 container, Python 3.13, `OPENBLAS_NUM_THREADS=1`.
  Container timing is noisy, so main and branch runs are interleaved and
  medians are reported.
- Raw records: [`data/fit_speed_v2/`](data/fit_speed_v2/).

## Result

Every model is bit-identical to `main`. Checked on three fronts: all 190
default-quality test losses match exactly (19 datasets × 5 seeds × the auto
and fixed-100-round arms); a 29-config battery (regression, binary,
multiclass, ranking, joint, plus leaf-wise, monotone, interaction
constraints, native and target-encoded categoricals, linear leaves, DRO,
MorphBoost, GOSS, DART, quantile, Poisson, feature bundling) hashes the same
artifact bytes and predictions; and the allocation-reuse gate's model hashes
match.

Fit time, 300 rounds at learning rate 0.05, depth 6:

| Data | Threads | main (s) | branch (s) | Ratio |
|---|---:|---:|---:|---:|
| 3,000 × 1,000, mostly 0/1 features | 1 | 7.31 | 3.34 | **0.46** |
| 3,000 × 1,000, mostly 0/1 features | 4 | 3.17 | 1.60 | **0.51** |
| 900 × 20, 12 low-cardinality features (1,000 rounds) | 1 | 0.84 | 0.78 | 0.93 |
| 900 × 20, 12 low-cardinality features (1,000 rounds) | 4 | 0.86 | 0.78 | 0.91 |
| 60,000 × 18, continuous | 1 | 3.80 | 3.80 | 1.00 |
| 60,000 × 18, continuous | 4 | 2.69 | 2.65 | 0.99 |
| 40,000 × 8, continuous | 1 | 2.04 | 1.99 | 0.98 |
| 40,000 × 8, continuous | 4 | 1.72 | 1.76 | 1.02 |
| 800 × 17, continuous | 4 | 0.146 | 0.146 | 1.00 |

Wide data with few distinct values per feature (one-hot, indicator and
count columns, common in OpenML and tabular work generally) fits about
twice as fast. Dense continuous data is unchanged within noise (median of 7
interleaved runs). The default-quality suite is mostly dense continuous
data, and its total time is unchanged within run-to-run noise (auto rounds
174.5 s vs 181.6 s, fixed 21.0 s vs 21.6 s, runs not interleaved).

## What changed

Before, every feature's histogram had `max_bin + 1` slots (255 by default),
even a 0/1 feature that fills two. Each tree node zeroed, filled, copied,
subtracted (parent minus sibling) and scanned all 255 slots for every
feature, so on a 1,000-feature matrix that is 255,000 slots per node,
almost all empty. Profiling the 3,000 × 1,000 case on `main` showed
histogram subtraction at 11%, `memmove` at 10% and `malloc`/`free` at about
13% of the fit.

Histograms now use a compact layout when the binned matrix allows it: each
feature gets as many slots as its largest used bin plus one, and its missing
bin goes last. The kernel maps the matrix-wide missing bin to that slot, so
data bin `b` keeps index `b` and rows are summed in the same order as
before. Split scans read the feature's own missing slot, and the "left
side already holds every non-missing bin" edge rule still compares against
the uniform scan limit, so every candidate threshold and its gain are the
same as before.

The layout is computed once per matrix and is skipped (uniform layout as
before) for bundled matrices (`feature_bundling`), matrices without a
column-major copy, and matrices where every feature already fills its
histogram.

## Where the time goes now

Per marginal round on `main`, AlloyGBM is still 2 to 5 times slower than
LightGBM at matched depth and learning rate (4 threads): 60k × 18
regression 9.4 vs 3.0 ms, 40k × 8 binary 5.6 vs 1.3 ms, 10-class 1.8k × 216
65 vs 34 ms. On dense data the fit splits between histogram building
(~20%), the split scan (~19%), row partitioning (~15%) and the out-of-bag
prediction replay used with `row_subsample` (~9%). Those are the next
targets.

## Reproduce

```bash
# main and branch builds in separate virtual environments, then for each:
python docs/benchmarks/data/fit_speed_v2/time_fit.py bin_3k_x1000_sparse 300 1
python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --output out.json
```
