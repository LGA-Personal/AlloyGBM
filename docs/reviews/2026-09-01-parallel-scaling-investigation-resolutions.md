# Parallel Scaling Investigation — Resolutions

| Date | Reviewer | Version | Commit range | Status |
|---|---|---|---|---|
| 2026-09-01 | Claude Fable 5 | v1.0.0-prep | `a4c7622`..`eac4056` | Addressed |

Follow-up to [the parallel scaling investigation](2026-09-01-parallel-scaling-investigation.md).
Seven changes landed. Every one was A/B'd against a **simultaneously rebuilt
baseline** in absolute seconds, and every one leaves the trained model
bit-identical.

## Result

Best of three, `n_estimators=60`, `max_depth=6`, Apple M4:

| Shape | 1 thread | | 10 threads | |
|---|---:|---:|---:|---:|
| | before | after | before | after |
| 500k x 40 | 4.96 s | **4.23 s** | 2.71 s | **1.53 s** |
| 200k x 320 | 5.42 s | **4.91 s** | 1.58 s | **1.23 s** |
| 100k x 16 | 0.65 s | **0.57 s** | 0.43 s | **0.32 s** |

Multi-threaded fits are 22–44% faster; single-threaded 9–15% faster.

## Correction to the original analysis

The original document reported "parallel efficiency" against an assumed 10
equal cores and concluded efficiency had collapsed to 18%. **The test machine
is an Apple M4: 4 performance cores and 6 efficiency cores.** Weighting E-cores
at roughly a third of a P-core puts the ceiling near 6x, not 10x, so those
efficiency percentages were computed against an unreachable target.

This matters for how the remaining gap is read. Measured on the same machine,
with matched thread budgets, rounds, depth, and bin counts:

| Shape | Threads | AlloyGBM | LightGBM | XGBoost |
|---|---:|---:|---:|---:|
| 500k x 40 | 1 | 4.08 s | 2.93 s | 3.46 s |
| | 10 | 1.49 s | 1.14 s | 0.89 s |
| | speedup | 2.73x | 2.57x | 3.91x |
| 200k x 320 | 1 | 8.26 s | 12.02 s | 11.60 s |
| | 10 | 2.29 s | 3.74 s | 2.95 s |
| | speedup | 3.61x | 3.22x | 3.94x |

AlloyGBM now **scales as well as LightGBM** on both shapes and is the fastest
of the three on wide data. The residual gap is single-thread throughput on
row-heavy, feature-light data, not parallelism.

## What was fixed

1. **Nested Rayon in split finding** (`a4c7622`). Split finding forked a second
   Rayon region inside node-level parallelism, where the inner unit was a
   shortlist scan too small to pay for the join. Made sequential:
   0.29 s -> measured against 1.61 s for the nested variant.

2. **The 16-feature tile floor** (`3c8c47e`). `compute_optimal_tile_size`
   clamped tiles to >= 16 features, capping histogram parallelism at
   `ceil(features / 16)` workers — three, for a 40-feature fit, at any `n_jobs`.
   Floor lowered to 4 (best of {1, 2, 4, 8, 16}). Narrower tiles also shrink the
   per-tile arena, so cache residency improves rather than degrades.
   10 threads: 2.57 s -> 2.43 s on 500k x 40.

   This exposed a latent bug the floor had been masking:
   `IterationDiagnostics.n_active_features` was reporting the internal tile
   count, which varies with `n_jobs`, rather than the number of features
   eligible to split. Fixed to report `sampled_feature_count`.

3. **Serial complement replay** (`48a26c6`). With `row_subsample < 1.0` the
   builders apply a round only to sampled rows and replay the complement
   afterwards. That replay was serial and, once the histogram ceiling lifted,
   had become 21% of a 10-thread fit with *no* scaling. It is a pure scatter —
   each row writes only its own slot — so it parallelizes bit-identically.
   10 threads: 2.43 s -> 2.11 s.

4. **Node/tile parallelism as strict either/or** (`f2b2ea5`). Once a level had
   enough nodes to parallelize across, tile parallelism switched off entirely.
   Level-wise growth has 2, 4, and 8 nodes at depths 1–3, so those levels ran
   2-, 4-, and 8-wide with the rest of the pool idle — and each level touches
   roughly the whole row set. Now nested, but only when the node loop cannot
   fill the pool on its own. 10 threads: 2.11 s -> 1.89 s.

5. **Per-round row sampling** (`6264ab0`). 23.6% of a 10-thread fit and fully
   serial. It built a `row_count`-sized index list purely to compare its length
   against `row_count`, ranked the whole index space and then sorted both halves
   back into index order, and returned `Vec<usize>` that every caller
   immediately re-collected into `Vec<u32>`. Now: a short-circuiting `all()`, a
   pivot lookup plus one ascending scan that emits both lists already sorted,
   and `u32` throughout. ~18 MB of per-round allocation and a 500k-element sort
   removed. 10 threads: 1.89 s -> 1.65 s; 1 thread: 4.92 s -> 4.69 s.

6. **Re-gathered gradients** (`4130644`). The histogram kernels read
   `gradients[row_index]`, repeating the same scattered gather once per feature
   — 40 sparse passes over the gradient array for a 40-feature fit, and worse
   with depth, since a deep node's rows are spread thinly enough that most of
   each cache line is discarded. Gathered once per node into row-list order
   (LightGBM's ordered-gradients trick). 1 thread: 4.69 s -> 4.37 s.

   Per-level timing is what pointed here: every level was stuck between 2.4x and
   3.6x, *including the root*, where 10 tiles are available and tree structure
   cannot explain the ceiling. A limit that uniform across levels is a
   memory-access limit, not a parallelism one.

7. **Three cache lines per bin update** (`eac4056`). The arena holds separate
   grad / hess / grad_sq / count arrays, so one logical bin update wrote three
   or four distinct cache lines. Rows now accumulate into a 16-byte-per-bin
   interleaved scratch — one line per update — folded into the arena once per
   feature, which is O(bins) against an O(rows) loop. 1 thread: 4.41 s -> 4.23 s.

## What was tried and reverted

**Hoisting the bin-storage dispatch out of the inner loop.** `col_bin` branches
on the bundle map and on U8-vs-U16 storage for every element; slicing the
column once per feature and dispatching on width ahead of the loop looked like
an obvious win. It was consistently *slower* — 200k x 320 single-thread 5.16 s
-> 5.63 s, reproduced across an alternating A/B. LLVM was already unswitching
those loop-invariant branches, and the rewrite added bounds-checked reslicing
without removing anything real. Reverted.

**Dropping the per-bin count update** (as a probe to size the cost of the third
array). Invalid as a measurement: zeroed counts make `min_rows_per_leaf` reject
every split, so the trees collapse to stumps. It measured less work, not faster
work, and was discarded rather than acted on.

## Against the original plan

The original document proposed row-block partial histograms with a
deterministic reduction, and framed the cross-`n_jobs` determinism guarantee as
the obstacle — since no peer offers that guarantee, and it forces the serial
and parallel paths to share one summation order.

**That trade never had to be made.** Every change above is bit-identical across
thread counts, and together they closed the scaling gap to LightGBM without
touching the guarantee. Items 1 and 3 of the plan (cheaper reduction, not
paying it single-threaded) are moot; item 4 (parallelising the per-round
passes) turned out to matter only for the complement replay — gradients and
loss profile at 1.6% and 0.5% and are not worth parallelising. Item 2 (the tile
floor) was real and is fixed.

The lesson recorded there — always A/B against a simultaneously measured
baseline, in absolute time — is what caught the reverted change above.

## Instrumentation

An opt-in stage profiler (`crates/engine/src/profiling.rs`, `ALLOYGBM_PROFILE=1`)
now reports a per-stage breakdown of the round loop plus histogram / split-find
/ partition sub-stages, and an `other (untimed)` residual — loop wall minus the
sum of the timed stages. The residual is what surfaced finding 5: row sampling
had no timer, so a quarter of the loop sat in a bucket the report never showed.
It is now 2.5%.

## Remaining

Single-thread throughput on row-heavy, feature-light shapes (500k x 40:
AlloyGBM 4.08 s vs LightGBM 2.93 s). Histogram construction is ~70% of it.

The obvious next step — making `HistogramArena` array-of-structs throughout,
rather than folding an interleaved scratch into an SoA arena as finding 7 does
— was investigated and **should not be pursued as stated**. `HistogramBundle`
stores SoA, and the split scanners read it as SoA slices (`grad_sums() ->
&[f32]`), which is what lets `dro_scan` and `morph_scan` vectorize. An AoS
arena would therefore still transpose at `to_bundle`; it would relocate that
cost rather than remove it, while working against the scanners' layout. The
scratch in finding 7 already captures the accumulation-side cache benefit, and
the transpose is O(bins) against an O(rows) accumulation.

Closing the remaining gap means finding a cheaper inner loop rather than a
different arena layout. Note also that the comparison above is default-vs-
default: AlloyGBM's auto policy subsamples rows at 0.8 while LightGBM does not,
so AlloyGBM is doing ~20% less histogram work for that result — the per-update
gap is wider than the wall-clock gap suggests.
