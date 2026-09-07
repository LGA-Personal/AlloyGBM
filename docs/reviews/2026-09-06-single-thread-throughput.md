# Single-Thread Throughput: Problem Statement

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-09-06 | Claude Opus 5 | v1.0.0-prep | `a35bc42` | Open — see [the ideas board](../ideas/single-thread-optimization-board.md) |

**In one sentence:** AlloyGBM is at accuracy parity with LightGBM, XGBoost, and
CatBoost and scales across threads about as well as they do, but it is 1.4x to
5.3x slower per core, and that single number is now the whole competitiveness
gap.

All measurements below are on one host — **Apple M4, 4 performance + 6
efficiency cores, macOS** — with matched hyperparameters, thread budgets, bin
counts, and sampling rates. Peer libraries are the control in every comparison.

---

## 1. Where we actually stand

### Accuracy — at parity

Median of 5 seeds across the 15-scenario curated suite. Differences inside
AlloyGBM's own seed noise are counted as ties.

| | Win | Tie | Lose |
|---|---:|---:|---:|
| vs LightGBM | 1 | 14 | 0 |
| vs XGBoost | 1 | 13 | 1 |
| vs CatBoost | 2 | 12 | 1 |

A direction-only sign test agrees: better than XGBoost on 8/15 (median margin
+0.01%, p = 1.00), better than CatBoost on 9/15 (+1.30%, p = 0.61), better than
LightGBM on 5/15 (−0.68%, p = 0.30). `histogram_stress` is a genuine standout at
0.149 RMSE against 0.31–0.34 for all three peers.

Earlier claims that AlloyGBM was "less accurate" came from single-seed readings
of one synthetic generator and do not survive error bars. See
[`docs/benchmarks/data/2026-09-06-seed-variance/`](../benchmarks/data/2026-09-06-seed-variance/README.md).

### Parallel scaling — competitive

On 1M rows, 1 → 10 threads: AlloyGBM 2.43–2.58x, LightGBM 2.81–2.98x, CatBoost
3.35–3.38x, XGBoost ~1.0x (it does not scale on this host). AlloyGBM began this
cycle at 1.60–1.74x; seven optimizations closed that gap.

**Caveat worth stating plainly:** part of a good speedup ratio is having a slow
baseline, because more work per core amortizes parallel overhead better.
Expect the ratio to *fall* as single-thread throughput improves. That is
success, not regression — judge absolute wall time, never the ratio.

### Single-thread speed — the gap

100 rounds, depth 6, one thread, matched parameters:

| Rows x features | AlloyGBM | LightGBM | XGBoost | vs LightGBM |
|---|---:|---:|---:|---:|
| 2,000 x 20 | 0.272 s | 0.051 s | 0.169 s | **5.29x** |
| 5,600 x 20 | 0.291 s | 0.082 s | 0.195 s | 3.56x |
| 20,000 x 20 | 0.415 s | 0.171 s | 0.258 s | 2.43x |
| 100,000 x 20 | 1.068 s | 0.574 s | 0.579 s | 1.86x |
| 500,000 x 20 | 4.590 s | 2.183 s | 2.329 s | 2.10x |

The 216-cell deep sweep agrees: AlloyGBM is the slowest of the four at one
thread in **all nine** deep configurations, by 1.4x to 3.6x.

---

## 2. The diagnosis

**The gap grows as the dataset shrinks.** That is the signature of fixed
per-round cost, not per-row work. Profiling confirms it
(`ALLOYGBM_PROFILE=1`, one thread):

| Fixture | tree_build | split_find | histogram | partition |
|---|---:|---:|---:|---:|
| 2,000 x 20, depth 6, 100 rounds | 99.7% | **93.2%** | 6.8% | 0.0% |
| 400k x 40, depth 8, 60 rounds | 89.7% | 25.2% | **66.2%** | 8.6% |
| 400k x 40, depth 12, 40 rounds | 96.6% | **62.5%** | 36.1% | 1.4% |

Two distinct regimes, and they want different fixes:

**Small data and deep trees are split-finding bound.** Split cost is
`nodes x features x bins` and does **not** shrink with row count. At 2,000 rows
and depth 6 that is 63 nodes x 20 features x 256 bins ≈ 322,000 bin evaluations
to fit 40,000 data values. Nodes at the deepest level hold roughly 30 rows each
and we still scan all 256 bins for every feature. Depth makes it worse: node
count doubles per level while histogram work stays bounded by the row set, so
split-finding overtakes histogram construction between depth 8 and 12.

**Large shallow data is histogram bound.** At 400k x 40 and depth 8, histogram
construction is 66% and the per-row inner loop dominates. This is the regime
where the gap is smallest (~1.5–2x), which suggests per-row work is closer to
competitive than the fixed costs are.

### One confirmed, unexploited inefficiency

`best_split_for_feature_standard_simd` scans every bin **twice**:

```rust
// For each NaN direction, evaluate gain across all bins in 8-wide chunks.
for &default_left in &[true, false] {
```

The two passes differ only in which side receives `missing_grad`,
`missing_hess`, and `missing_count`. When a feature has no missing values those
are all zero, so both passes compute identical gains — and because the update
uses `gain_materially_exceeds`, the second pass can never win the tie. It is
duplicated work on every dense dataset, which is most of them.

Measured by applying the skip and reverting (details in
[idea 1 on the board](../ideas/single-thread-optimization-board.md)):
−37% at 2,000 rows, −26% at 20,000, −14% at 200,000 x 20 depth 8, with
**bit-identical artifacts** and all 807 cargo / 1024 pytest tests passing.

That is necessary but not sufficient. Even with it, split-finding remains
roughly 87% of a small fit, and 2,000 rows goes from 5.3x behind LightGBM to
about 3.5x.

---

## 3. What we are trying to achieve

**Goal:** close the per-core throughput gap without giving up anything we
currently have.

**Non-negotiable constraints.** These are not preferences; a proposal that
violates one is not usable:

1. `unsafe_code = "forbid"` workspace-wide. No exceptions.
2. **Trained models must stay bit-identical across `n_jobs`.** This is a
   published guarantee. XGBoost demonstrates it is compatible with being the
   fastest library in most of our cells, so it is not the thing costing us
   speed.
3. **Artifact format is stable from v1.0.0.** A 1.x artifact must stay readable
   by any later 1.x release.
4. Rust edition 2024, MSRV 1.92.0.
5. Accuracy must not regress. We are at parity; a speed change that trades it
   away is not a win.

**Preferred:** changes that leave trained models byte-for-byte unchanged. Those
are verifiable in seconds and carry no accuracy risk. Changes that alter model
outputs are permitted but need an accuracy measurement with error bars, not a
single-seed check.

---

## 4. Measurement discipline

This section exists because several plausible-sounding optimizations in this
cycle turned out to be regressions, and one apparent success was an artifact of
how it was measured.

- **Always A/B against a simultaneously rebuilt baseline, in absolute seconds.**
  One change in this cycle improved the parallel *speedup ratio* from 1.83x to
  1.88x while making both the single- and multi-threaded absolute times worse
  (4.96 s → 5.54 s and 2.71 s → 2.95 s). The ratio improved only because the
  baseline got slower.
- **Verify bit-identity** with an artifact SHA-256 across `n_jobs` 1 and 4 and
  against the pre-change build. This is fast and catches the class of error
  where a "speedup" came from doing less work.
- **Accuracy claims need seed variance.** Single-scenario deltas on the curated
  suite are dominated by seed noise: median scenario spread is 9.3%, and three
  scenarios exceed 58%. Use `benchmarks/curated_seed_variance.py`.
- **Profile before optimizing.** Work in this cycle was initially aimed at
  gradient computation and loss, which profiling later showed to be 0.5% and
  0.1% of runtime.
- **Tests must be shown to fail without the fix.** Revert the change, watch the
  new test fail, restore it, watch it pass.

### Things already tried that did not work

Recording these so they are not re-proposed without new evidence:

| Attempt | Outcome |
|---|---|
| Row-block partial histograms | Slower in absolute terms; reduction cost dominated. Implementation was likely at fault (too many blocks, per-block allocation) rather than the idea. |
| Hoisting the bin-storage `U8`/`U16` dispatch out of the inner loop | Measurably slower. LLVM already unswitches those loop-invariant branches; the rewrite added bounds-checked reslicing. |
| Enabling the dormant row-major histogram kernel | Bit-identical but slower. It fixes data locality while scattering across a 164 KB arena; the per-feature kernel keeps the histogram L1-resident. |
| Array-of-structs histogram arena | `HistogramBundle` is SoA because the SIMD split scanners read it that way, so an AoS arena relocates the transpose rather than removing it. |
| Caching the fit thread pool | ~1% at 128 workers, nothing at realistic counts; not worth a process-wide cache and a `fork()` hazard. |

---

## 5. Where the ideas live

Candidate optimizations, their evidence, and expert commentary are on the
[single-thread optimization board](../ideas/single-thread-optimization-board.md).
That document is open for contribution and has a structure for adding and
critiquing proposals.
