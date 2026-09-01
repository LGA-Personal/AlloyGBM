# Parallel Scaling Investigation

| Date | Reviewer | Version | Commit | Status |
|---|---|---|---|---|
| 2026-09-01 | Claude Fable 5 | v1.0.0-prep | `089d524` | Bottleneck identified; one bug fixed, optimization deferred |

Goal: close AlloyGBM's parallel-scaling gap against LightGBM, XGBoost, and
CatBoost by adopting their techniques. Peer C++ sources are not available
locally (the wheels ship compiled binaries only), so this works from their
published designs and anchors every claim to measurements of *our* code.

## What the profiling found

On 500k rows x 40 features, 60 rounds, depth 6 (10 logical CPUs):

| Threads | Fit | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 4.96 s | 1.00x | 100% |
| 2 | 3.31 s | 1.45x | 72% |
| 4 | 2.80 s | 1.71x | 43% |
| 8 | 2.68 s | 1.79x | 22% |
| 10 | 2.71 s | 1.83x | 18% |

Amdahl's law puts the serial fraction near **49%** — the constraint is not the
efficiency of the parallel regions but how much of the work is parallel at all.

### The dominant cause: histogram parallelism is bound by feature count

Histogram construction parallelizes over *feature tiles*, and
`compute_optimal_tile_size` clamps a tile to at least 16 features
(`raw_tile.clamp(16, 64)`). A 40-feature fit therefore yields
`ceil(40/16) = 3` tiles and can never use more than three workers, whatever
`n_jobs` says. Measured speedup tracks the tile count almost exactly:

| Features | Tiles | 1 thread | 10 threads | Speedup |
|---:|---:|---:|---:|---:|
| 16 | 1 | 0.81 s | 0.55 s | 1.49x |
| 40 | 3 | 1.52 s | 0.73 s | 2.08x |
| 160 | 10 | 4.59 s | 1.64 s | 2.80x |
| 320 | 20 | 7.80 s | 2.49 s | 3.13x |

A second confirmation: holding the data fixed and deepening the trees (more
parallel tree work per round against the same fixed serial passes) lifts the
speedup from 1.85x at depth 2 to 2.81x at depth 10.

**This is the concrete architectural difference from the peers.** LightGBM and
XGBoost partition the *rows* into blocks, build a partial histogram per block,
and reduce — so parallelism scales with cores independently of feature count.
CatBoost scales best of all (4.0–4.2x measured) partly for a structural reason:
oblivious trees share one split across every node at a level, so a level is a
single parallel sweep with no per-node fragmentation.

### Secondary: per-round passes over all rows are serial

`objectives/`, `round.rs`, and `loss.rs` contain no Rayon at all, so gradient
computation, prediction updates, and loss are serial O(N) passes every round.
The peers cover exactly these with `omp parallel for`.

## What was fixed

**Thread count could change the trained model (correctness).** Partition
gradient statistics were reduced in chunks whose width came from
`rows.len() / rayon::current_num_threads()`, so the summation order — and
therefore node statistics and leaf values — varied with `n_jobs`. Fits from
roughly 100k rows upward produced genuinely different artifacts at different
thread counts.

This directly contradicted a guarantee published in the README, CHANGELOG,
release notes, and release checklist. It survived because the existing
thread-invariance tests use ~1,000 rows, where the drift stays below the last
mantissa bit of every leaf value.

The chunk width is now a fixed constant, and the fix is performance-neutral
(1 thread 4.96 s -> 4.92 s; 10 threads 2.69 s -> 2.68 s). Two regression tests
were added: a Rust test asserting bit-identical partition statistics at 1/2/4/8
threads on adversarially-scaled gradients, and a Python test at 100k rows —
the smallest size measured to reproduce the bug, since 60k does not.

## What was tried and reverted

A data-parallel histogram kernel (row blocks, per-block partial histograms,
deterministic in-order reduction) plus Rayon over the per-round gradient and
prediction passes. It raised the *speedup ratio* from 1.83x to 1.88x, which
looked like progress, but a clean back-to-back A/B showed it was **slower in
absolute terms on both axes**:

| Variant | 1 thread | 10 threads | Speedup |
|---|---:|---:|---:|
| Baseline | 4.96 s | 2.71 s | 1.83x |
| With changes | 5.54 s | 2.95 s | 1.88x |

The ratio improved only because the single-threaded baseline got slower — the
classic speedup-metric trap. Root causes: block-partial reduction adds
`blocks x tile_features x bins` extra adds that single-threaded execution does
not need, and AlloyGBM's cross-`n_jobs` determinism guarantee forces both the
serial and parallel paths to share one summation order, so that cost cannot be
skipped when running on one thread. Reverted rather than shipped.

## What a real fix needs

The prototype was too shallow. Doing this properly means:

1. **Reducing the reduction.** Accumulate per-block partials in a layout that
   makes the fold cheap (or fold pairwise during collection) so the extra work
   is proportional to `blocks x bins`, not to a full arena pass per block.
2. **Not paying it single-threaded.** Either make the fold cheap enough to
   disappear into noise, or revisit whether byte-identical-across-`n_jobs` is
   the right guarantee. No peer library offers it — LightGBM, XGBoost, and
   CatBoost all permit thread-count-dependent floating-point results. It is a
   genuine differentiator, but it is also what forecloses the cheapest version
   of this optimization, and that trade should be made deliberately.
3. **Removing the tile-width floor.** The 16-feature minimum exists for cache
   behaviour; with row-block parallelism carrying the parallelism, tiles can be
   sized purely for L2 residency.
4. **Parallelising the per-round passes**, which is straightforward once the
   histogram path is settled and is worth roughly the remaining serial share.

Each step needs its own A/B against a *simultaneously measured* baseline, on
both thread counts, with absolute times — not speedup ratios.
