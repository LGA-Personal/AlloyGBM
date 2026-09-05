# Cross-Machine Portability of Parallel Tuning

| Date | Reviewer | Version | Commit | Status |
|---|---|---|---|---|
| 2026-09-02 | Claude Fable 5 | v1.0.0-prep | `1e2cf32` | One fix landed; one structural gap open |

Motivating question: AlloyGBM's parallel constants were tuned on one machine
(Apple M4, 4 performance + 6 efficiency cores). LightGBM is fast across very
different hosts. What in LightGBM, XGBoost, and CatBoost makes them adapt, what
of it transfers, and what does not?

Peer sources are not vendored in the installed wheels, so this reads their
published source and anchors conclusions to measurements of *our* code.

## What the peers actually do

### LightGBM: measure, don't predict

`Dataset::GetShareStates` does not use a formula to choose between its
column-wise and row-wise histogram strategies. When neither `force_col_wise`
nor `force_row_wise` is set, it **builds histograms both ways at training
start, times each with `steady_clock`, and keeps the winner** — the overhead is
reported to the user as part of the decision.

This is the single most important idea here, and it is why LightGBM is fast on
machines its authors never touched: it does not need a model of the host,
because it measures the host.

### LightGBM: block counts driven by threads, floored by work

`Threading::BlockInfo` is used throughout:

```
nblock     = min(num_threads, ceil(cnt / min_cnt_per_block))
block_size = SIZE_ALIGNED(ceil(cnt / nblock))
```

The thread count is the *primary* term. A minimum granularity can only ever
reduce the block count below the thread count when there genuinely is not
enough work. It never caps the block count for any other reason.

### LightGBM: row-wise parallelism is not bounded by feature count

The row-wise path (`MultiValBinWrapper`) blocks over **data rows**, giving each
block its own histogram buffer (`hist_buf_ptr + num_bin_aligned * block_id * 2`)
and merging afterwards. Since `cnt` there is the row count, `BlockInfo` returns
`min(num_threads, huge)` — the pool always saturates, whatever the feature
count.

### XGBoost: container-aware thread counts, nested 2-D partitioning

`GetCfsCPUCount()` reads the Linux CFS quota, because `hardware_concurrency()`
reports the *host's* cores inside a container with a CPU limit.
`BlockedSpace2d` partitions (node, rows-within-node) together so an unbalanced
level still fills the pool, rather than parallelising over nodes alone.

### CatBoost

Oblivious trees share one split across a whole level, so a level is a single
parallel sweep with no per-node fragmentation. This is a consequence of the
model class, not a scheduling technique — not transferable without adopting
oblivious trees.

## What we already have

**Container and affinity awareness.** `resolve_fit_thread_count` uses
`std::thread::available_parallelism()`, which accounts for cgroup v1/v2 quotas
and `sched_getaffinity` masks. This is the equivalent of XGBoost's
`GetCfsCPUCount()` and we get it from the standard library. It is documented to
possibly overcount when those interfaces cannot be queried (e.g. under
sandboxing), which is the same failure mode XGBoost has. It is called once per
fit, not in hot code, which is what its documentation asks.

**Thread-count-driven nesting.** The level builder already consults
`rayon::current_num_threads()` to decide whether to nest tile parallelism under
node parallelism, rather than using a fixed node-count threshold.

## Fixed: the tile-width floor re-created its own ceiling

`compute_optimal_tile_size` clamped tile *width* to a 4-feature minimum, which
is a cap on tile *count* of `feature_count / 4`. Expressed in LightGBM's terms,
we were capping `nblock` for a reason other than insufficient work:

| Host threads | Features | Tiles before | Starved? |
|---:|---:|---:|---|
| 10 | 40 | 10 | no |
| 32 | 40 | 10 | **yes** |
| 64 | 40 | 10 | **yes** |
| 128 | 320 | 80 | **yes** |

The floor is a cache preference, not a correctness rule, so it now yields when
honouring it would leave threads with no tile. This is a no-op on the tuning
machine for every shape measured, which is exactly why it was invisible.

Pinned by `auto_tile_size_never_caps_tiles_below_thread_count`, which sweeps
thread counts to 256 — none of them reachable on the development host. That is
the point: this class of bug cannot be caught by benchmarking on one machine.

## Open: feature-count-bound histogram parallelism

Even with the floor fixed, 40 features yield at most 40 tiles. Our histogram
parallelism is bounded by feature count; LightGBM's row-wise path is bounded by
row count. On a wide machine with narrow data — 64+ cores, tens of features —
that is a hard ceiling we cannot tile our way out of.

Closing it means row-block partial histograms. The earlier
[parallel scaling investigation](2026-09-01-parallel-scaling-investigation.md)
tried this and reverted it, but for an implementation reason (up to 31 blocks
with per-block arena allocation) rather than a fundamental one. With a small
block count and reused arenas the reduction is `blocks x tile_features x bins`
additions against `rows x tile_features` accumulations — well under 1% at
realistic sizes.

### Where LightGBM's approach does *not* transfer

**Timing-based strategy selection is incompatible with our determinism
guarantee.** Row-blocked and column-wise accumulation sum floats in different
orders, so they produce different models. If we chose between them by
stopwatch, two runs on the same machine could disagree — a far worse property
than the one we would be buying.

Measured, rather than assumed. Fitting each library at `n_jobs` 1, 4 and 8 on
400k x 30 with a heavy-tailed target (a few values 10^6 larger, so any change
in accumulation order surfaces instead of being absorbed), and taking the
maximum absolute prediction difference against the single-threaded fit:

| Library | n_jobs=4 vs 1 | n_jobs=8 vs 1 |
|---|---:|---:|
| AlloyGBM | **0** | **0** |
| XGBoost (`hist`) | **0** | **0** |
| LightGBM (auto) | 1.91e-11 | 1.91e-11 |
| LightGBM (`force_row_wise`) | 1.91e-11 | 1.91e-11 |
| LightGBM (`force_col_wise`) | 1.91e-11 | 1.91e-11 |

So LightGBM's results *do* drift with thread count, in every mode — but by
~1e-11 on predictions of order 1e6, which is around 1e-17 relative and of no
practical consequence. Its float64 histograms keep the drift far below
anything a user would notice. Only on a benign target does it vanish entirely,
which is why the earlier check at 6-decimal RMSE showed nothing.

**The important correction is XGBoost's column.** It is bit-exact across thread
counts *and* it is the fastest library in most cells of the deep scaling sweep.
That is the refutation of the framing carried over from the earlier
investigation, which treated the determinism guarantee as the thing
foreclosing competitive scaling. XGBoost demonstrates that exactness and speed
are not in tension. Our gap is not the price of determinism; it is a
throughput gap we have not yet closed.

There is a version that fits us, and it is stronger than LightGBM's:

- block over rows with a block count derived from the **row count alone**,
  never from the thread count;
- use the same blocking for serial and parallel execution.

Then results are identical across `n_jobs` *and* across machines while
parallelism stops depending on feature count. The cost is that the reduction
always runs, including single-threaded.

This changes every model artifact once, because summation order changes. That
is acceptable before 1.0.0 and expensive after it, so it is a decision to take
deliberately rather than a change to slip in.

## Tried and reverted: caching the fit thread pool

Each fit builds a fresh Rayon pool, spawning one OS thread per worker; LightGBM
and XGBoost reuse a persistent OpenMP pool. The hypothesis was that this taxes
many-core hosts, since the cost scales with worker count and is paid per fit —
once per split in a cross-validation loop.

Implemented with a single-entry cache and a pid guard (Rayon workers do not
survive `fork()`, and fitting before a `multiprocessing`/loky fork is ordinary
usage). Measured on 20-40 short fits:

| Workers | Rebuilt per fit | Cached |
|---:|---:|---:|
| 4 | 20.0 ms/fit | 20.0 ms/fit |
| 10 | 22.8 ms/fit | 22.8 ms/fit |
| 64 | 38.0 ms/fit | 37.6 ms/fit |
| 128 | 52.2 ms/fit | 51.5 ms/fit |

About 1% at 128 workers and nothing at all at realistic ones. Reverted: a
process-wide mutable cache and a fork hazard are not worth an unmeasurable
gain. Worth revisiting only if profiling on a genuinely many-core host shows
pool construction to be material — the numbers above are from a 10-core
machine, where spawning 128 threads may be cheaper than on a large NUMA host.

## Recommendation

1. Landed: the tile-floor fix, which removes a real many-core ceiling at no
   cost on any machine.
2. Decide before 1.0.0 whether to adopt row-count-derived row-block
   histograms. It is the only remaining way to make parallelism independent of
   feature count, it is cheaper than the reverted prototype suggested, and the
   one-time artifact change is much easier to justify now than later.
3. Do not adopt timing-based strategy selection. It is the best idea in
   LightGBM and the one least compatible with what AlloyGBM promises: the
   choice itself would vary run to run, so two fits on one machine could
   disagree. Note this is a stricter bar than LightGBM meets — its own results
   drift with thread count, if only by ~1e-17 relative.
4. Treat single-thread throughput, not scaling, as the priority. The deep
   scaling sweep puts AlloyGBM last in all nine configurations at one thread,
   while our speedup from 1 to 10 threads is the best or second-best in seven
   of them. Scaling work has reached diminishing returns; the per-core inner
   loop has not.
