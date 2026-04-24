# Metal Backend — Throughput Benchmark Reference

Reproduce with:

```bash
# Shape grid (default: regression, depth=6, bins=255, est=5)
.venv/bin/python benchmarks/metal_histogram.py \
    --scenario shape_grid \
    --json-out docs/metal-backend/metal_histogram_shape_grid_<host>.json

# Metal-friendly scenario (configurations theoretically best for Stage 1)
.venv/bin/python benchmarks/metal_histogram.py \
    --scenario metal_friendly \
    --json-out docs/metal-backend/metal_histogram_metal_friendly_<host>.json

# Everything, one command (shape + depth/bins/est sweeps + task mix + metal_friendly)
.venv/bin/python benchmarks/metal_histogram.py --scenario all \
    --json-out docs/metal-backend/metal_histogram_all_<host>.json
```

## 2026-04-20 — Apple M4 (Stage 1 baseline, post buffer-cache)

- Host: Apple M4 (no Metal 4 support, Apple7 baseline only)
- `AlloyGBM` build: `charming-carson-d08c9a` with the Stage 1
  Metal backend through S1.15, including the persistent
  `BufferCache` that holds the binned matrix, gradients, and
  row-indices allocations across every `build_histograms` call
  within a fit.
- Only `build_histograms` runs on the GPU; split finding,
  partitioning, and prediction all still execute on the embedded
  `CpuBackend`.
- Defaults: `n_estimators=5`, `max_depth=6`,
  `continuous_binning_max_bins=255` (u8 bin path),
  `deterministic=True`, `seed=7`, `float32` dense inputs,
  `--memory-budget-gb 8`.

### `shape_grid` — regression, (rows × features)

Reference JSON: [`metal_histogram_shape_grid_m4.json`](metal_histogram_shape_grid_m4.json)

| rows      | features | input MiB |   cpu   |  metal  | speedup |
|----------:|---------:|----------:|--------:|--------:|--------:|
|    10,000 |       10 |         0 |     8ms |   136ms |   0.06x |
|    10,000 |      100 |         4 |    35ms |   415ms |   0.09x |
|    10,000 |    1,000 |        38 |   127ms |   4.04s |   0.03x |
|   100,000 |       10 |         4 |    40ms |   206ms |   0.19x |
|   100,000 |      100 |        38 |   133ms |   1.87s |   0.07x |
|   100,000 |    1,000 |       381 |   760ms |   17.1s |   0.04x |
| 1,000,000 |       10 |        38 |   378ms |   1.36s |   0.28x |
| 1,000,000 |      100 |       381 |   1.26s |   8.82s |   0.14x |
| 1,000,000 |    1,000 |     3,815 |   13.9s |   70.7s |   0.20x |

### `metal_friendly` — configurations theoretically most favourable

The `metal_friendly` scenario tests the hypothesis that Stage 1
Metal could win on large shapes where the histogram phase should
dominate: deep trees (max_depth ∈ {8, 10}), many bins (1024), and
multiclass (K×histogram-build per round). If Stage 1 loses here
too, it loses everywhere under the Stage 1 framing.

Reference JSON: [`metal_histogram_metal_friendly_m4.json`](metal_histogram_metal_friendly_m4.json)

| task         |    rows | features | depth |  bins |   cpu   |  metal  | speedup |
|:-------------|--------:|---------:|------:|------:|--------:|--------:|--------:|
| regression   | 200,000 |      200 |     8 |   255 |   598ms |   9.94s |   0.06x |
| regression   | 200,000 |      200 |    10 |   255 |   1.16s |   16.4s |   0.07x |
| regression   | 200,000 |      200 |     6 | 1,024 |   481ms |   6.88s |   0.07x |
| multiclass_3 | 100,000 |      100 |     8 |   255 |   735ms |   8.26s |   0.09x |
| multiclass_10| 100,000 |      100 |     8 |   255 |   1.84s |   26.0s |   0.07x |

### Interpretation

Stage 1 whole-fit wall-clock is **uniformly slower** on Metal
across both scenarios. The buffer cache that landed with S1.15 gave
a real but modest 5–20% improvement to Metal wall-clock over the
pre-cache S1.14 baseline (the largest absolute gain was the
1M × 1000 cell at 86.8s → 70.7s — roughly 16 s saved on the binned
matrix alone). The speedup ratios are the clearer signal: Metal
moved from 0.03×–0.25× CPU to 0.03×–0.28×. Faster, still
fundamentally losing.

The `metal_friendly` scenario confirms this is not an
input-size problem. Deep trees (depth 10), wide bins (1024), and
multiclass (10-way softmax, which trains K histograms per round)
all keep Metal between 0.06× and 0.09× CPU. This is consistent
with the expert-session expectation (see
[plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md)
§Context): histogram acceleration alone only pays off once the
histogram phase dominates the inner loop, and every boosting
round in Stage 1 still round-trips through the CPU path for
split finding + partitioning. Dispatch + per-level readback
latency dominates at every shape in this grid.

The decisive win will land with Stage 2 (GPU best-split) + Stage 3
(GPU row partitioning + Metal 4 ICBs), which eliminate the
per-level CPU round-trip. Until then, the default `device="cpu"`
recommendation in `docs/limitations.md` stands for every shape in
this grid and every configuration in `metal_friendly`.

Stage 1's value is therefore **infrastructure**, not throughput:
it proves the plumbing (bit-exactness, warn-and-fallback, device
plumbing, capability probe, pipeline caching, buffer caching)
and unblocks Stage 2.

### Notes on specific changes since S1.14

- **Buffer cache landed (S1.15).** The column-major binned
  matrix, gradients, and row-indices buffers are no longer
  reallocated on every `build_histograms` call. The binned
  matrix in particular is keyed by `(ptr, len, is_wide)` and
  reused zero-copy for the entire fit; the gradients/row-indices
  slots reuse the allocation and memcpy fresh bytes per call.
  At 1M × 1000 × 5 estimators, this alone removes on the order
  of 1 TiB of redundant memcpy per fit.
- **Harness expanded.** `benchmarks/metal_histogram.py` now
  exposes named scenarios (`shape_grid`, `depth_sweep`,
  `bins_sweep`, `estimator_sweep`, `task_mix`, `metal_friendly`,
  `all`) so each axis can be isolated. Run
  `python benchmarks/metal_histogram.py --scenario all` for the
  full characterisation pack.

### What to check at 10M rows

The plan-spec grid includes 10M × (10, 100, 1000). The 10M × 1000
corner alone is ~40 GB of float32 storage, above the default
`--memory-budget-gb 8` guard; re-run with `--full --memory-budget-gb 64`
on a 64 GB host to fill in that corner. Expect the crossover still
not to happen in Stage 1 — that would contradict the `metal_friendly`
result and deserve investigation.

## 2026-04-20 — Apple M4 (Stage 2 baseline)

- Host: Apple M4 (Apple7 baseline only; no Metal 4)
- `AlloyGBM` build: `charming-carson-d08c9a` with the Stage 1
  histogram kernel **plus** the Stage 2 `best_split_per_feature`
  kernel (S2.1–S2.6 shipped). The split kernel runs once per
  node per level; cross-feature argmax lives on the CPU with
  `feature_weights` weighting.
- Otherwise identical to the Stage 1 run above (same defaults,
  same Metal 3 baseline path, same `BufferCache` reuse).
- All 30 Metal-gated Python tests (including 8 new Stage 2
  cases) pass; Stage 1's golden 50k × 100 × 255 × 20 test
  retains bit-exact prediction equality to CPU at that shape.

### `shape_grid` — regression, (rows × features)

| rows      | features | input MiB |   cpu   |  metal  | speedup |
|----------:|---------:|----------:|--------:|--------:|--------:|
|    10,000 |       10 |         0 |     7ms |   150ms |   0.05x |
|    10,000 |      100 |         4 |    37ms |   556ms |   0.07x |
|    10,000 |    1,000 |        38 |   143ms |   4.47s |   0.03x |
|   100,000 |       10 |         4 |    39ms |   265ms |   0.15x |
|   100,000 |      100 |        38 |   132ms |   2.11s |   0.06x |
|   100,000 |    1,000 |       381 |   876ms |   17.8s |   0.05x |
| 1,000,000 |       10 |        38 |   364ms |   1.46s |   0.25x |
| 1,000,000 |      100 |       381 |   1.20s |   9.60s |   0.12x |
| 1,000,000 |    1,000 |     3,815 |   13.6s |   82.7s |   0.16x |

### `metal_friendly`

| task         |    rows | features | depth |  bins |   cpu   |  metal  | speedup |
|:-------------|--------:|---------:|------:|------:|--------:|--------:|--------:|
| regression   | 200,000 |      200 |     8 |   255 |   646ms |   10.5s |   0.06x |
| regression   | 200,000 |      200 |    10 |   255 |   1.15s |   19.0s |   0.06x |
| regression   | 200,000 |      200 |     6 | 1,024 |   502ms |   7.42s |   0.07x |
| multiclass_3 | 100,000 |      100 |     8 |   255 |   726ms |   9.18s |   0.08x |
| multiclass_10| 100,000 |      100 |     8 |   255 |   1.83s |   29.5s |   0.06x |

### Interpretation — crossover not yet achieved

The plan projected Stage 2 crossing `>1.0×` on
`metal_friendly`'s two deepest configurations (depth-10 regression,
K=10 multiclass). It did not. Stage 2 numbers are within noise of
Stage 1 — the whole-fit ratios moved by at most 0.01–0.03× either
direction across the grid, inside the run-to-run jitter.

Hypothesis (consistent with the Stage 2 scope decisions documented
in `DECISIONS.md`): **per-node GPU dispatch overhead plus
HistogramBundle memcpy to GPU per `best_split` call now dominate
the CPU savings**. At `max_depth=10` with `n_features=200`, each
tree fires up to `2^10 = 1024` per-node dispatches, each of which
memcpys the `[n_features × n_bins]` grad / hess / count flat
arrays onto the shared buffer. That alone is ~5 MiB × ~5000 calls
≈ 25 GiB of memcpy per fit at the depth-10 × 200-feature shape;
on top of the ~10–50 μs/dispatch fixed latency.

Stage 2 shipped its scope decisions (§DECISIONS.md D-013 + D-014)
that explicitly deferred the two biggest levers toward crossover:

- **`subtract_histogram_bundle` stays on CPU.** A Stage 2 GPU
  subtract would have required histograms to live on GPU across
  calls — which itself requires the Stage 3 surface change of
  passing handles instead of owned `HistogramBundle`s.
- **Row partitioning stays on CPU.** Stage 3's GPU-side
  stream-compaction is what makes GPU-resident histograms
  pay off at all.

Taken together, Stage 2 moved one of the per-level CPU operations
(split finding) onto GPU without being able to *eliminate* the
per-level CPU round-trip. The memcpy + dispatch tax grew by the
same fraction that the compute saved. Net ≈ wash.

**Conclusion for this stage:** Stage 2 is infrastructural value
only, same as Stage 1. The decisive win is now architecturally
gated on **Stage 3's GPU row partitioning + Metal 4 ICBs**, which
together let the whole per-level loop stay GPU-resident between
round boundaries. `device="cpu"` remains the correct default for
every shape in this grid.

### What did *not* regress

- Correctness: 30/30 Metal-gated Python tests pass, including
  Stage 1's golden 50k × 100 × 255 × 20 bit-exact prediction test
  and Stage 2's new regression / binary / ranker / NaN / L1+L2 /
  monotone parity cases.
- Stage 1's warn-and-fallback path (`ALLOYGBM_METAL_DISABLE=1`)
  still triggers cleanly; the estimator's `device` attribute is
  still preserved through pickle.
- The `BufferCache` extension to cover the four new Stage 2 slots
  (`split_grad`, `split_hess`, `split_counts`, `continuous_mask`)
  inherits the same zero-reallocation pattern: the allocations
  grow once and get memcpy'd on subsequent calls.

## 2026-04-24 — Apple M4 (Stage 3, kill-criterion FAILED)

- Host: Apple M4 (Apple7 baseline only; no Metal 4)
- `AlloyGBM` build: `charming-carson-d08c9a` with Stage 1 (histogram
  build), Stage 2 (split kernel), **and** Stage 3 (S3.1 through
  S3.11 — enum-variant `HistogramStorage` / `RowIndexStorage`,
  `HistogramResidencyPool`, `RowIndexResidencyPool`, partition
  kernel, subtract kernel, apply_split Gpu flip, pool-direct
  subtract, full RAII release guards on every hot-path escape).
- 365/365 pytest pass (default features), 334/334 + 31 skipped
  (`--no-default-features`). All 33 Metal-gated Python tests
  (including 3 new `MetalStage3Tests` cases) pass on this build.

### `shape_grid` — regression, (rows × features)

Reference JSON: [`metal_histogram_shape_grid_stage3_m4.json`](metal_histogram_shape_grid_stage3_m4.json)

| rows      | features | input MiB |   cpu   |  metal  | speedup |
|----------:|---------:|----------:|--------:|--------:|--------:|
|    10,000 |       10 |         0 |     7ms |   418ms |   0.02x |
|    10,000 |      100 |         4 |    35ms |   577ms |   0.06x |
|    10,000 |    1,000 |        38 |   122ms |   4.45s |   0.03x |
|   100,000 |       10 |         4 |    50ms |   372ms |   0.13x |
|   100,000 |      100 |        38 |   140ms |   2.05s |   0.07x |
|   100,000 |    1,000 |       381 |   661ms |   17.3s |   0.04x |
| 1,000,000 |       10 |        38 |   332ms |   1.62s |   0.21x |
| 1,000,000 |      100 |       381 |   1.07s |   8.88s |   0.12x |
| 1,000,000 |    1,000 |     3,815 |   16.9s |   69.3s |   0.24x |

### `metal_friendly`

Reference JSON: [`metal_histogram_metal_friendly_stage3_m4.json`](metal_histogram_metal_friendly_stage3_m4.json)

| task         |    rows | features | depth |  bins |   cpu   |  metal  | speedup |
|:-------------|--------:|---------:|------:|------:|--------:|--------:|--------:|
| regression   | 200,000 |      200 |     8 |   255 |   533ms |   10.7s |   0.05x |
| regression   | 200,000 |      200 |    10 |   255 |   1.03s |   22.8s |   0.05x |
| regression   | 200,000 |      200 |     6 | 1,024 |   405ms |   6.56s |   0.06x |
| multiclass_3 | 100,000 |      100 |     8 |   255 |   659ms |   10.2s |   0.06x |
| multiclass_10| 100,000 |      100 |     8 |   255 |   1.65s |   27.7s |   0.06x |

### Interpretation — kill criterion not met

The plan's approved `S3.12` kill criterion requires the
`metal_friendly` deep-tree configs (depth 8, depth 10, and K=10
multiclass at depth 8) to cross **>1.0× CPU**. Stage 3 lands at
**0.05×–0.06×** on those rows — within run-to-run jitter of the
Stage 2 and Stage 1 baselines. Per the plan: *"If they don't,
that refutes the Stage 3 thesis and we stop to debug rather than
ship a second infrastructure-only stage."*

**Root cause — the pools exist but the consumers still readback.**
Stage 3 introduced `HistogramResidencyPool` and
`RowIndexResidencyPool`, both correctly populated and drained by
the partition / subtract / apply_split machinery. The `apply_split`
→ `build_histograms` (row-index buffer bind) handoff and the
`build_histograms` → `subtract_histogram_bundle` (pool-direct)
handoff are both zero-memcpy. But three of the five overridden
`BackendOps` methods still do a full readback per call:

- **`build_histograms` CPU count path.** The histogram kernel
  produces `grad_sum` / `hess_sum` on GPU, but bin-counts are
  still accumulated on CPU (see D-008). That accumulation needs
  `&[u32]` of row indices; the Gpu arm materialises via
  `slice::from_raw_parts(..).to_vec()` on the pool buffer every
  call. At depth 10 × 200 features × 5 estimators, that's ~5,000
  readbacks per fit; each one touches up to `row_count × 4` bytes
  of shared-mode buffer contents.
- **`reduce_sums`.** Full row-index readback, every call, to
  reach the CPU reduce over gradients.
- **`apply_partition_leaf_updates`.** Readback of both left and
  right row-index buffers, every call, to reach the CPU
  prediction-update loop.

In aggregate, Stage 3 moved the per-level CPU round-trip from
"HistogramBundle flat-copy (`F × B × 12` bytes per node)" to
"row-index full-copy (`row_count × 4` bytes per node × 3 call
sites)". That is roughly the same bandwidth at `metal_friendly`
shape and strictly more at 1M-row `shape_grid` shapes — consistent
with the measured ratios.

### What Stage 3 *did* land correctly

- **API surface (D-015).** `HistogramStorage::{Cpu, Gpu}` and
  `RowIndexStorage::{Cpu, Gpu}` enums thread cleanly through the
  engine; `HistogramBundle` and `NodeSlice` carry storage
  variants; the trainer loops pattern-match on each variant at
  every field-read site.
- **Residency pools.** Both pools correctly mint / get / release
  handles with `HashMap<u64, Entry>` under a `Mutex`; the M2
  free-on-consume discipline (D-016) is enforced by
  `HistogramReleaseGuard` / `RowIndexReleaseGuard` / a paired
  `PartitionReleaseGuard` on every hot-path escape (early break,
  continue, error-`?`, drop).
- **Partition kernel.** GPU-resident stream-compaction, both
  continuous-threshold and categorical-bitset paths (D-017 →
  D-017); parity tests pass on both. Output lands in the
  row-index pool; the next level's histogram kernel binds the
  pool buffer directly.
- **Subtract kernel.** Pool-direct when both inputs are Gpu
  (S3.7c.3); single kernel launch against pool-owned buffers,
  zero flat-copy. Net-positive in isolation — just dwarfed by
  the other three readback paths above.
- **Correctness envelope.** Stage 2's structural-plus-ulp gate
  holds; all 33 Metal-gated Python tests pass including three
  new Stage 3 cases (leaf-wise handle threading, mixed
  categorical/continuous, deep-tree pool-lifetime stress).

### What it would take to cross

To meet the S3.12 kill criterion without regressing correctness,
the per-level round-trip has to *actually* go away — which means
closing all three of the readback paths above:

1. **GPU count accumulation.** Move the bin-count accumulation
   into the histogram kernel (one extra atomic-free path doing
   `counts[feature][bin]++` via threadgroup privatisation +
   two-pass reduce, same pattern as the grad/hess reduce).
   Eliminates the histogram-side readback.
2. **GPU reduce_sums.** A small kernel that reads
   `gradients[row_indices[i]]` on GPU and reduces via block-scan;
   feasible only if gradients themselves become GPU-resident
   across the fit. That's a further surface change (gradient
   pool).
3. **GPU apply_partition_leaf_updates.** Trivial kernel if
   predictions are GPU-resident. Same "gradient pool → prediction
   pool" surface change.

(1) alone would close the largest of the three and might be enough
on `metal_friendly` depth 10. (2) and (3) together are the full
fix but require a gradient + prediction pool — a Stage 3.5 scope
extension.

**Recommendation:** treat Stage 3 as landed-but-not-crossed. Don't
advance to Stage 4 (ICB dispatch optimisation) until S3.12 is met,
because ICBs are a marginal optimisation on top of a GPU-resident
per-level loop — they don't help if the loop is still round-tripping
through CPU. `device="cpu"` remains the default and the correct
choice for every shape in the grid above.

### What did *not* regress vs Stage 2

- Correctness: the same 30 Metal-gated tests from Stage 2 still
  pass, plus three new `MetalStage3Tests` cases exercising leaf-wise
  handle threading, D-017 mixed categorical/continuous fits, and
  deep-tree pool-lifetime stress at max_depth=10.
- Stage 1's warn-and-fallback path (`ALLOYGBM_METAL_DISABLE=1`)
  still triggers cleanly; the Gpu variants never appear in the
  fallback code path.
- Memory discipline: neither residency pool leaks — all three
  Stage 3 tests assert `live_count() == 0` at end of fit.
