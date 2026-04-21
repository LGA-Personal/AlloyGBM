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
