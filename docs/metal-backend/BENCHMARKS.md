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
