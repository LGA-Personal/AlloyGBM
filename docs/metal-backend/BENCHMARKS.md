# Metal Backend — Throughput Benchmark Reference

Reproduce with:

```bash
.venv/bin/python benchmarks/metal_histogram.py \
    --json-out docs/metal-backend/metal_histogram_<host>.json
```

## 2026-04-20 — Apple M4 (Stage 1 baseline)

- Host: Apple M4 (no Metal 4 support, Apple7 baseline only)
- `AlloyGBM` build: `charming-carson-d08c9a` with the Stage 1
  Metal backend through S1.13. Only `build_histograms` runs on
  the GPU; split finding, partitioning, and prediction all still
  execute on the embedded `CpuBackend`.
- Settings: `n_estimators=5`, `continuous_binning_max_bins=255`
  (u8 bin path), `deterministic=True`, `seed=7`,
  `float32` dense inputs, `--memory-budget-gb 8`.

| rows      | features | input MiB | cpu     | metal   | speedup | note |
|----------:|---------:|----------:|--------:|--------:|--------:|:-----|
|    10,000 |       10 |         0 |     7ms |   143ms |   0.05x |      |
|    10,000 |      100 |         4 |    36ms |   438ms |   0.08x |      |
|    10,000 |    1,000 |        38 |   127ms |   4.13s |   0.03x |      |
|   100,000 |       10 |         4 |    39ms |   231ms |   0.17x |      |
|   100,000 |      100 |        38 |   132ms |   2.14s |   0.06x |      |
|   100,000 |    1,000 |       381 |   782ms |   18.2s |   0.04x |      |
| 1,000,000 |       10 |        38 |   379ms |   1.54s |   0.25x |      |
| 1,000,000 |      100 |       381 |   1.21s |   9.92s |   0.12x |      |
| 1,000,000 |    1,000 |     3,815 |   14.9s |   86.8s |   0.17x |      |

### Interpretation

Stage 1 whole-fit wall-clock is uniformly **slower** on Metal
across this grid. This is consistent with the expert-session
expectation (`/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md`
§Context): histogram acceleration alone only pays off once the
histogram phase dominates the inner loop. In Stage 1 each
boosting round still round-trips through the CPU path for split
finding + partitioning, and the Metal dispatch + readback latency
dominates at these sizes.

The decisive win will land with Stage 2 (GPU best-split) + Stage 3
(GPU row partitioning + Metal 4 ICBs), which eliminate the
per-level CPU round-trip. Until then, the default `device="cpu"`
recommendation in `docs/limitations.md` stands for every shape in
this grid.

Stage 1's value is therefore **infrastructure**, not throughput:
it proves the plumbing (bit-exactness, warn-and-fallback, device
plumbing, capability probe, caching) and unblocks Stage 2.

### What to check at 10M rows

The plan-spec grid includes 10M × (10, 100, 1000). The 10M × 1000
corner alone is ~40 GB of float32 storage, above the default
`--memory-budget-gb 8` guard; re-run with `--full --memory-budget-gb 64`
on a 64 GB host to fill in that corner. Expect the crossover still
not to happen in Stage 1 — that would contradict the expert
prediction and deserves investigation.
