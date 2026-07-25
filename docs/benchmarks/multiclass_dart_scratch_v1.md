# Multiclass DART Scratch Evidence (v1)

## Scope

This descriptive benchmark targets the final-review scaling case: 64 classes
with `dart_max_drop=2`. Only a few selected class-trees can contain material
stumps each round, so scratch clearing and finalization should follow those
distinct class slices rather than scanning all `classes * rows` values.

The harness has no timing gate:
`benchmarks/multiclass_dart_scratch_benchmark.py`.

## Command and Environment

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python \
  benchmarks/multiclass_dart_scratch_benchmark.py
```

Captured on 2026-07-25 on macOS 26.5.2, Apple M4, Python 3.13.5:

- 2,048 rows, 16 features, and 64 classes;
- 16 boosting rounds, depth 2, manual deterministic policy, seed 29;
- DART drop rate 0.75 and `dart_max_drop=2`; and
- five repetitions after an unmeasured standard warmup fit.

## Result

| Source | Arm | Timings (s) | Median (s) | DART/standard |
|---|---|---|---:|---:|
| `3db076c` | standard | 0.514299, 0.511306, 0.513132, 0.512089, 0.517032 | 0.513132 | - |
| `3db076c` | DART | 0.515435, 0.521820, 0.517270, 0.522916, 0.523189 | 0.521820 | 1.017x |
| `136bbd7` | standard | 0.517451, 0.509027, 0.514960, 0.509242, 0.512322 | 0.512322 | - |
| `136bbd7` | DART | 0.516011, 0.517545, 0.513991, 0.517825, 0.513506 | 0.516011 | 1.007x |
| `489a02c` + docs | standard | 0.514715, 0.512336, 0.514139, 0.510841, 0.516848 | 0.514139 | - |
| `489a02c` + docs | DART | 0.517823, 0.520845, 0.515611, 0.519126, 0.517869 | 0.517869 | 1.007x |

The initial optimized DART median was 1.1% lower in its paired local capture,
and final-head verification reproduced a `1.007x` DART/standard ratio. These
small deltas are within the range where host load and timer variance can
matter; they are not evidence for a stable speedup percentage. The structural
evidence is the mutation-sensitive Rust coverage that leaves stale untouched
class buffers unchanged while clearing and finalizing only first-seen
material classes. Flat selection order and phantom-tree normalization counts
remain separate and unchanged.
