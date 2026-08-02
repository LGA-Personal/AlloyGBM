# MorphBoost PR #132 Performance And Calibration

| Date | Source base | Candidate | Platform |
|---|---|---|---|
| 2026-08-02 | `77dbf6d` | `codex/morphboost-performance` | Apple M4, macOS 26.5.2, arm64 |

## Scope

PR #132 replaces ordinary post-warmup numeric MorphBoost candidate scoring with an exhaustive safe
SIMD scanner. It still evaluates every valid threshold and both missing-value directions. Warmup
uses the standard scanner exactly. Morph+DRO, factor-penalized Morph, and native categorical Morph
retain scalar scanning. The artifact format and public parameter surface are unchanged.

The semantic repair also computes the L1/DRO-adjusted parent gradient independently rather than
reconstructing it by summing independently adjusted children.

## Scanner timing

Seven release-mode repetitions were aggregated by median. Times are per feature scan.

| Maximum bins | Scalar baseline | SIMD candidate | Speedup |
|---:|---:|---:|---:|
| 16 | 23.77 us | 23.17 us | 1.03x |
| 64 | 25.25 us | 23.35 us | 1.08x |
| 255 | 224.91 us | 117.29 us | 1.92x |

The final verification rerun measured 22.98, 23.66, and 122.82 us respectively; the 255-bin
speedup remained 1.83x. No measured scanner shape regressed by more than 5%.

Across the nine-shape, five-seed native acceptance matrix, paired Morph fit time improved by a
median **30.4%** versus the frozen scalar control. This clears the predefined 15% end-to-end
fallback even though the 64-bin microbenchmark did not reach 1.5x.

The full matrix used five fixed seeds (`0` through `4`) for each case:

| Case | Family | Rows | Features | Rounds | Query count |
|---|---|---:|---:|---:|---:|
| reg-small-narrow | regression | 640 | 8 | 80 | - |
| reg-small-wide | regression | 640 | 128 | 80 | - |
| reg-tall-narrow | regression | 8,192 | 16 | 80 | - |
| reg-tall-wide | regression | 8,192 | 128 | 60 | - |
| reg-noisy-nonlinear | regression | 2,048 | 32 | 100 | - |
| binary-imbalanced | binary | 4,096 | 32 | 100 | - |
| multiclass-wide | multiclass | 2,048 | 96 | 80 | - |
| rank-small-query | ranking | 2,400 | 24 | 60 | 120 |
| rank-large-query | ranking | 4,096 | 24 | 60 | 8 |

At medium scale (200,000 rows, 400 features, 500 rounds), current total/native fit times were:

| Arm | Total fit | Native training |
|---|---:|---:|
| Auto | 23.09 s | 22.38 s |
| Morph | 23.46 s | 22.89 s |
| Morph + cosine | 23.92 s | 23.34 s |

These are candidate absolute timings, not a branch-to-branch speedup claim. The large profile was
omitted because its 500,000 x 780 float32 matrix alone is 1.56 GB before quantized, histogram, and
three-arm working memory. Numerai was not rerun because its external dataset was not present.

## Correctness and quality

Randomized scalar/SIMD tests cover 16, 64, and 255 bins, missing mass, L1, Hessian/row floors,
leaf-magnitude filtering, warmup/post-warmup rounds, balance boundaries, tail lanes, and non-finite
candidates. Candidate gains must satisfy `max(1e-5, 1e-5 * abs(scalar_gain))`; winners, direction,
and child statistics match outside material ties.

Against the frozen current-formula control, optimized quality had median change `0.0`, 97.8%
practical wins/ties, worst paired change `-0.18%`, and no task-family veto. The final five-seed
auto-versus-Morph matrix reproduced the committed decision exactly. It also confirms that Morph is
not universally more accurate: synthetic regression mean was materially below auto, while ranking
was approximately neutral/slightly positive. Users should A/B the mode on their validation data.

The public 60-round report likewise remained mixed: Morph trailed auto on California housing,
breast cancer, wine, and California ranking, but improved digits accuracy from 0.9611 to 0.9694.

## Formula and default decisions

- Raw gradients for the information term were rejected after 78 regularized/DRO paired fits:
  mean `-1.53%`, median `-0.30%`, worst pair `-21.99%`.
- Disabling balance was rejected: mean `-3.48%`, regression-family mean `-6.19%`.
- Information weights 0.05, 0.075, and 0.15 were rejected. Weight 0.075 was strongest but missed
  the predefined worst-pair bound (`-3.166%` versus `-3.0%`). The 0.10 default remains.
- EMA preparation measured 4.60 us per 8,192-row round (~0.22% of representative round time).
- A 64-category scalar scan measured 5.15 us, with a conservative 5.9% full-tree fit upper bound.
- Joint proxy/exact counts selected the same threshold in 5/5 fractional-Hessian fixtures; an exact
  shared count plane would add 25% histogram payload at two outputs.

Detailed records are committed under `benchmarks/results/pr132_morph_*`.

## Reproduction

```bash
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done

python benchmarks/morph_acceptance.py \
  --arms auto morph_current --seeds 0 1 2 3 4 \
  --output /tmp/pr132_morph_final_verification.json
python benchmarks/morph_report.py --quick --output /tmp/pr132_morph_report.csv
python benchmarks/perf_at_scale.py --scale medium \
  --arms auto morph morph_cosine
```
