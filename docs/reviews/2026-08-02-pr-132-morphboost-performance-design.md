# PR #132 MorphBoost Performance And Calibration Design

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-08-02 | OpenAI Codex | `main` after PR #131 | `77dbf6d` | Draft for review |

## Objective

Make MorphBoost materially faster and more efficient across dataset shapes while preserving its
adaptive split semantics, then admit narrowly scoped formula or default changes only when broad,
paired A/B evidence demonstrates a reliable quality improvement.

This PR closes section 2.3 of the 2026-07-02 special-modes review. It also repairs the MorphBoost
benchmark guardrails needed to evaluate that closure. The primary implementation remains an
exhaustive split search: every valid numeric threshold and both missing-value directions are
evaluated with the MorphBoost criterion.

## Current Evidence

The original review measured a small regression workload at 0.62 seconds for MorphBoost versus
0.24 seconds for auto mode. Current quick ablations still show roughly 1.5x-2x MorphBoost overhead
on small workloads. Older 200,000-row by 400-feature results, however, placed MorphBoost within
noise of auto mode after histogram and tiling optimizations. The cost is therefore shape-dependent;
the PR must not optimize one small fixture at the expense of tall or wide workloads.

The remaining hot path is the post-warmup numeric split scan. It performs the prefix accumulation
serially and then evaluates the normalized gradient gain, three logarithmic information terms, and
an optional exponential balance adjustment for each threshold and missing direction. The standard
scanner evaluates corresponding candidate arithmetic eight lanes at a time. The installed `wide`
version supports lane-wise `ln` and `exp`, making an exhaustive Morph SIMD scanner practical.

Two supporting issues must be resolved before benchmark conclusions are trusted:

- `benchmarks/morph_ablation.py` passes a truncated query-ID vector as ranking group sizes, so its
  ranking arm currently fails instead of measuring quality;
- its ranking metric is an RMSE proxy rather than a group-aware ranking metric.

The fast Python fingerprint call is a constant dictionary lookup and is not a meaningful fit-time
bottleneck. Morph EMA preparation copies gradients into reusable contiguous scratch before SIMD
moment calculation; this is a plausible secondary cost but will be changed only if profiling shows
a repeatable end-to-end benefit.

## Selected Approach

The PR uses a measurement-led sequence:

1. establish trustworthy scanner and end-to-end baselines;
2. repair documented warmup equivalence;
3. implement exhaustive post-warmup Morph SIMD;
4. profile and conditionally optimize secondary costs;
5. evaluate formula/default candidates one at a time against the optimized current formula;
6. retain only candidates that pass the predefined quality and performance gates.

A standard-gain top-k shortlist is not part of the default design. It can miss the exact candidates
that distinguish MorphBoost from ordinary boosting. It will be reconsidered only as a separate,
explicitly approximate follow-up if exhaustive SIMD fails to recover meaningful scanner time.

## Benchmark Architecture

### Scanner Microbenchmark

The CPU benchmark suite will compare the scalar Morph oracle, the optimized Morph scanner, and the
standard scanner over deterministic histograms with 16, 64, and 255 usable bins. Cases cover:

- balanced and highly skewed bin counts;
- no missing values and substantial missing mass;
- zero and nonzero L1 regularization;
- minimum-row, minimum-Hessian, and minimum-leaf-magnitude filtering;
- near-tied candidate gains;
- warmup, early post-warmup, and late-fit blend coefficients.

Each reported value is the median of repeated runs after warmup. The benchmark must expose
per-feature scan time and selected threshold/direction so speed cannot hide a correctness failure.

### End-To-End Shape Matrix

Deterministic synthetic datasets will cover these independent shape regimes:

| Shape | Representative purpose |
|---|---|
| small/narrow | Python and fixed-cost overhead, shallow histograms |
| small/wide | feature-parallel split-search pressure |
| tall/narrow | histogram throughput and EMA row passes |
| tall/wide | memory bandwidth, tiling, and existing at-scale behavior |
| noisy nonlinear | MorphBoost's intended adaptive-split use case |
| sparse-signal high-dimensional | feature-selection behavior |
| imbalanced classes | balance adjustment and minority structure |
| small and large queries | ranking quality across query-size distributions |

The full quality matrix includes regression, binary classification, multiclass classification,
and LambdaMART ranking. Regression reports RMSE and MAE; classifiers report log loss and accuracy;
ranking reports NDCG@10. Dataset rows, not query IDs, remain contiguous for ranking and train/test
splits occur on query boundaries.

### Benchmark Ladder

Benchmark cost is controlled by promotion stages:

1. **Every edit:** Rust scanner oracle and microbench smoke cases, plus the quick synthetic A/B.
2. **Surviving implementation:** full shape/task matrix with at least five fixed seeds.
3. **Surviving formula/default candidate:** `morph_report.py` on its public datasets.
4. **Final candidate only:** the large `perf_at_scale.py` case and Numerai benchmark when its data
   and dependencies are locally available.

Numerai and peer-library comparisons are corroborating final evidence, not tuning datasets.

## Warmup And Parent-Gain Semantics

MorphBoost documents pure standard-gain split selection before `morph_warmup_iters`. Ordinary
numeric Morph scans without DRO or factor penalties will therefore dispatch directly to the
standard SIMD scanner during warmup. Categorical warmup will likewise use the standard categorical
criterion where no additional mode requires the scalar scaffold.

The gain input will distinguish parent, left-child, and right-child gradient signals. The standard
parent term must use the parent gradient after applying L1 or DRO adjustment once; it must not use
the sum of two independently adjusted child gradients. Tests will pin threshold, missing direction,
gain tolerance, and child statistics against the standard scanner for nonzero L1 and missing-value
cases.

This is a documented-contract correction. Model changes are expected only in configurations where
the previous parent construction diverged, and those changes will still be quantified in the A/B
report.

## Exhaustive SIMD Scanner

The optimized numeric scanner retains the existing scalar cumulative prefix pass. Candidate
evaluation then proceeds in eight-lane chunks for each missing direction:

1. load cumulative gradient, Hessian, and row-count lanes;
2. derive right-side and missing-routed statistics;
3. apply validity and leaf-magnitude masks;
4. compute L1-adjusted child and parent gradient terms;
5. compute normalized Newton gain;
6. compute standardized side/parent information values with lane-wise logarithms;
7. apply the balance adjustment only to lanes below the imbalance threshold;
8. mask non-finite, padded, and edge candidates;
9. reduce candidates with the existing material-gain comparison and deterministic ordering;
10. reconstruct one `SplitCandidate` from the winning prefix entry.

Per-round values such as blend coefficients, inverse smoothing, inverse gradient scale, and
normalization constants are precomputed or broadcast once. Existing thread-local split scratch is
reused; the scanner must not allocate per feature or per candidate.

The scalar Morph implementation remains the semantic oracle. Randomized histogram tests compare
all candidate gains with `abs_error <= max(1e-5, 1e-5 * abs(scalar_gain))` and require identical
winners except when the scalar oracle's top candidates are tied under the existing
`gain_materially_exceeds` rule. Adversarial fixtures cover zero counts, extreme but finite
gradients, tiny Hessians, missing-only bins, and tail chunks shorter than eight lanes.

Morph+DRO and factor-penalized Morph retain their scalar fallback in this PR. They receive parity
and compatibility tests, but their scanner performance belongs to the dedicated DRO and
neutralization work. Native categorical Morph remains scalar unless profiling shows it consumes at
least 10% of representative categorical fit time.

## Secondary Optimization Decisions

Secondary work is conditional rather than assumed:

- **EMA preparation:** optimize only if profiling attributes at least 3% of native fit time to the
  scratch copy/moment pass or a prototype yields a repeatable 3% end-to-end improvement. Reusing
  diagnostics moments is allowed only if EMA update ordering and finite-value behavior are pinned;
  any f32-to-f64 behavior change enters the quality A/B.
- **Categorical scanner:** vectorize only if categorical profiling crosses the 10% threshold and
  the change fits without duplicating the numeric scanner architecture.
- **Python fingerprint:** no performance work. Dead-branch cleanup may be included only if it is
  behavior-neutral and does not expand the public API scope.
- **Joint-output row counts:** evaluate only after the main scanner lands. If joint Morph quality
  reveals a repeatable gap from Hessian-derived count proxies, prototype one shared count plane per
  feature/bin rather than per output. Promote it only if quality improves and histogram memory/time
  remain within the limits below.

## Formula And Default Experiments

Formula candidates run sequentially. Each winner becomes the next control; rejected candidates are
recorded with their result rather than combined into an opaque sweep.

### Experiment A: Information Gradient Source

Compare the current use of L1/DRO-adjusted child gradients in both criterion components with a
separated formulation: raw gradient sums feed the EMA-standardized information term, while
regularized or robust effective gradients feed Newton gain and leaf solving. This better aligns the
information statistic with the raw-gradient EMA but changes trained models.

### Experiment B: Balance Adjustment

Compare the current adjustment with disabled and lower-strength variants. The public boolean
contract remains unless evidence supports a default behavior change; this PR will not add a new
continuous public parameter merely to expose experimental values.

### Experiment C: Information Weight

If A or B is promoted, recalibrate `info_score_weight` over a narrow set centered on the current
default: `0.05`, `0.075`, `0.1`, and `0.15`. The current optimized formula at `0.1` remains the
control. Other Morph parameters are held fixed to avoid an unbounded joint hyperparameter search.

### Experiment D: Joint Counts, Conditional

For joint multi-output Morph only, compare the current Hessian count proxy with exact shared row
counts. This experiment proceeds only if focused joint datasets show that count approximation
changes candidate ordering or quality materially. It is omitted if the evidence is neutral or if
the shared count plane produces disproportionate memory or histogram-build cost.

Learning-rate schedules, morph rate, depth penalty, evolution pressure, tree growth, and estimator
auto-policy heuristics are held fixed. They are separate axes and will not be retuned in this PR.

## Promotion Gates

### Correctness

- all outputs and gains are finite for valid finite inputs;
- warmup matches standard split selection under ordinary Morph configurations;
- scalar and SIMD Morph select the same threshold and missing direction outside material ties;
- child statistics and feature-weight ordering remain unchanged;
- level-wise and leaf-wise growth remain deterministic for a fixed seed and worker count;
- existing artifacts remain loadable; the artifact schema does not change.

### Performance

- median post-warmup numeric scanner speedup is at least 1.5x for the 64-bin and 255-bin cases;
- no scanner fixture regresses by more than 5%;
- scanner-dominated small/medium Morph fits improve by at least 15%;
- no end-to-end shape regresses by more than 5%, including tall/wide workloads already near auto;
- no new per-feature allocation is introduced;
- peak memory grows by no more than 5% for the main scanner work;
- a conditional joint-count plane must report its separate memory delta and stay within 10% on
  representative joint fits.

If exhaustive SIMD misses the 1.5x microbenchmark target, profiling determines whether lane-wise
transcendentals or surrounding scalar work is responsible. The PR may still proceed if it delivers
at least 15% end-to-end improvement with all correctness gates, but it must not silently substitute
an approximate top-k scanner.

### Formula Or Default Quality

For each dataset/seed, define normalized improvement as `(control - candidate) / |control|` for
error metrics and `(candidate - control) / |control|` for higher-is-better metrics. Dataset cases
receive equal aggregate weight regardless of row count. The primary metrics are RMSE for
regression, log loss for binary and multiclass classification, and NDCG@10 for ranking. A practical
tie is an absolute normalized change no greater than 0.1%. The paired bootstrap uses a fixed seed
and resamples dataset/seed pairs 10,000 times.

A formula/default candidate is promoted only when:

- every result is finite;
- the aggregate mean improvement is positive and at least 0.25%;
- median improvement is nonnegative;
- at least 60% of paired cases are wins or practical ties;
- no task-family mean is worse than 0.5%;
- no individual dataset/seed primary metric is worse than 3%;
- a paired bootstrap 95% confidence interval has a lower bound above -0.25%;
- no task family's mean normalized Morph-versus-auto gap worsens by more than 0.5%.

Accuracy, MAE, and secondary metrics act as vetoes for pathological tradeoffs even when the primary
metric passes. Thresholds may be tightened before experiments begin but are not relaxed after
results are observed.

## Test Matrix

Rust coverage includes:

- scalar/SIMD candidate-gain and winner parity over deterministic randomized histograms;
- warmup parity with standard numeric and categorical scanners;
- L1, missing routing, minimum-row/Hessian/leaf filters, feature weights, and ties;
- early/late Morph coefficients and balance-penalty boundaries;
- scalar fallback coverage for DRO and factor penalties;
- benchmark fixtures at all target bin counts.

Python coverage includes:

- repaired regression, binary, multiclass, and ranking ablations;
- both tree-growth strategies and deterministic repeated fits;
- missing values and native categorical features;
- quantile regression, GOSS, DART, warm start, and validation early stopping;
- Morph+DRO and Morph+PL compatibility smoke tests without claiming their dedicated speedups;
- benchmark gate unit tests, including deliberate rejected candidates.

The final verification gate is:

- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace`;
- the complete Python suite;
- benchmark contract tests;
- scanner microbenchmarks and the full local A/B matrix;
- Sphinx with warnings treated as errors.

## Documentation And Review Closure

The PR updates the MorphBoost user guide, benchmark documentation, changelog, Sphinx mirror, and
`2026-07-02-v0.12.10-special-modes-resolutions.md`. The resolution records before/after scanner and
fit timings, the final quality matrix, any formula/default changes, and rejected experiments.

Performance claims distinguish microbenchmark speedup from end-to-end fit speedup and identify the
tested hardware. If no formula/default candidate passes, the PR explicitly states that current
training behavior was retained apart from the warmup correctness repair and expected SIMD
floating-point differences.

## Compatibility And Non-Goals

The artifact format, predictor, loaded-model behavior, and public estimator classes remain
compatible. Newly trained Morph artifacts may differ because exhaustive SIMD changes floating-point
evaluation order, the warmup parent correction changes affected regularized fits, or a promoted
formula/default candidate intentionally changes split selection. Determinism is required within a
fixed build, platform, seed, and worker configuration; byte equivalence with prior Morph artifacts
is not required.

This PR does not implement DRO SIMD, PL top-k histogram construction, GPU MorphBoost, top-k split
approximation, new learning-rate schedules, broad auto-policy recalibration, or a general MorphBoost
hyperparameter tuner. Those changes require separate attribution and acceptance evidence.
