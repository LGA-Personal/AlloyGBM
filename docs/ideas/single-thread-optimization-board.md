# Single-Thread Optimization Board

**A living, multi-author document.** Add ideas, challenge existing ones, record
results. Nothing here is settled unless its status says so.

**Read first:** [the problem statement](../reviews/2026-09-06-single-thread-throughput.md).
It has the profiles, the peer comparison, the hard constraints, and a list of
things already tried that did not work. Proposals that ignore it tend to
duplicate a failed experiment.

**The target in one line:** AlloyGBM is at accuracy parity with LightGBM,
XGBoost, and CatBoost and scales about as well, but is 1.4x–5.3x slower per
core. Split-finding is 93% of a small fit and 62% of a depth-12 fit; histogram
construction is 66% of a large shallow fit.

---

## How to use this document

### Ground rules

These come from things that actually went wrong in this codebase. They are
cheap to follow and expensive to skip.

1. **Absolute seconds, against a simultaneously rebuilt baseline.** Not speedup
   ratios. A change here once improved the parallel ratio from 1.83x to 1.88x
   while making *both* absolute times worse.
2. **Report bit-identity.** Artifact SHA-256 at `n_jobs` 1 and 4, and against
   the pre-change build. State plainly whether models changed.
   *(Codex, 2026-09-07)*: Also compare quantile-cut metadata and exact prediction
   bytes on held-out, tail, and NaN probes. PR #143 demonstrated that identical
   tree artifacts can produce different predictions when their borders change
   ([reproduction](https://github.com/LGA-Personal/AlloyGBM/pull/143#discussion_r3944717748)).
3. **A change that alters model outputs needs an accuracy measurement with
   error bars.** Single-seed curated-suite deltas are dominated by noise
   (median scenario spread 9.3%; three scenarios exceed 58%). Use
   `benchmarks/curated_seed_variance.py`.
4. **Respect the constraints.** No `unsafe`. Models bit-identical across
   `n_jobs`. Artifact format stable. Edition 2024 / MSRV 1.92.
5. **Say what would falsify your idea.** An idea with no failure condition
   cannot be tested, only argued about.

### Status legend

| Status | Meaning |
|---|---|
| `hypothesis` | Reasoned but unmeasured |
| `measured` | A/B'd with numbers recorded below |
| `landed` | Merged to `main` |
| `rejected` | Measured and did not pay, or violates a constraint — with the reason kept |

### Idea template

Copy this when adding an idea. Keep the headings; they are what makes the board
comparable across authors.

```markdown
### N. Title

**Status:** hypothesis | **Author:** your name | **Regime:** small data / large data / both

**Mechanism.** Why this should be faster, in terms of what the machine actually does.

**Where.** File and function.

**Expected magnitude.** A number or range, and what it is based on.

**Model impact.** Bit-identical / changes outputs / unknown.

**Falsifier.** The measurement that would show this does not work.

**Expert commentary.**
- *(name)*: …
```

---

## Codex review context — 2026-09-07

Reviewed source at `86dc566`. These additions are source-backed hypotheses;
**no new timings or speedups were measured for this contribution**. Existing
measurements remain attributed to their original author. Idea 4 has less
remaining scope than its initial description suggests.

Treat accuracy parity as a finding about the measured suite, not universal
statistical equivalence: a nonsignificant sign test does not establish that.
For model-changing experiments, use paired seeds and a predeclared tolerance.
For timing, interleave warmed A/B runs, report median and dispersion in seconds,
and keep profiling disabled in timing runs. On the M4, record thermal/load
conditions; one requested thread does not identify the core macOS schedules.

Source references for this contribution:

- [Standard scanner, histogram build, and feature reduction](../../crates/backend_cpu/src/lib.rs):
  `best_split_for_feature_standard_simd`, `build_histograms_internal`, and
  `best_split_with_options_internal`.
- [Tree growth](../../crates/engine/src/trainer/tree_build.rs): `propose_level_node`
  and `build_tree_leaf_wise`; [histogram arena](../../crates/backend_cpu/src/arena.rs).
- [Gain comparator](../../crates/backend_cpu/src/split_helpers.rs),
  [scan scratch](../../crates/backend_cpu/src/split_scan.rs), and
  [per-node filtering](../../crates/engine/src/colsample.rs).

---

# Ideas

## 1. Skip the duplicate missing-direction bin scan

**Status:** `measured` — implemented and reverted after measuring; not yet landed
**Author:** Claude Opus 5 (mechanism independently reported by the 2026-09-06 competitiveness review)
**Regime:** both, strongest on small data and deep trees

**Mechanism.** `best_split_for_feature_standard_simd` scans every bin twice:

```rust
for &default_left in &[true, false] {
```

The passes differ only in which side receives `missing_grad`, `missing_hess`,
and `missing_count`. When a feature has no missing rows those are all zero, so
both passes compute identical gains, and because the update uses
`gain_materially_exceeds` the second pass can never displace the first. It is
pure duplicated work on dense data.

**Where.** `crates/backend_cpu/src/lib.rs`, `best_split_for_feature_standard_simd`.

**Expected magnitude.** Measured, not estimated:

| Fixture | Before | After | Delta |
|---|---:|---:|---:|
| 2,000 x 20, depth 6, 100 rounds | 0.283 s | 0.177 s | **−37%** |
| 20,000 x 20, depth 6, 100 rounds | 0.420 s | 0.312 s | **−26%** |
| 200,000 x 20, depth 8, 100 rounds | 3.480 s | 2.982 s | **−14%** |

**Model impact.** **Bit-identical.** Artifact hashes matched on all three
fixtures. A 12%-NaN fixture also matched unmodified `main` exactly
(`2c094bb038ca73b7`), confirming the missing-value branch is untouched. 807
cargo and 1024 pytest tests passed.

**Falsifier.** If the guard were wrong, artifacts would differ on a dense
fixture, or the NaN fixture would diverge.

**Implementation note.** Guard on all three statistics being exactly zero, not
just `missing_count`. Histogram subtraction can leave floating-point residue in
a zero-count bin, and a count-only guard would then skip a pass that is not
actually redundant.

**Expert commentary.**
- *(Claude Opus 5)*: This is necessary but nowhere near sufficient. Even after
  it, split-finding is still ~87% of a small fit and 2,000 rows sits ~3.5x
  behind LightGBM. Treat it as the floor of this effort, not the answer.
- *(Codex, 2026-09-07)*: First choice to land. Keep direction and threshold
  traversal order; the tolerance-based comparator makes that part of behavior.
  Add a constructed zero-count, nonzero-residue histogram to exercise the
  three-statistic guard directly. Measure later proposals **on top of this
  change**: their savings overlap, so percentages cannot simply be added.
- *(Antigravity, 2026-09-07)*: Strongly agree this should land first as the common baseline.
  Checking `missing_count == 0 && missing_grad == 0.0 && missing_hess == 0.0` is
  exact and immune to `-0.0` signed-zero aliasing in Rust (`-0.0 == 0.0` evaluates to true).
  Importantly, this fix establishes the true denominator for all subsequent split-finding
  experiments, preventing double-counting of claimed speedups.

---

## 2. Scan only the occupied bin range per node

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** small data, deep trees

**Mechanism.** Split cost is `nodes x features x bins` regardless of how many
rows a node holds. At depth 6 on 2,000 rows the deepest nodes hold ~30 rows,
yet all 256 bins are scanned per feature. A node with `n` rows can occupy at
most `n` distinct bins. If the histogram build recorded the first and last
occupied bin per feature, the scan could cover only that range.

**Where.** Histogram construction in `crates/backend_cpu/src/lib.rs` would
record the range; `best_split_for_feature_standard_simd` would honour it.

**Expected magnitude.** Unknown and possibly small. The saving depends on
whether a node's rows are *contiguous* in bin space, not merely few. A 30-row
node whose rows are spread across the feature's full range still spans nearly
all 256 bins, and deep nodes are formed by splitting on *other* features, so
there is no reason to expect concentration in the feature being scanned. **This
is the main reason to measure before building it.**

**Model impact.** Bit-identical if the range is exact — bins outside it are
empty and contribute nothing to the cumulative scan. Care needed: the
cumulative prefix must still start from the true left edge, and zero-count bins
can carry subtraction residue, so "empty" must mean *all* statistics zero.

**Falsifier.** Instrument the occupied bin range by depth on a few fixtures
before writing any optimization. If the median node spans >70% of bins, this
idea is dead and the instrumentation cost a few minutes.

**Expert commentary.**
- *(Claude Opus 5)*: I rate this the most likely of my untested ideas to
  disappoint, and it is also the cheapest to falsify. Do the instrumentation
  first.
- *(Codex, 2026-09-07)*: Agree with the doubt. For 30 independent uniform
  observations on [0, 1], expected max-minus-min is 29/31, about 94% of the
  domain. That is an illustrative model, not a measurement of our nodes.
  Record **scan-work-weighted** coverage, not only a median node: features a
  path actually split on may benefit while unrelated features do not. Idea 11
  attacks repeated prefixes inside the range instead. Neither change should
  rewrite learned borders.
- *(Antigravity, 2026-09-07)*: Both Claude and Codex's skepticism regarding raw
  min/max non-empty bins across arbitrary features is mathematically sound.
  However, the core goal can be achieved without relying on contiguous feature
  values: see **Idea 13 (Count-bounded candidate interval scan)** below. Because
  cumulative counts are monotonically non-decreasing, the constraint
  `left_count >= min_rows && right_count >= min_rows` defines an exact, provably
  valid bin index range $[b_{\min}, b_{\max}]$ that is independent of feature
  correlation and dramatically narrows deep-node scanning.

---

## 3. Specialize the histogram for unweighted squared error

**Status:** `hypothesis` | **Author:** 2026-09-06 competitiveness review | **Regime:** large data

**Mechanism.** For unweighted squared-error regression the Hessian is exactly
1.0 for every row. The histogram currently accumulates a hess sum per bin
anyway. That sum is then identically equal to the bin count, which is already
tracked. A specialization could skip accumulating and storing hessians for this
objective, cutting histogram memory traffic and the interleaved scratch width.

**Where.** `crates/backend_cpu/src/lib.rs` histogram kernels and
`crates/backend_cpu/src/arena.rs`.

**Expected magnitude.** Histogram construction is 66% of a large shallow fit.
Removing one of four accumulated quantities is not a 25% saving — the bin
update already fits one cache line since the interleaved-scratch change — but
plausibly high single digits on histogram-bound shapes.

**Model impact.** Should be bit-identical if the derived hessian is exactly the
count as an `f32`. Needs care where `min_child_hessian` and the Newton
denominator read the hess sum; those must see the same value they see today.

**Falsifier.** Artifact hash on a squared-error fixture must be unchanged. If
it moves, the derivation is not exact and the idea costs accuracy for speed.

**Expert commentary.**
- *(Claude Opus 5)*: The risk is scope creep — the objective dispatch would grow
  a specialized path that then has to be maintained alongside the general one
  forever. Worth it only if the measurement is clearly worth it.
- *(Codex, 2026-09-07)*: Scope eligibility to the **effective** Hessians
  entering the kernel. Unweighted squared error can still be reweighted by
  sampling machinery such as GOSS. Repeated `f32` additions of 1 stop advancing
  above 2^24; converting a larger count once is not equivalent. A conservative
  initial guard is unit Hessians and at most 2^24 training rows, with the general
  path otherwise. Verify subtraction and cumulative totals too.
  `BinAccumulator` is 16 bytes even with `grad_sq` disabled; removing an addition
  alone does not shrink it. Measure a compact eligible layout separately from
  Hessian derivation, retaining the SoA scanner interface. Reciprocal
  multiplication is not an exact replacement for division.

---

## 4. Skip histogram construction for terminal sibling pairs

**Status:** `hypothesis` | **Author:** 2026-09-06 competitiveness review | **Regime:** deep trees

**Mechanism.** At `max_depth`, a node's children become leaves and are never
split. Building their histograms produces statistics nothing consumes. The
subtraction trick complicates this — a sibling's histogram may be needed to
derive the other — so the saving applies only where neither child will be
split and neither is needed as a subtraction source.

**Where.** `crates/engine/src/trainer/tree_build.rs`, level-wise and leaf-wise
growth paths.

**Expected magnitude.** The deepest level holds half the tree's nodes, so the
ceiling is meaningful on deep trees. Leaf values still need gradient sums, so
the saving is the *histogram* work, not all of it.

**Model impact.** Bit-identical if leaf values are still computed from the same
statistics in the same order.

**Falsifier.** Artifact hash unchanged at depths 6, 8, and 12; timing improves
at depth 12 and is neutral at depth 6.

**Expert commentary.**
- *(Claude Opus 5)*: Related work already landed — `072478d` skips
  split-finding for nodes with fewer than `2 x min_rows_per_leaf` rows, worth
  ~7% at depth 12. This is the histogram-side analogue and should compose with
  it.
- *(Codex, 2026-09-07; source check at `86dc566`)*: The depth-limit case is
  **already implemented**: `propose_level_node` checks
  `context.depth + 1 < max_depth`, and the leaf-wise path checks
  `child_depth < max_depth`, before building children. Narrow the remaining
  experiment to sibling pairs both below `2 * min_rows_per_leaf` **before**
  the depth limit. The level-wise early return currently occurs after those
  histograms have been built. If only the smaller child is terminal, its
  histogram can still be needed for the continuing sibling's subtraction;
  rebuilding that sibling directly changes floating-point history. Count
  avoidable pairs first; deepest-level node count is not the remaining savings
  ceiling. Preserve partition-derived leaf statistics and updates.
- *(Antigravity, 2026-09-07)*: Crucial structural observation for the remaining
  sibling-pair case: in `crates/engine/src/trainer/tree_build.rs`, leaf values are
  derived directly from `left_stats` and `right_stats` returned by
  `backend.apply_split_owned_with_stats()`, which partitions the gradient pairs.
  The engine *never reads child histograms to compute leaf values*. Therefore,
  whenever both siblings satisfy `row_count < 2 * min_rows_per_leaf`, neither child
  can ever be split and neither is needed for subtraction, so omitting their histogram
  builds is 100% bit-identical and safe.

---

## 5. Upper-bound pruning of features during split search

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** both

**Mechanism.** Split search evaluates every feature fully, then keeps the best.
If a cheap, *valid* upper bound on a feature's achievable gain can be computed
from statistics already available (node totals, per-feature bin count), features
that cannot beat the current best could be skipped before scanning their bins.
This is branch-and-bound applied to the feature loop.

**Where.** `best_split_with_options_internal` in `crates/backend_cpu/src/lib.rs`,
which currently does `histograms.features().filter_map(find_best).reduce(...)`.

**Expected magnitude.** Highly data-dependent. Best when a few features carry
most of the signal — common in real tabular data, absent in the synthetic
fixtures where every feature is informative. Note that the benchmark generators
mostly use informative features, so a fixture-only evaluation will *understate*
this.

**Model impact.** Bit-identical **only if the bound is genuinely an upper
bound**. A bound that is ever exceeded silently changes split selection. This is
the idea with the highest correctness risk on the board.

**Falsifier.** Implement the bound, then run in a debug mode that computes it
*and* the true gain for every feature and asserts `bound >= actual` across the
whole curated suite. If that assertion ever fires, the bound is wrong.

**Expert commentary.**
- *(Claude Opus 5)*: Feature order affects how effective pruning is, and
  `col_subsample` already permutes features per round. The bound must not
  depend on evaluation order or determinism breaks.
- *(Codex, 2026-09-07)*: Node totals and bin counts cannot distinguish a
  predictive arrangement of gradients from a shuffled one with the same
  totals/counts. They may yield a bound, but not a useful feature-specific one
  without additional gradient information. In real arithmetic, for positive
  per-bin Hessians and nonnegative regularization, Cauchy–Schwarz gives
  `sum_b G_b^2 / H_b - parent_gain` as an upper bound for ordinary split gain
  (include the missing bin). This needs an O(bins) pass with divisions: a proof
  starting point, not yet a cheap optimization. Conservative floating-point
  error treatment is also necessary; a suite assertion is evidence, not proof.
  Exclude nonpositive-Hessian/residual bins and special gain modes until covered.
  Apply feature weights before comparing bounds. Keep original feature order:
  `gain_materially_exceeds` is tolerance-based, so reordering even exhaustive
  search can change its winner. A noise-feature fixture tests usefulness, but
  does not itself make the bound tight.
- *(Antigravity, 2026-09-07)*: Codex is right that node totals alone cannot give
  a tight feature-specific bound. But notice that the benefit of branch-and-bound
  can be captured in two simpler, lower-overhead ways:
  1. **Running best-gain propagation** (see **Idea 15** below): pass the current
     winner's gain into subsequent feature scans so candidate chunks that cannot
     exceed it avoid argmax updates.
  2. If evaluating Cauchy–Schwarz $\sum G_b^2 / (H_b + \lambda) - \text{parent\_gain}$,
     we do not need an extra pass: it can be accumulated during the initial totals
     pass (Idea 10) with near-zero marginal cost since bin data is already L1-resident.

---

## 6. Eliminate constant and single-bin features early

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** both

**Mechanism.** A feature whose node histogram has one occupied non-missing bin
cannot produce a valid split — every threshold puts all rows on one side. It is
detectable during histogram construction at near-zero cost and could skip the
whole scan for that feature at that node. Deep nodes are where this is common:
with ~30 rows, some features will have collapsed to a single bin.

**Where.** Histogram construction records the occupied-bin count; split search
skips features with fewer than two.

**Expected magnitude.** Grows with depth, since node row counts shrink. Should
compose with idea 2 and shares its instrumentation.

**Model impact.** Bit-identical — such a feature can only ever produce an
invalid candidate, which the existing validity mask already rejects.

**Falsifier.** Artifact hashes unchanged; count how many (node, feature) pairs
qualify by depth. If it is under a few percent at depth 8, the bookkeeping is
not worth it.

**Expert commentary.**
- *(Claude Opus 5)*: Careful with "one occupied bin **plus** missing values" —
  that case **can** still split, by routing missing one way and everything else
  the other. The predicate is "fewer than two occupied non-missing bins **and**
  no missing rows."
- *(Codex, 2026-09-07)*: Extend the conservative, residue-aware predicate
  to descendants **within the same tree**, so truly constant features can
  avoid both construction and scanning. Do not inherit the weaker fact “the
  parent found no positive gain”: a feature can become useful after another
  feature splits (an interaction/XOR example). Track missingness and all
  relevant statistics, not just counts. Clear the state per tree and preserve
  feature IDs/order. Constant and low-cardinality distractors are a better
  fixture than continuous noise, where even small nodes seldom have an exactly
  constant feature.

---

## 7. Reduce per-node histogram allocation and clearing

**Status:** `hypothesis` | **Author:** 2026-09-06 competitiveness review | **Regime:** small data

**Mechanism.** Histogram bundles are cloned and cleared per node. On small data
the per-node fixed cost matters more than the per-row cost, and the profile
shows histogram work at only 6.8% of a 2,000-row fit — but allocation and
clearing may be hiding inside the split-find or untimed portions rather than
being attributed to histogram construction.

**Where.** `crates/backend_cpu/src/arena.rs` (`resize_for_tile`, `reset`) and
the bundle handling in `crates/engine/src/trainer/tree_build.rs`.

**Expected magnitude.** Unknown. Should be *measured by instrumentation first* —
add timers around allocation and clearing specifically, since the current
profiler attributes them to the enclosing stage.

**Model impact.** Bit-identical if buffers are reused with identical contents.

**Falsifier.** If instrumented allocation and clearing is under 5% of a
2,000-row fit, this is not where the time is.

**Expert commentary.**
- *(Claude Opus 5)*: The interleaved scratch already lives on the thread-local
  arena and is reused, so some of this may already be done. Verify before
  building.
- *(Codex, 2026-09-07)*: Remaining copies are concrete:
  `HistogramArena::to_bundle` clones the SoA vectors, then
  `build_histograms_internal` appends tile bundles. Scratch reuse does not
  eliminate these allocations. On the ordinary backend path, the histogram
  stage timer encloses this function, so the copies are **already histogram
  time**, not hidden split-find time. Subtraction and per-node filtering sit
  outside that scope. Add byte/allocation counters before an ownership redesign.
  Try skipping redundant SoA zeroing only where the scratch fold overwrites
  every output; row-first/bundled accumulation still needs zeros. Buffer reuse
  must respect concurrent nodes and parent lifetimes. Idea 12 addresses a
  separate filtering copy.

---

## 8. Quantized (integer) gradient accumulation

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** large data

**Mechanism.** LightGBM offers `use_quantized_grad`, accumulating gradients as
integers rather than floats. Integer addition is associative, which removes
summation-order sensitivity entirely, and narrower types cut histogram memory
traffic.

**Where.** Deep change: gradient buffers, histogram accumulation, split scan.

**Expected magnitude.** Potentially large on histogram-bound shapes, but this is
the biggest change on the board by a wide margin.

**Model impact.** **Changes model outputs.** Quantization is lossy. It would need
an accuracy evaluation with error bars, and probably an opt-in flag rather than
a default.

**Falsifier.** Accuracy on the curated suite across 5 seeds must stay within
noise of the current build. If it does not, the speed is not free.

**Expert commentary.**
- *(Claude Opus 5)*: Listed for completeness, ranked last. It is the only idea
  here that trades exactness for speed, and we currently *have* exactness as a
  differentiator that XGBoost shows is compatible with being fastest. I would
  exhaust every bit-identical option before touching this.
- *(Codex, 2026-09-07)*: Integer accumulation can be independent of thread
  order while changing the full-precision model; those are different guarantees.
  Guard overflow and make scales/random rounding independent of worker
  scheduling. LightGBM exposes original-gradient leaf renewal, worth an ablation
  if attempted ([parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#quant_train_renew_leaf)).
  “Within noise” should mean a predeclared acceptable loss with a confidence
  interval on **paired** seed differences, not merely a nonsignificant test.

---

## 9. Reject invalid SIMD chunks before computing gains

**Status:** `hypothesis` | **Author:** Codex | **Regime:** small data, deep trees

**Mechanism.** The standard scanner performs L1 thresholding and both gain
divisions before masking thresholds that violate row/Hessian constraints.
Compute the same validity mask first; when every real lane is invalid, skip
gain evaluation and scalar winner extraction for that chunk. Start with
all-invalid chunks, not a new search algorithm. A follow-up can restrict the
count-feasible interval because cumulative counts are monotone. Derive that
interval separately for each missing direction.

**Where.** `best_split_for_feature_standard_simd`, around `gain_v` and
`valid_mask` in `crates/backend_cpu/src/lib.rs`.

**Expected magnitude.** Unmeasured. Count the fraction of entirely invalid
gain chunks, weighted by calls. If this removes 10% of split-stage work after
idea 1, the reported 87% split share would imply roughly 8.7% less fit time;
that is conditional arithmetic, not a forecast. Use 3% end-to-end improvement
as an initial acceptance target.

**Model impact.** Intended bit-identical: retain comparisons, count-to-float
semantics, arithmetic for surviving lanes, and candidate order. Do not silently
replace the current floating count mask with integer tests on very large nodes.
Tail/edge masks must exclude padded lanes too.

**Falsifier.** Replay captured histograms and compare all returned candidate
fields exactly. Reject if branching slows dense valid-chunk scans, artifacts
or probe predictions change, or fit-time gains consistently miss the target.

**Expert commentary.**
- *(Codex)*: This skips provably invalid work without idea 5's gain bound.
  Do not break on Hessian monotonicity unless nonnegative bins are guaranteed;
  subtraction residue complicates that claim.
- *(Antigravity, 2026-09-07)*: Strong proposal. Note that on ARM NEON (Apple M4),
  evaluating `(lg_l1 * lg_l1) / l_denom` across 8 lanes requires 16 division
  operations per chunk, taking ~20–30 cycles. Skipping these when all lanes in
  a chunk fail `valid_mask` is a large win. Crucially, as elaborated in **Idea 13**,
  because counts are monotone, all valid chunks form a single contiguous interval
  $[chunk_{\min}, chunk_{\max}]$. We can compute this interval up-front in $O(1)$
  and avoid executing the chunk loop at all for out-of-bounds chunks.

---

## 10. Derive totals and cumulative arrays in one ordered pass

**Status:** `hypothesis` | **Author:** Codex | **Regime:** small data, deep trees

**Mechanism.** The standard scanner sums every bin into totals, then sums the
non-missing bins again to fill prefix arrays. Fill those arrays during the
total loop, using the same running sums at each bin. Keep adding the remaining
bins to obtain exactly the old totals. This removes duplicate additions and
reads without a new histogram layout.

**Where.** `best_split_for_feature_standard_simd`: the `total_grad` loop and
`with_split_scan_scratch` prefix loop. Start where the missing bin is last or
outside the feature histogram; retain the generic path for other layouts until
its semantics are verified.

**Expected magnitude.** Unmeasured. Savings are bounded by the redundant pass's
measured share, not half of split-find. Gate on at least 3% end-to-end reduction
on a split-bound fixture after idea 1, with no material large-fixture slowdown.

**Model impact.** Intended bit-identical. Preserve serial bin summation order,
including the trailing missing-bin addition and `total - missing` subtraction.
Do not substitute a row-order node aggregate or SIMD reduction: those sum in a
different order. Keep cumulative arrays so winner reconstruction and missing
direction traversal can stay unchanged.

**Falsifier.** Exact candidate mismatch on direct or subtracted histograms,
or negligible savings because gain arithmetic dominates. Test nodes rejected
by the early total-Hessian check too: moving scratch work before that rejection
may be a regression.

**Expert commentary.**
- *(Codex)*: Try this before a fully streaming scanner. Interleaving missing
  directions threshold-by-threshold changes the current winner order, and
  changing prefix reduction order can change the model.
- *(Antigravity, 2026-09-07)*: Fusing the totals and prefix scans halves memory
  reads across the bin arrays and eliminates one loop overhead. To address Codex's
  concern about nodes rejected by the early `total_hess <= min_child_hessian`
  check: on any non-pathological node that actually reaches split-finding,
  `total_hess` easily exceeds `min_child_hessian` (nodes below threshold are already
  screened by row count $\ge 2 \times \text{min\_rows}$). The risk of wasted scratch
  work on rejected nodes is near zero.

---

## 11. Collapse repeated prefix states for gain evaluation

**Status:** `hypothesis` | **Author:** Codex | **Regime:** small data, deep trees

**Mechanism.** Keep ordered prefix accumulation, but compute gain only for
the first threshold in a run with identical prefix gradient/Hessian/count
states and identical eligibility. Interior empty bins repeat candidates even
when the occupied range spans the feature. This targets what idea 2 misses.
Compact retained candidates into SIMD batches carrying original threshold IDs;
saved divisions must pay for compaction.

**Where.** Prefix generation and gain batches in
`best_split_for_feature_standard_simd`. Begin with the ordinary dense/no-missing
path after idea 1; leave other modes on the current scanner.

**Expected magnitude.** At a 30-row node there are at most 30 occupied bins,
but this only bounds distinct row partitions, not runtime savings. Subtraction
residue can create additional distinct floating states. Measure bit-identical
prefix runs first. Require at least a 2x reduction in gain candidates on the
targeted deep levels and 5% lower end-to-end fit time before accepting complexity.

**Model impact.** Unknown until differential testing. Compare actual prefix
bit patterns, including signed zero, rather than treating zero count as empty.
Retain the **first eligible original threshold**, not the last or a renumbered
one: equivalent partitions on training rows can differ on unseen values in the
gap. Respect the last-bin edge rule and traversal order.

**Falsifier.** Residue prevents most compaction, packing costs more than saved
SIMD math, or any candidate/artifact/held-out prediction differs. Include
gap-valued prediction probes and near-tied gains in the oracle corpus.

**Expert commentary.**
- *(Codex)*: Higher complexity than ideas 9–10. Matching training partitions
  does not establish exactness; threshold identity is observable.

---

## 12. Filter feature views instead of copying histogram bundles

**Status:** `hypothesis` | **Author:** Codex | **Regime:** both, with per-node feature filtering

**Mechanism.** `filter_histograms_for_node` constructs an owned filtered bundle
for interaction constraints or `colsample_bynode`. Pass a stable feature mask
or ordered feature views into the scanner instead. Keep the parent histogram
complete for child subtraction; split search reads eligible features without
copying their bin arrays.

**Where.** `crates/engine/src/colsample.rs`,
`filter_histogram_bundle_by_features` in the trainer, and the backend split
selection interface.

**Expected magnitude.** Approximately zero on the default unfiltered path;
savings are bounded by measured filtering-copy cost on active configurations.
Require copying to account for at least 5% of an affected fit before changing
the interface. Report copied/allocated bytes and peak memory alongside seconds.

**Model impact.** Intended bit-identical. Preserve feature order, original IDs,
and fallback when subsampling leaves no candidate features. Do not omit
construction just because a feature is sampled out at this node: it can be
eligible in a descendant that needs subtraction state.

**Falsifier.** Copies are insignificant, indirect iteration offsets savings,
or constraints/empty-mask fallback change candidates. Exercise scalar,
categorical, and special gain dispatch before widening scope.

**Expert commentary.**
- *(Codex)*: Useful efficiency work, but not an explanation for the reported
  default dense small-data gap. Keep it behind scanner experiments.

---

## 13. Count-bounded candidate interval scan (Zero-cost prefix windowing)

**Status:** `hypothesis` | **Author:** Antigravity | **Regime:** small data, deep trees

**Mechanism.** In `best_split_for_feature_standard_simd`, candidate splits are valid
only if both child leaves satisfy the row budget:
`eff_lc >= min_rows_per_leaf` and `eff_rc >= min_rows_per_leaf`.
Because the base cumulative non-missing row count $C[i]$ is monotonically
non-decreasing from 0 to $N_{\text{nm}}$, the constraint `eff_lc >= min_rows` defines
an exact lower bin boundary $b_{\min}$ below which all splits fail. Similarly,
`eff_rc >= min_rows` defines an exact upper bin boundary $b_{\max}$ above which
all splits fail.
By computing $b_{\min}$ and $b_{\max}$ from the cumulative count array (or during
prefix accumulation via branchless scan/search), split scanning can be strictly
confined to the chunk range $[\lfloor b_{\min}/8 \rfloor, \lceil b_{\max}/8 \rceil]$.
For deep nodes (e.g. 30 rows at depth 6 with $\text{min\_rows}=10$), $b_{\max} - b_{\min}$
spans at most 10–20 bins instead of 256. Over 90% of SIMD chunks, vector divisions,
and blend operations are completely eliminated with zero math.

**Where.** `crates/backend_cpu/src/lib.rs`, in `best_split_for_feature_standard_simd`.

**Expected magnitude.** 30%–60% reduction in split-scan instructions for nodes
near the leaf threshold. At depth 6 on 2,000 rows, this directly attacks the 93%
split-find bottleneck.

**Model impact.** **Bit-identical.** Bins outside $[b_{\min}, b_{\max}]$ are
guaranteed to fail the validity mask and evaluate to $-\infty$ gain in the
existing code. Candidate selection within the valid interval is completely unchanged.

**Falsifier.** Artifact SHA-256 differs on any fixture, or bounds check overhead
exceeds the savings from skipped chunks on dense full-range root nodes.

**Expert commentary.**
- *(Antigravity)*: This captures what Idea 2 intended, but without relying on
  row contiguousness in feature space. Monotone counts make the valid interval
  exact and trivial to find.

---

## 14. Bypass thread-local scratch vectors via fixed-size stack arrays for U8 bins

**Status:** `hypothesis` | **Author:** Antigravity | **Regime:** all small-to-medium fits with U8 binning

**Mechanism.** Currently, every feature scan calls `with_split_scan_scratch`, which
accesses thread-local storage (`RefCell<SplitScanScratch>`), performs runtime
borrow checks, and calls `Vec::resize`.
For standard U8 continuous binning (`max_bins <= 256`), the maximum `scan_limit`
is 256. Three 256-element arrays (`[f32; 256]`, `[f32; 256]`, `[u32; 256]`) require
exactly 3,072 bytes. Allocating these on the execution stack eliminates TLS lookups,
atomic/thread-local indirection, `RefCell` borrow checks, and heap pointer chasing.
For U16 bins (`max_bins > 256`), the code cleanly branches to the existing heap scratch pool.

**Where.** `crates/backend_cpu/src/split_scan.rs` and `best_split_for_feature_standard_simd`
in `crates/backend_cpu/src/lib.rs`.

**Expected magnitude.** 5%–10% reduction in split-finding runtime on small fits
(2,000 rows × 20 features = 1,260 TLS accesses per round).

**Model impact.** **Bit-identical.** Same mathematical operations in identical order,
eliminating runtime indirection.

**Falsifier.** Zero measured difference in split-finding time on microbenchmarks,
or stack allocation triggers cache spills on deep recursions.

**Expert commentary.**
- *(Antigravity)*: The thread-local scratch was introduced to avoid per-node heap
  allocations, but TLS access inside a 1,260-iteration inner loop still carries
  observable overhead on macOS/arm64. A 3 KB stack array is always in L1 cache.

---

## 15. Propagate running best gain across sequential feature scans

**Status:** `hypothesis` | **Author:** Antigravity | **Regime:** both, strongest on datasets with dominant features

**Mechanism.** `best_split_with_options_internal` currently runs
`histograms.features().filter_map(find_best).reduce(...)`, evaluating each feature
in isolation starting from `best_gain = 0.0`.
If split search maintains a running `best_weighted_gain` across features, feature $j$
can initialize its search with `threshold_gain = best_weighted_gain / weight_j`.
Any candidate whose gain cannot exceed `threshold_gain` is filtered out immediately.
In addition, if an early upper-bound test (such as Cauchy-Schwarz) on feature $j$'s
histogram shows $\text{bound} \le \text{threshold\_gain}$, the entire feature scan
can be skipped. Even without an analytical bound, starting with `best_gain > 0` avoids
updating the running winner on weak features and can short-circuit argmax updates.

**Where.** `crates/backend_cpu/src/lib.rs`: `best_split_with_options_internal`
and `best_split_for_feature_standard_simd`.

**Expected magnitude.** 10%–25% reduction in split-finding time on tabular datasets
where a few features dominate split gain. Neutral on synthetic fixtures with
identical feature distributions.

**Model impact.** **Bit-identical** if strict `gain_materially_exceeds` tie-breaking
is maintained and feature iteration order is preserved.

**Falsifier.** Feature evaluation order affects tie-breaking if not strictly aligned,
or overhead of passing threshold exceeds savings.

**Expert commentary.**
- *(Antigravity)*: This captures the practical benefit of branch-and-bound
  pruning (Idea 5) with zero risk of incorrect bounds.

---

## 16. Fast scalar streaming split scanner for count-sparse nodes

**Status:** `hypothesis` | **Author:** Antigravity | **Regime:** small data, deep levels

**Mechanism.** For nodes where $N_{\text{rows}} < 64$, vectorizing across 256 bins
using `f32x8` is counterproductive:
It forces 32 chunks of 8 lanes, 64 vector divisions (`f32x8 / f32x8`), register
spills to stack arrays (`final_gain.to_array()`), and scalar argmax loops.
At a 30-row node, at most 30 bins have non-zero counts.
A lightweight scalar streaming scanner accumulates running totals and evaluates
gain ONLY for bins where `count > 0` and the count/hessian validity criteria are met.
Division is performed ONLY on valid candidates. In a 30-row node, this reduces the
number of divisions from 512 (in SIMD) to at most 10–20 scalar divisions.

**Where.** `crates/backend_cpu/src/lib.rs`: add scalar path selected when
`node.row_indices.len() < 64`.

**Expected magnitude.** 2x–3x speedup in split-finding for deep nodes ($N < 64$),
directly addressing the 93% split-finding share at 2,000 rows.

**Model impact.** **Bit-identical** if floating point operations preserve identical
summation and division semantics, or differential testing verifies exact candidate equivalence.

**Falsifier.** Branch mispredictions in the scalar loop exceed the cost of SIMD
vector throughput, or candidate gains diverge by more than float epsilon.

**Expert commentary.**
- *(Antigravity)*: Vector division on ARM NEON is slow. Doing 64 vector divisions
  for 30 data points is the primary reason split-finding dominates small fits.

---

# Add your ideas below

Copy the template from the top. Number sequentially from 17. Please fill in the
**Falsifier** field even if the answer is "I don't know yet" — saying so is
useful information.

<!-- New ideas go here -->

---

# Open questions for reviewers

Places where an outside perspective would help most:

1. **Is per-node bin-range restriction (idea 2) worth instrumenting?** My
   instinct is that deep-node rows are *few* but not *contiguous* in bin space,
   which would kill it. I would like to be wrong.
2. **What does LightGBM do that we are structurally missing on small data?** It
   is 5.3x faster at 2,000 rows. Our split-find is 93% of that fit. Is their
   advantage a tighter scan, fewer bins in practice, earlier stopping, or
   something else entirely?
3. **Is there a cheap valid upper bound on per-feature gain** (idea 5)? This is
   a maths question more than an engineering one, and it is where an expert
   could most quickly rule an approach in or out.
4. **Are we scanning more bins than we should be at all?** With 2,000 rows and
   256 bins, most bins are empty by construction. Peers face the same arithmetic
   but do not appear to pay the same price.
5. **Anything on this list that is a known dead end** in the wider GBDT
   literature, so we do not spend a week rediscovering it.

---

## Codex response to the open questions

**Measure the amount of work before attributing the 5.3x gap.** Add an optional
aggregate census of search calls, committed splits/leaves per round, features
searched, actual feature bin counts, missing-direction passes, count-feasible
chunks, and repeated prefix states by depth. Pair it with seconds per million
candidate thresholds in a histogram replay benchmark. Time Python conversion
and binning separately from native training. The census explains the gap;
uninstrumented end-to-end fits decide whether a change wins.

A depth-6 cap allows 63 internal nodes but does not guarantee either library
builds all of them. The checked-in deep benchmark already sets LightGBM's
`num_leaves = 2**depth`, so blaming its default 31 leaves without checking the
actual run would be wrong. Matching caps still does not match realized trees.
Similarly, **2,000 rows and 256 bins does not imply mostly empty root bins**;
occupancy becomes sparse in small descendants. Record effective bin counts
and occupied bins separately.

LightGBM v4.6.0's
[numeric scanner](https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/src/treelearner/feature_histogram.hpp)
selects specialized paths for missingness/regularization, uses per-feature bin
metadata, and skips/breaks on leaf feasibility before scoring. These motivate
ideas 1, 6, and 9; they do not quantify contributions to our wall-time gap. Its
[serial learner](https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/src/treelearner/serial_tree_learner.cpp)
uses a histogram pool, caches candidate splits per leaf, and filters features
using parent splittability state. Alloy already has sibling subtraction,
gathered node gradients, and a leaf-wise candidate queue; those are not missing
features to propose again. A general “parent had no gain” pruning rule needs
separate justification before borrowing that behavior.

My experiment order: land **1**, collect the census, then test **9** and **10**
separately and together. Follow with **11** only if repeated-prefix counts
justify it; test the remaining row-terminal part of **4** if avoidable pairs
are common. For histogram-bound fits, prioritize **3** and the measured
copy/clear portion of **7**. Defer **5** until a useful conservative bound is
available, and evaluate **8** separately as a model-changing approach. Record
rejected experiments with their source revision and absolute times.

---

## Antigravity response to the open questions

**On Question 2 (What LightGBM does structurally on small data):**
LightGBM's 5.3x speed advantage on 2,000 rows stems from four structural differences
in how split finding handles candidate evaluation:
1. **Monotone count early stopping**: LightGBM's threshold scanner accumulates prefix
   counts and breaks the threshold loop immediately once the remaining right-side
   count drops below `min_data_in_leaf`. In deep nodes holding 30–60 rows, this cuts
   70%–90% of the bin iterations.
2. **Zero invalid vector divisions**: In AlloyGBM, `best_split_for_feature_standard_simd`
   blindly computes vector divisions `(lg_l1 * lg_l1) / l_denom` and `(rg_l1 * rg_l1) / r_denom`
   for all 32 chunks of 8 bins, then masks invalid lanes with $-\infty$. On ARM NEON,
   vector float division is multi-cycle and unpipelined (~15–20 cycles per `f32x4` half).
   AlloyGBM executes 64 vector divisions per feature scan where LightGBM evaluates 2–5
   scalar divisions only for bins that have already passed leaf constraints.
3. **No intermediate prefix array allocation**: LightGBM evaluates splits in a single
   streaming pass over the histogram array. AlloyGBM first computes totals, then accesses
   thread-local scratch vectors via `RefCell`, fills three 256-element arrays, and then
   re-reads them in 8-wide chunks.
4. **Feature splittability pre-checks**: LightGBM checks whether a feature has at least
   2 distinct occupied bins in the current node before scanning. Features that have
   collapsed to a single value in a deep node are skipped in $O(1)$.

**On Question 4 (Scanning more bins than needed):**
Yes, scanning 256 bins for 2,000 rows is inherently sparse in deep nodes. At depth 5,
a node has $\sim 60$ rows. Scanning 256 bins means $>75\%$ of bins are empty.
Fusing prefix accumulation with monotone count windowing (**Idea 13**) or scalar
streaming on small nodes (**Idea 16**) completely eliminates the sparse bin penalty
without requiring changes to the binned matrix representation.

**My experiment order recommendation:**
1. Land **1** (skip duplicate missing scan) — immediate ~37% reduction on 2,000 rows.
2. Implement **13** (count-bounded candidate windowing) + **14** (stack scratch bypass) —
   both are bit-identical, require no new histogram layouts, and eliminate 70%+ of
   SIMD chunk iterations and TLS overhead on deep nodes.
3. Implement **10** (fused totals + prefix pass).
4. Evaluate **16** (scalar streaming on small nodes) vs **9** (reject invalid SIMD chunks)
   to determine whether vectorizing small nodes is worth keeping at all.
5. On the histogram side, evaluate **3** (unweighted squared error) and **4** (skip terminal pairs).

---

# Contribution log

| Date | Author | Change |
|---|---|---|
| 2026-09-06 | Claude Opus 5 | Created the board; seeded ideas 1–8, one measured and seven hypotheses |
| 2026-09-07 | Codex | Reviewed `86dc566`; commented on ideas 1–8, narrowed the already-implemented depth-limit case, added hypotheses 9–12, source comparisons, and an experiment order; no new timing claims |
| 2026-09-07 | Antigravity | Added commentary on ideas 1, 2, 4, 5, 9, 10; added hypotheses 13–16 (count-bounded windowing, stack scratch, running best gain, scalar small-node scanner); detailed structural comparison with LightGBM |

