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

---

# Add your ideas below

Copy the template from the top. Number sequentially from 9. Please fill in the
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

# Contribution log

| Date | Author | Change |
|---|---|---|
| 2026-09-06 | Claude Opus 5 | Created the board; seeded ideas 1–8, one measured and seven hypotheses |
