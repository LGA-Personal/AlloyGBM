# Accuracy at Depth: Ideas Board

| Date | Author | Version | Branch | Status |
|---|---|---|---|---|
| 2026-09-19 | Claude Opus 5 | v1.0.0-prep | `perf/auto-policy-depth-accuracy` | **Open for contribution** |

**In one sentence:** AlloyGBM matches LightGBM, XGBoost and CatBoost on accuracy at
moderate depth, but it degrades far more than LightGBM or XGBoost as depth grows —
and the degradation is almost entirely confined to classification and multiclass
objectives, which points at a specific structural cause rather than a tuning
constant.

This document exists to collect ideas for fixing that, from several
contributors, before any of them are built. **Every idea here needs a reviewer
field filled in and a falsifier stated.** The companion document for the
just-completed throughput cycle is
[the single-thread optimization board](single-thread-optimization-board.md); it
closed 18 ideas, and the method that worked there is the method proposed here.

---

## 1. What we measured

Full curated suite (16 scenarios, 4 libraries), five depths, one and ten threads,
plus 5-seed variance runs at depths 6 and 12. Matched sampling (0.8/0.8),
matched bin counts, learning rate, rounds and seed; `num_leaves = 2**max_depth`
for LightGBM so its default of 31 does not cap it below the depth-wise peers; one
thread budget applied to every library's own knob *and* to the OpenMP/BLAS
environment.

### Accuracy is at parity at depth 6

5-seed medians, ties judged against each scenario's own measured seed spread:

| | Win | Tie | Loss |
|---|---:|---:|---:|
| vs LightGBM | 1 | 14 | 0 |
| vs XGBoost | 1 | 13 | 1 |
| vs CatBoost | 1 | 12 | 2 |

### It is not at parity at depth 12

| | Win | Tie | Loss |
|---|---:|---:|---:|
| vs LightGBM | 1 | 12 | 2 |
| vs XGBoost | 1 | 9 | **5** |
| vs CatBoost | 3 | 9 | 3 |

### The cause is depth sensitivity, not a level shift

Change in metric from depth 6 to depth 12 (positive = worse with more depth),
5-seed medians:

| Library | Median across 15 scenarios | Got worse on | Median on the 6 log-loss scenarios |
|---|---:|---:|---:|
| **AlloyGBM** | **+4.3%** | **12/15** | **+21.8%** |
| LightGBM | +0.0% | 8/15 | +2.4% |
| XGBoost | +0.3% | 9/15 | +2.5% |
| CatBoost | +10.3% | 10/15 | +26.2% |

LightGBM and XGBoost are essentially flat from depth 6 to 12. We lose ~22% on
log-loss. CatBoost is worse than us, which is worth noting but is not the bar.

### The split by objective is the most informative single fact

| Scenario | Metric | AlloyGBM d6→d12 |
|---|---|---:|
| abalone_regression | rmse | +4.3% |
| bike_sharing | rmse | −1.5% |
| california_housing | rmse | −3.0% |
| dense_numeric | rmse | +0.6% |
| dow_jones_financial | rmse | −0.2% |
| panel_time_series | rmse | +0.4% |
| **histogram_stress** | rmse | **+43.2%** |
| adult_income | log_loss | +5.6% |
| breast_cancer | log_loss | +9.9% |
| digits_multiclass | log_loss | +17.6% |
| wine_multiclass | log_loss | +26.0% |
| synthetic_multiclass | log_loss | +37.9% |
| synthetic_classification | log_loss | +60.7% |

Regression is fine with one exception. Classification and multiclass are
uniformly bad. Any explanation that does not account for that asymmetry is
probably wrong.

### Two separate findings not to conflate with the above

- **`panel_time_series`** carries a stable ~5% deficit at *both* depths. It
  flipped from tie to loss at depth 12 only because its seed spread narrowed
  from 5.5% to 1.7%, not because the gap grew. That looks like a standing
  weakness independent of depth.
- **`histogram_stress`** is simultaneously our largest accuracy *win* (40–56%
  better than all three peers at both depths, far outside noise) and the worst
  regression-side depth degrader (+43.2%). It may be a different phenomenon from
  the classification story.

---

## 2. What we found in the code

### The auto training policy has no depth awareness whatsoever

Every knob keys off row count, feature count, binned density, or target
variance. None reads `max_depth` or leaf count
(`crates/engine/src/trainer/mod.rs`, `resolve_auto_iteration_controls`;
`crates/engine/src/trainer/policy.rs`):

| Knob | Keyed on | Depth-aware? |
|---|---|---|
| `min_rows_per_leaf` | row-count buckets → 1 / 2 / 4 / 8 / 16 | no |
| `min_split_gain` | binned density, rows × features → 0.001 / 0.0001 / 0 | no |
| `row_subsample` | rows ≥ 2,048 → 0.9, ≥ 16,384 → 0.8 | no |
| `col_subsample` | features ≥ 32 → 0.8 / 0.65 / 0.5 | no |
| auto split L2 | rows < 1,024 **and** ≥ 8 features **and** variance > 4 | no |

A depth-6 tree holds at most 64 leaves; a depth-12 tree at most 4,096. At 8,192+
rows the policy asks for `min_rows_per_leaf = 16` in both cases — roughly 64x
more leaf capacity under an identical leaf-size floor.

Note also that the auto split-L2 trigger requires **fewer than 1,024 rows**.
Every scenario in which we measured depth-12 degradation is larger than that, so
those fits received *no* automatic regularization at all.

### The policy differentiates ranking, but not classification

`resolve_auto_iteration_controls` takes an `is_ranking: bool` and zeroes
`min_split_gain` for ranking objectives. There is no equivalent branch for
log-loss objectives, even though that is where the problem lives.

### Our leaf constraint cannot self-tighten; XGBoost's can

| | AlloyGBM | XGBoost |
|---|---|---|
| Leaf-capacity knob | `min_rows_per_leaf` (a **row count**) | `min_child_weight` (a **Hessian sum**) |
| Default | auto: 1–16 by row count | **1** |
| `min_child_hessian` default | **0.0** | n/a (same knob) |

For squared error the Hessian is identically 1.0, so a row count and a Hessian
sum are the same constraint. For log-loss the Hessian is `p(1−p)`, which shrinks
toward zero as the model grows confident — so **XGBoost's constraint tightens by
itself as boosting proceeds, and ours does not**. That mechanism predicts
exactly the regression-versus-classification asymmetry in the table above, which
is why it is idea 1.

We default `min_child_hessian` to 0.0, which disables even the manual version of
that constraint.

### We grow level-wise by default; LightGBM grows leaf-wise

`TreeGrowth::Level` is the default. The benchmark matched nominal capacity via
`num_leaves = 2**max_depth`, but not how capacity is *consumed*: a leaf-wise
frontier stops adding leaves when gain is exhausted, whereas level-wise growth
builds every node at every level subject only to the floors above.

---

## 3. What we are trying to achieve

**Goal:** be as accurate as the best peer across the whole depth range and across
objectives, not just at the depths the policy was tuned for. Concretely: get the
depth 6 → 12 log-loss degradation from +21.8% down toward LightGBM's and
XGBoost's +2.5%, without giving up the depth-6 parity we already have.

**Non-negotiable constraints.** A proposal that violates one is not usable:

1. **Must not regress depth ≤ 8.** We are at parity there (1W/14T/0L vs
   LightGBM at depth 6). A fix that trades shallow accuracy for deep accuracy is
   not a fix.
2. **Trained models must stay bit-identical across `n_jobs`.** This is a
   published guarantee and is unaffected by anything proposed here, but it must
   be re-verified, because several ideas touch split selection.
3. `unsafe_code = "forbid"` workspace-wide. Rust edition 2024, MSRV 1.92.0.
4. **Artifact format is stable from v1.0.0.** A 1.x artifact must stay readable
   by any later 1.x release. Changing *default behaviour* is permitted; changing
   the format is not.
5. **Throughput must not regress materially.** The branch this sits on just took
   a 2,000-row fit from 0.292 s to 0.074 s, bit-identically. Reviewers should
   assume that is defended.

**Everything here changes model outputs.** That is the fundamental difference
from the throughput board, where 6 of 18 ideas landed bit-identically. No
bit-identity check can validate an idea on this board; every one needs accuracy
evidence with error bars.

---

## 4. Measurement discipline

This section is longer than it would otherwise be because the previous cycle
produced two measurement errors that changed conclusions, and both are easy to
repeat here.

- **5-seed medians, always. Single-seed deltas on this suite are not evidence.**
  Measured seed spreads: median 9.3%, and `breast_cancer` 130%,
  `digits_multiclass` 59–66%, `wine_multiclass` **854%**. Early in this
  investigation a single-seed comparison with a 1% tie band produced "88 losses
  vs 32 wins" against LightGBM; the correct 5-seed answer for the same build was
  1 win, 14 ties, 0 losses. Use `benchmarks/curated_seed_variance.py`.
- **Judge each scenario against its own spread**, not a global band. A 100% swing
  on `wine_multiclass` is noise; a 5% swing on `panel_time_series` (spread 1.7%)
  is real.
- **Absolute values matter as well as ratios.** Relative gaps explode when a
  metric approaches zero. A log-loss moving 0.08 → 0.20 is +154% and may matter
  less than 2% on a large RMSE.
- **Re-run peers under the fairness gate**; do not compare a new build against
  numbers in this document. Earlier in this project a peer comparison was
  invalidated because AlloyGBM ran on auto sampling while the peers ran at
  1.0/1.0.
- **Verify your instrumentation reads what you think it reads.** A counter added
  during the last cycle reported "0.5%–2.1% of bins are exactly empty"; it was
  reading a shadowed binding and the true figure was 90.7%. Two ideas were
  wrongly closed on it. Print a handful of raw values and sanity-check them
  before trusting an aggregate.
- **Price a change against what has already landed**, not against the original
  profile. Two throughput ideas were worth a tenfold win when proposed and 0%
  once an earlier change removed the cost they targeted.
- **Measure the ceiling before building the correctness machinery.** An
  unguarded, knowingly-incorrect probe is often enough to decide whether an idea
  deserves a real implementation.

### Status legend

| Status | Meaning |
|---|---|
| `hypothesis` | Proposed, not yet measured |
| `probing` | Ceiling or mechanism being measured |
| `building` | Implementation in progress |
| `landed` | Merged, with accuracy evidence |
| `rejected` | Measured and does not pay — reason recorded |
| `reopened` | Previously closed on evidence later found faulty |

### Idea template — copy this

```markdown
## N. Title

**Status:** `hypothesis` | **Author:** your name | **Regime:** which objectives / depths / data shapes

**Mechanism.** What is wrong today and what the change does about it.

**Where.** Files and functions.

**Expected effect.** On which scenarios, and roughly how much. Say "unknown" if
that is the honest answer.

**Model impact.** What changes in trained models, and for which users.

**Falsifier.** The measurement that would kill this idea. Required.

**Reviewer commentary.**
- *(your name, date)*: 
```

---

# Ideas

Ideas 1–9 are seeded by Claude Opus 5 and are deliberately a mix: some are
near-certain calibration work, some are design changes with real trade-offs, and
at least one is probably wrong but worth ruling out explicitly. Rank them, argue
with them, and add your own from 10 onward.

---

## 1. Default `min_child_hessian` above zero, so the leaf constraint self-tightens

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** classification and multiclass, all depths

**Mechanism.** Our leaf-capacity floor is a row count, so it means the same thing
at boosting round 1 and round 500. XGBoost's `min_child_weight` is a Hessian sum,
and for log-loss the per-row Hessian `p(1−p)` shrinks as predictions sharpen — so
the same numeric threshold demands progressively more rows per leaf as training
proceeds. That is an automatic, self-scaling brake on late-round overfitting
which we simply do not have, because `min_child_hessian` defaults to 0.0.

For squared error the Hessian is identically 1.0 and the two formulations
coincide, which is exactly why regression is unaffected in our measurements.

**Where.** `crates/core/src/config.rs` (default), `crates/engine/src/trainer/policy.rs`
(`split_selection_options_with_resolution_for_training`), and the auto-policy
resolution in `crates/engine/src/trainer/mod.rs`.

**Expected effect.** This is the leading candidate to explain the whole
classification gap. If it is right, a non-zero default should recover most of the
+21.8% log-loss degradation with little effect on regression. If it is wrong, the
asymmetry in the data needs another explanation.

**Model impact.** Changes default classification models for every user. Regression
and ranking should be nearly unaffected (Hessian ≡ 1 for squared error), which
makes the blast radius unusually easy to bound — and easy to verify.

**Falsifier.** Sweep `min_child_hessian` manually at depth 12 on the six log-loss
scenarios. If no value recovers meaningful accuracy, the self-tightening story is
wrong and this idea dies along with the reasoning behind ideas 2 and 4.

**Reviewer commentary.**
- *(Claude Opus 5)*: The most testable idea on the board and the one I would run
  first, because it is the only one whose mechanism explains the
  regression-versus-classification split without extra assumptions. Note the
  interaction with idea 2: if this works, depth-scaling a row count may be
  unnecessary, and shipping both would double-regularize.
- 

---

## 2. Scale `min_rows_per_leaf` with the tree's leaf capacity

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** all objectives, deep trees

**Mechanism.** The straightforward calibration. The auto policy currently picks
`min_rows_per_leaf` from row count alone, so a depth-12 tree gets ~64x the leaf
capacity of a depth-6 tree under the same floor. Scale the floor so that expected
rows-per-leaf stays above some target: for instance require
`min_rows_per_leaf >= rows / (k * 2**depth)` for some k, or bucket on
`rows / 2**depth` instead of on rows.

**Where.** `resolve_auto_iteration_controls` in `crates/engine/src/trainer/mod.rs`.

**Expected effect.** Should blunt depth degradation across all objectives,
including the `histogram_stress` regression case. Less targeted than idea 1 and
more likely to cost accuracy at shallow depth if the scaling is wrong.

**Model impact.** Changes default models at depth > ~8 for every objective.

**Falsifier.** Sweep `min_data_in_leaf` manually at depth 12. If the value that
fixes depth 12 also measurably hurts depth 6, a single depth-independent constant
cannot serve both and the scaling law is the whole problem — which is worth
knowing before writing one.

**Reviewer commentary.**
- *(Claude Opus 5)*: I rank this second, not first, because a row count cannot
  distinguish "a leaf with 20 confident rows" from "a leaf with 20 uncertain
  rows", and the measurement says the problem is specific to the objectives where
  that distinction exists.
- 

---

## 3. Give the auto policy a leaf budget rather than trusting `max_depth`

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** all objectives, deep trees

**Mechanism.** LightGBM's primary capacity knob is `num_leaves`, not depth, and it
is flat at depth 6→12 in our data. Rather than regularizing harder at depth, the
auto policy could cap *effective* capacity: derive a leaf budget from the data
(say a function of rows and features) and stop growing when it is hit, treating
`max_depth` as a ceiling rather than a target. We already have `max_leaves` for
leaf-wise growth; this would extend the idea to the level-wise default.

**Where.** `resolve_auto_iteration_controls`, plus the level-wise growth loop in
`crates/engine/src/trainer/tree_build.rs`.

**Expected effect.** Potentially the cleanest match to LightGBM's behaviour. Also
the largest behavioural change: users who set `max_depth=12` today get trees that
may stop well short of it.

**Model impact.** Substantial. Arguably changes the meaning of `max_depth` under
auto policy, which is a documentation and expectations problem as much as a code
one.

**Falsifier.** Emulate it without code: at depth 12, set `tree_growth="leaf"` with
`max_leaves` swept across a range, and compare against depth-12 level-wise. If a
leaf budget recovers accuracy, the idea has legs; if leaf-wise at matched leaf
count is no better, capacity shape is not the issue.

**Reviewer commentary.**
- *(Claude Opus 5)*: Worth testing precisely because the falsifier is cheap — we
  already support leaf-wise growth and `max_leaves`, so the experiment needs no
  new code at all.
- 

---

## 4. Make the auto policy objective-aware

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** classification and multiclass

**Mechanism.** The policy already special-cases ranking (it zeroes
`min_split_gain` because ranking gradients have a different gain scale). There is
no equivalent branch for log-loss, despite that being where every measured
problem is. Whatever the eventual fix, it may well need to apply to
classification and not regression — and the policy currently has no way to
express that.

**Where.** `resolve_auto_iteration_controls` (already receives `is_ranking`;
would need objective identity rather than a single bool).

**Expected effect.** Enabling, not curative. This is the plumbing that ideas 1, 2
and 5 may need in order to be applied selectively rather than globally.

**Model impact.** None on its own.

**Falsifier.** If the sweep shows the same regularization setting is optimal for
both regression and classification, this is unnecessary complexity and should be
dropped.

**Reviewer commentary.**
- *(Claude Opus 5)*: Deliberately listed as a separate idea so it can be argued
  on its own merits. My concern is that per-objective policy branches multiply
  and become untestable; if one knob serves everything, that is much better.
- 

---

## 5. Extend the auto split-L2 trigger beyond tiny datasets

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** all objectives, deep trees

**Mechanism.** `should_apply_auto_split_l2` fires only when rows < 1,024, features
≥ 8, and target variance > 4. Every scenario where we measured depth-12
degradation has more than 1,024 rows, so none of them received any automatic L2
at all. L2 in the gain denominator is the standard remedy for over-eager deep
splits; the trigger may simply be scoped too narrowly.

**Where.** `should_apply_auto_split_l2` and `AUTO_SPLIT_L2_NOISY_SMALL_WIDE` in
`crates/engine/src/trainer/policy.rs`.

**Expected effect.** Unknown magnitude, but the cheapest possible experiment:
`lambda_l2` is already a user-settable parameter, so the ceiling can be measured
with no code change at all.

**Model impact.** Changes default models wherever the widened trigger fires.

**Falsifier.** Sweep `lambda_l2` at depth 12. If no value recovers accuracy
without hurting depth 6, L2 is not the right lever here.

**Reviewer commentary.**
- *(Claude Opus 5)*: Probably partially effective and probably not the best
  answer, since L2 shrinks leaf values everywhere rather than specifically
  discouraging tiny leaves. Include it in the sweep because it is free to test.
- 

---

## 6. Depth-scaled leaf shrinkage in the default path

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** all objectives, deep trees

**Mechanism.** MorphBoost already applies a per-leaf depth penalty
(`depth_penalty_base ^ (depth/3)`) and per-iteration shrinkage, but only in
`training_mode="morph"`. Deep leaves are the least reliable estimates in the
tree; scaling their values down by depth is a cheap, continuous alternative to
refusing to create them.

**Where.** Leaf-value computation in `crates/engine/src/trainer/tree_build.rs`;
the existing morph implementation is the reference.

**Expected effect.** Unknown. Attractive because it degrades gracefully rather
than imposing a hard cutoff, and because a working implementation already exists
to borrow from.

**Model impact.** Changes all deep models.

**Falsifier.** Run `training_mode="morph"` at depth 12 and compare its depth
degradation against the standard path. If morph already degrades much less, the
mechanism is worth lifting into the default; if it degrades the same, this is not
the lever. **This costs nothing to check and should be one of the first things
anyone runs.**

**Reviewer commentary.**
- *(Claude Opus 5)*: The cheap falsifier makes this unusually good value — we may
  already have the answer sitting in an existing mode.
- 

---

## 7. Post-growth pruning

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** all objectives, deep trees

**Mechanism.** Every idea above prevents bad splits before they are made, using
information available at split time. Classical CART instead grows greedily and
then prunes back by cost-complexity, which can use information the growth phase
did not have — including a validation signal.

**Where.** New pass after `build_tree_level_wise` / `build_tree_leaf_wise`.

**Expected effect.** Unknown, and this is the largest implementation on the board.

**Model impact.** Substantial.

**Falsifier.** If the cheaper preventive ideas (1, 2, 5) close the gap, this is
not worth the complexity. It should be evaluated last, and only if they fail.

**Reviewer commentary.**
- *(Claude Opus 5)*: Listed for completeness and ranked last on cost. I would
  rather exhaust the policy-level options first. Also worth noting none of the
  three peers does classical post-pruning by default, which is weak evidence
  against its necessity.
- 

---

## 8. Investigate `histogram_stress` and `panel_time_series` as separate problems

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** two specific scenarios

**Mechanism.** Not a fix — a warning against averaging. `histogram_stress` is
both our biggest win (40–56% better than every peer, at both depths) and the
worst regression-side depth degrader (+43.2%). `panel_time_series` carries a
stable ~5% deficit at both depths that has nothing to do with depth at all.
Neither fits the classification story, and a global regularization change could
easily damage the `histogram_stress` win while chasing the log-loss gap.

**Where.** Diagnostic, not a code change. Scenario definitions are in
`benchmarks/run_model_comparison.py`.

**Expected effect.** None directly; protects against a false positive elsewhere.

**Model impact.** None.

**Falsifier.** If the eventual fix improves log-loss scenarios *and* leaves
`histogram_stress` intact, this concern was unnecessary — which is a fine outcome
to discover.

**Reviewer commentary.**
- *(Claude Opus 5)*: Include `histogram_stress` as an explicit guard metric in
  every sweep, not just an averaged row. It is the scenario we are most likely to
  break by accident.
- 

---

## 9. Null hypothesis: the peers are simply better-tuned at depth, and this is not fixable by policy

**Status:** `hypothesis` | **Author:** Claude Opus 5 | **Regime:** framing

**Mechanism.** Stated so it can be tested rather than assumed away. It is possible
that no auto-policy setting recovers the gap, and the difference comes from
something more structural — leaf-wise versus level-wise growth, or the histogram
binning interacting with deep splits. If the manual sweeps in ideas 1, 2 and 5
all fail to recover depth-12 accuracy, that is the finding, and the honest
response is to document depth > 8 as a known weakness rather than ship a fix that
does not work.

**Where.** n/a.

**Expected effect.** n/a.

**Falsifier.** Any sweep result that *does* recover depth-12 accuracy kills this
and we proceed with that lever.

**Reviewer commentary.**
- *(Claude Opus 5)*: Worth keeping on the board explicitly. In the last cycle
  twelve of eighteen ideas were rejected on measurement, several of which had
  looked obviously correct beforehand. A stated null hypothesis makes it easier
  to accept that outcome if the data points there.
- 

---

# Add your ideas below

Copy the template from section 4. Number sequentially from 10. Please fill in
the **Falsifier** field even if the answer is "I don't know yet" — saying so is
useful information. If you disagree with the framing in sections 1–3, say so
there rather than working around it.

<!-- New ideas go here -->

---

# Open questions for reviewers

Where an outside perspective would help most:

1. **Is a Hessian-based leaf constraint the right default for log-loss, or is a
   depth-scaled row count enough?** This is the central design fork. Ideas 1 and
   2 are alternative answers and shipping both would over-regularize. The
   Hessian version has XGBoost's precedent and explains our measured asymmetry;
   the row-count version is simpler and objective-agnostic. Which would you ship,
   and what would change your mind?
2. **Should `max_depth` under auto policy be a target or a ceiling?** Idea 3
   would let the policy stop well short of the requested depth. Is that a
   reasonable thing for an auto mode to do, or a violation of user intent?
3. **How much per-objective branching is acceptable in an auto policy** before it
   becomes untestable? Idea 4 opens that door deliberately.
4. **Is there a regularizer that is self-tightening like `min_child_weight` but
   objective-agnostic?** That would resolve question 1 by making it moot.
5. **Anything on this list that is a known dead end** in the wider GBDT
   literature, so we do not spend a week rediscovering it.
6. **Is the +43.2% depth degradation on `histogram_stress` the same phenomenon
   as the classification gap, or something else?** It is our best result and the
   one we are most likely to break.

---

# Contribution log

| Date | Author | Change |
|---|---|---|
| 2026-09-19 | Claude Opus 5 | Created the board. Measured the depth-accuracy gap across 5 depths / 16 scenarios / 4 libraries plus 5-seed variance at depths 6 and 12; audited the auto policy; seeded ideas 1–9 and six open questions. |
