# Auto-regularization calibration protocol

**Status:** pre-registered 2026-09-21, before any policy constant was fitted.
**Owner:** Claude Opus 5. **Related:** [accuracy-at-depth-board.md](accuracy-at-depth-board.md).

## The risk this exists to control

We are about to choose the *form* and the *constants* of an automatic
regularization policy — how `lambda_l2` should scale with tree depth, class
count and sample count — by measuring candidate settings on a benchmark suite.
That is fitting a model. The scenarios are the training data, the policy is the
model, and the reported improvement is a training-set score unless something
stops it being one.

This is not hypothetical. Two results in this program have already turned out to
be properties of fixtures rather than of the library:

- The headline `-35%` depth-12 log-loss gain was largely an artefact of
  `synthetic_classification` being 97/3 imbalanced. Real low-K data gives
  `-3` to `-4%`.
- A claimed "isolated within-K contrast" rested on `synthetic_multiclass` being
  K=3. It is K=5. The contrast never existed.

A policy tuned on the same suite that produced those numbers would inherit both
problems and report a large in-sample win.

## Calibration set

Used freely to choose the policy form and fit its constants.

- **Class-count sweep** (identical except K): `kgrid_n8k_p16_k2`,
  `kgrid_n8k_p16_k5`, `kgrid_n8k_p16_k10`, `kgrid_n8k_p16_k20`.
- **Shape grid**: `grid_n8k_p16_sig16`, `grid_n8k_p16_sig4`,
  `grid_n8k_p64_sig16`, `grid_n32k_p16_sig4`.
- **Already-measured scenarios** (all of these have been seen, so they cannot
  serve as holdout): `adult_income`, `magic_gamma`, `letter_recognition`,
  `covertype_multiclass`, `synthetic_classification`, `synthetic_multiclass`,
  `digits_multiclass`, `wine_multiclass`, `breast_cancer`, and the regression and
  ranking scenarios used as guards.

## Holdout set — frozen

Created 2026-09-21 and **not measured under any candidate policy** until the
policy is frozen. Chosen to sit at parameter values the calibration set does not
contain, because a scaling law is most likely to fail between and beyond its
fitted points.

| scenario | why it is a real test |
| --- | --- |
| `kgrid_n8k_p16_k3` | K between the fitted K=2 and K=5 points |
| `kgrid_n8k_p16_k14` | K between K=10 and K=20 |
| `pendigits_multiclass` | real, K=10, n=7,494 — small-n high-K, a combination no calibration scenario has |
| `sensorless_multiclass` | real, K=11, n=58,509 — large-n high-K, 48 continuous features |

## Rules

1. The holdout is measured **once**, after the policy form and every constant
   are committed to git.
2. If the holdout result is bad, that is the finding and it gets reported as
   such. It does not license re-tuning and re-measuring; doing that converts the
   holdout into calibration data and there is no second holdout.
3. Any later change to the policy requires a **new** holdout set, built the same
   way, at parameter values the then-current calibration set does not contain.
4. Guard scenarios (`histogram_stress`, `panel_time_series`) remain guards in
   both phases: they check for collateral damage, not for gain.
5. A calibration win that does not survive holdout is reported as a *negative*
   result for the policy, not as a property of the holdout data.

## What "frozen" means

The policy is frozen when the function mapping (depth, class count, row count,
objective) to a regularization strength is committed with its constants as
literals, its tests written, and its intended default behaviour stated. Reading
holdout numbers before that point voids the holdout.

## Amendment 2026-09-21: the class-count cells were regenerated

The first `kgrid` family held mean separation fixed across K, which made the
cells progressively harder rather than merely differently-classed: the K=20 cell
reached **0.913 of the random-guess log-loss**, i.e. nearly unlearnable. Maximal
shrinkage trivially wins on a near-noise task, so that sweep measured difficulty
and not class count, and its result (λ=20 optimal at every K) is void.

All `kgrid` cells — calibration and holdout — were regenerated with the mean
separation solved per K to hold Bayes accuracy at ~0.80. The holdout remains
valid because no policy result had been measured on it; only the data changed,
and it changed before any holdout measurement, not after.

This is recorded rather than quietly fixed because the failed sweep is exactly
the kind of in-sample artefact this protocol exists to catch, and it was caught
by comparing against `ln(K)` rather than by any guard written in advance.

## Calibration result, 2026-09-21

Seven λ arms (0, 0.1, 0.5, 1, 2, 5, 20) × depths {6, 12} × 5 paired seeds.
Data: `benchmarks/results/accuracy_depth/calib_lambda/` and
`.../calib_kfixed/` (the difficulty-matched class-count re-run).

### Synthetic and real disagree, in direction as well as magnitude

| group | optimal λ | benefit at depth 12 |
| --- | --- | --- |
| all 8 synthetic scenarios | **20 in every one** | −7% to −45% |
| 4 real scenarios | 0.1, 0.5, 5, 20 | −2.4% to −4.3% |

The difficulty-matched class-count sweep says λ benefit *rises* with K
(K=2 −14.45%, K=20 −45.51%, λ=20 best throughout). Real data says the opposite:
optimal λ *falls* as K rises — 20 and 5 at K=2, 0.5 at K=7, 0.1 at K=26. The
controlled instrument is internally valid (K is genuinely isolated, every cell
equally learnable) and externally invalid: it does not predict the real-data
behaviour it was built to explain.

**Consequence: the synthetic scenarios cannot calibrate this policy.** A fit
weighted by scenario count would be dominated 8-to-4 by cells that all say
λ=20, which on real data costs `letter_recognition` +76.09% and
`covertype_multiclass` +16.93%.

### Depth 6: no λ helps real data

| scenario | best real result at depth 6 |
| --- | --- |
| `adult_income` | −0.23% (3/5) |
| `magic_gamma` | +0.01% |
| `covertype_multiclass` | +0.40% |
| `letter_recognition` | +0.02% |

Nothing is distinguishable from zero, and everything degrades as λ rises. The
depth-conditionality is the one part of the original finding that survives
cleanly.

### Depth 12: a small λ helps, but no single value helps everything

| scenario | K | λ=0.1 | λ=0.5 | λ=1 | λ=5 | λ=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adult_income` | 2 | +0.44% | −0.86% | −1.93% | −3.37% | **−4.25%** |
| `magic_gamma` | 2 | −0.21% | −1.93% | −1.95% | **−4.08%** | −3.38% |
| `covertype_multiclass` | 7 | −1.81% | **−2.38%** | −2.17% | +3.65% | +16.93% |
| `letter_recognition` | 26 | **−4.28%** | +0.70% | +4.46% | +22.91% | +76.09% |

No λ helps all four. λ=0.5 is the best compromise: it helps three
(−0.86%, −1.93%, −2.38%) and costs the fourth only +0.70%.

### Why no scaling law is being fitted

Optimal λ on real data is monotone in K across the four points, and λ ≈ 40/K²
fits them tolerably. It is not being shipped, for three reasons:

1. **Four points, and the candidate forms disagree.** Per-class curvature
   ≈ (1/K)(1−1/K) motivates λ ∝ 1/K, which fits *worse* than the λ ∝ 1/K² the
   data prefers. Choosing the steeper form because it fits four points better is
   fitting noise.
2. **K does not determine λ\* even within the real set.** `adult_income` and
   `magic_gamma` are both K=2 with n within 25% of each other, and their optima
   differ 4× (20 versus 5).
3. **The only controlled K evidence contradicts the sign of the fit.** Adopting a
   law that the one experiment built to isolate K rejects would mean trusting
   four confounded real points over a clean synthetic contrast, without being
   able to say why the synthetic one fails to transfer.

### Recommendation

A **depth-gated constant**, not a fitted law: λ = 0.5 for deep fits, 0 otherwise.
Justified entirely by real-data evidence, with the synthetic majority excluded on
the transfer failure above. Expected benefit ~1–2% on real deep log-loss fits;
worst observed real cost +0.70%.

Still unresolved before this can ship:
- **The depth threshold is unmeasured.** Only 6 and 12 were tested; everything
  between is interpolation, and the gate has to fire somewhere specific.
- **λ=0.5 has not been run against the regression and ranking guards** — the
  earlier guard sweep used 0.1, 1, 5 and 20.
- **Four real calibration datasets is thin** for any policy, including this one.

## Regression and ranking calibration, 2026-09-22

Eight arms (0, 0.1, 0.5, 1, 2, 5, 20, 50) × depths {6, 12} × 5 paired seeds over
7 real regression scenarios, 1 synthetic regression, 2 real-derived ranking and
1 synthetic ranking. Data: `benchmarks/results/accuracy_depth/calib_regrank/`.
The ladder was extended to 50 because squared error has Hessian ≡ 1.0, making λ
a phantom-row count against leaves the auto policy already fills with 8–16 real
rows; a ladder stopping at 20 could have reported "λ barely helps regression"
as an artefact of where it stopped.

**A sign error is corrected above.** NDCG is higher-is-better. The Task 5 verdict
read `california_ranking`'s raw percentage delta as if lower were better and
called a benefit a cost. Every ranking number below is orientation-corrected.

### Depth 6: λ does little for regression, and large λ harms it

Only 2 of 7 real regression scenarios improve at ≥4/5 seeds
(`abalone_regression` −1.71%, `panel_time_series` −3.95%, both at λ=50). Several
degrade at large λ: `california_housing` +2.43%, `bike_sharing` +4.15%,
`wine_quality_white` +2.93% at λ=50. The depth gate holds for regression as it
does for classification.

### Depth 12: real regression benefits modestly, and prefers a far larger λ

| scenario | best λ (≥4/5 seeds) | benefit | λ=20 |
| --- | ---: | ---: | ---: |
| `panel_time_series` | 50 | −5.91% | −5.22% (5/5) |
| `abalone_regression` | 50 | −3.89% | −1.97% (5/5) |
| `bike_sharing` | 50 | −1.74% | −1.19% (5/5) |
| `california_housing` | 20 | −1.12% | −1.12% (4/5) |
| `dense_numeric` | 0.1 | −1.04% | −1.04% (3/5) |
| `wine_quality_white` | 1 | −0.48% | +0.74% (1/5) |
| `dow_jones_financial` | none | — | +0.14% (2/5) |

λ=20 is the best single compromise: four solid improvements (−1.12% to −5.22%,
4–5/5 seeds) against two small harms (+0.14%, +0.74%).

### The optimal λ differs 40× between objectives

| objective | real-data compromise λ at depth 12 | typical benefit |
| --- | ---: | ---: |
| log-loss (classification) | **0.5** | 1–2% |
| squared error (regression) | **20** | 1–5% |

This is mechanical, not incidental. Squared error has `h ≡ 1`, so a leaf's
denominator is `rows + λ` and λ=0.5 against an 8–16 row floor is a 3–6% shrink.
Log-loss curvature `Σ p(1−p)` collapses as predictions sharpen, so the same λ can
dominate the denominator outright. **A single global λ cannot serve both**, which
is idea 4 (objective-aware auto policy) arriving as a requirement rather than a
hypothesis.

### Ranking is not calibratable from this evidence

The two real-derived ranking scenarios disagree at every λ above 0.1.
`california_ranking` improves (best −3.03% at λ=0.5, 4/5); `parkinsons_ranking`
has no setting reaching 4/5 seeds and degrades from λ=1 upward (+2.27% to
+5.34%). Seed agreement is 1–4/5 throughout against bands near 21%, so most of
this is noise. With two real-derived scenarios — one of which has query groups
constructed for the benchmark — there is no basis for a ranking λ default.
**Recommendation: leave ranking at λ=0, and treat it as uncalibrated rather than
as measured-neutral.**

### The synthetic/real gap replicates in regression

`histogram_stress` (the one synthetic regression scenario) gives −19.82% at λ=50
versus −1% to −6% across the seven real ones — the same order-of-magnitude
inflation seen in classification. The pattern now holds in both objectives and
is a property of the fixtures, not of one task type.

## Retraction of the objective-split claim, 2026-09-22

The regression/ranking write-up above concluded that optimal λ differs 40× "between
objectives" and that an objective-aware policy was therefore a requirement. **A
cross-check I ran the next day refutes it.** At depth 12, comparing λ=0.5 against
λ=20 on the same scenarios:

| scenario | group | λ=0.5 | λ=20 | prefers |
| --- | --- | ---: | ---: | --- |
| `adult_income` | CLS K=2 | −0.86% | **−4.25%** | 20 |
| `magic_gamma` | CLS K=2 | −1.93% | **−3.38%** | 20 |
| `covertype_multiclass` | CLS K=7 | **−2.38%** | +16.93% | 0.5 |
| `letter_recognition` | CLS K=26 | **+0.70%** | +76.09% | 0.5 |

Only 2 of 4 classification scenarios prefer the "log-loss" value. The two that
want λ=20 are the binary ones. The split is not objective versus objective — it
tracks **class count**, and binary log-loss behaves like squared error. The
plausible mechanism is effective curvature per leaf (`h ≡ 1` for squared error,
`p(1−p) ≤ 0.25` for binary, `≈(1/K)(1−1/K)` per class for K-way softmax), which
would make the right form a λ *relative* to typical leaf Hessian. I have not
verified that: dividing the observed λ\* by a crude leaf-Hessian estimate gives
ratios spanning 0.17 to 12.5, so the relative form does not fit as stated and
would need actual leaf-Hessian instrumentation to test.

## Frozen policy v1, 2026-09-22

Committed before any holdout measurement.

```
lambda_auto(objective, row_count, min_rows_per_leaf, max_depth):
    if user set lambda_l2 explicitly:        return user value   # never overridden
    if objective is ranking:                 return 0.0         # uncalibrated
    if 2**max_depth <= row_count / min_rows_per_leaf: return 0.0 # not deep enough
    return 0.5
```

**Why λ=0.5 and not the larger value that wins more often.** λ=20 improves 6 of 11
real scenarios at depth 12 versus λ=0.5's 5 — but its worst case is **+76.09%**
(`letter_recognition`) and **+16.93%** (`covertype_multiclass`), at every depth
tested. λ=0.5's worst case across all 11 real scenarios and all five depths at
which it fires is **+0.70%**. A rule that would route around the catastrophe needs
to condition on class count, and the only such rule I can fit rests on two
multiclass calibration scenarios with an arbitrary boundary. Taking the smaller,
safe constant costs perhaps 1–3% on regression and buys a bounded downside.

**Why the gate is a support criterion rather than a depth constant.** The effect is
a ramp, not a step: benefit grows monotonically with depth on `adult_income`,
`magic_gamma` and `covertype_multiclass`, and `california_housing` and
`bike_sharing` flip sign between depth 8 and 10. Choosing the depth that
maximises measured benefit would be fitting the cutoff to 11 scenarios. The
criterion `2^depth > rows / min_rows_per_leaf` instead fires exactly when the
tree's leaf capacity outruns the leaf support the data can provide at the floor,
and it reproduces the clean transitions: `bike_sharing` (13,903 rows, floor 16)
crosses at depth 9.76 and its benefit appears at depth 10; `california_housing`
(16,512, floor 16) crosses at 10.01 and its benefit appears at 12.

**Blast radius.** Default `max_depth` is 6 ([config.rs:245](../../crates/core/src/config.rs)).
At depth 6 the criterion fires only for datasets under ~64 rows, so default
behaviour is unchanged for every realistic dataset. This is opt-in by depth.

**Expected effect, stated before the holdout is read:** on deep real fits, roughly
1% median log-loss or RMSE improvement on about half of scenarios, worst case
about +0.7%. Prediction for the holdout: all four scenarios are multiclass K≥3 and
all cross the support threshold at depth 12 but not at depth 6, so the policy
should be exactly inert at depth 6 and give small improvements at depth 12, with
no scenario worse than roughly +1%.

**What this freeze does not cover.** The λ=20 branch is not in the policy and must
not ship without its own holdout — and note the holdout I built contains no
regression or binary scenario, so it cannot test that branch at all. That is a gap
in my holdout design: I chose it to test class-count generalisation before knowing
the policy's largest risk would be on the low-K side.
