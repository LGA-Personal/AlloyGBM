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
