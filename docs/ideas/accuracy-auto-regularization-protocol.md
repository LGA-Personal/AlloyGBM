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
