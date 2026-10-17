# Bounded log-link Hessians and gradient-regularized Newton (v1)

Dev-suite evidence for the opt-in `glm_hessian="bounded"`, `grn_strength` and
`grn_form` parameters. Dev suite only (seeded 75/25 splits, 3 seeds, training
rows capped at 5,000, 4 threads, container); the held-out OpenML run has not
been done, so none of this changes a default.

Raw results: `perf-program/results/grn-loglink/{poisson,gamma,tweedie,full_dev}.json`
in the project's shared folder. Each table was produced with
`python benchmarks/special_modes.py RESULT.json --baseline <arm>`.

## Setup

- Poisson and Tweedie (`p = 1.5`): the 27 dev regression tasks with `y >= 0`.
  Gamma: the 18 with `y > 0`. Metric is test RMSE of the predicted mean.
- Baseline is the objective at defaults (auto rounds, exact Hessian, Poisson
  with `poisson_max_delta_step=0.7`). Every ratio below is arm / baseline:
  below 1.0 means lower loss or a faster fit. W/T/L is per cell, paired by seed
  (a tie is inside two seed standard errors or under 0.5%).
- GRN arms use `grn_form="per_row"` unless marked `leaf`.

## Results

| Arm | Poisson loss | W/T/L | Time | Gamma loss | W/T/L | Time | Tweedie loss | W/T/L | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded | **0.966** | 10/13/4 | 0.77x | **0.965** | 3/15/0 | 1.14x | 0.991 | 1/25/1 | 0.92x |
| lr 0.3 | 1.013 | 9/14/4 | 0.51x | 1.145 | 2/8/8 | 0.46x | 1.264 | 2/10/15 | 0.37x |
| bounded, lr 0.3 | 0.980 | 10/11/6 | 0.42x | 0.988 | 2/10/6 | 0.48x | 1.016 | 1/14/12 | 0.37x |
| GRN 0.3 | 1.023 | 0/23/4 | 1.01x | 0.974 | 2/16/0 | 1.16x | 1.002 | 0/27/0 | 1.05x |
| GRN 1.0 | 0.994 | 0/23/4 | 1.08x | 0.973 | 1/17/0 | 1.37x | 1.001 | 1/25/1 | 1.23x |
| GRN 1.0, leaf | 0.998 | 1/25/1 | 1.01x | 0.970 | 1/17/0 | 1.10x | 0.999 | 1/24/2 | 1.02x |
| GRN 1.0, lr 0.3 | 0.995 | 9/17/1 | 0.56x | 0.967 | 3/9/6 | 0.55x | 1.101 | 0/15/12 | 0.44x |

## Findings

- **Bounded Hessians are the stronger change.** At the default learning rate
  they cut loss 3.4% on Poisson and 3.5% on Gamma and tie on Tweedie, with no
  cell worse than 1.3% except Poisson `visualizing_soil` (2.03x: every seed
  stops at 59 trees at RMSE 0.62, while the baseline's three seeds range
  0.20 to 0.46). Best cells: Poisson SGEMM 0.38x, Gamma SAT11 runtime 0.61x,
  Tweedie Buzz-in-social-media 0.84x.
- **They also make a high learning rate safe.** Raising the learning rate to
  0.3 with exact Hessians costs 1.3% on Poisson, 14.5% on Gamma and 26% on
  Tweedie (Tweedie Buzz-in-social-media blows up 68x, Gamma Bike Sharing 3.5x).
  With bounded Hessians the same 0.3 runs fit in 0.37 to 0.48x the time at
  0.980 / 0.988 / 1.016 of the default-rate baseline, which is the "same loss
  with fewer rounds" win the plan asked for.
- **GRN at the default learning rate is a tie on Poisson and Tweedie** and a
  2.7% Gamma gain that the bounded Hessian already captures. At learning rate
  0.3 it removes most of the Gamma damage (1.145 to 0.967) but only part of the
  Tweedie damage (1.264 to 1.101), so it is a weaker stabilizer than the
  bounded Hessian on these objectives. The per-leaf form barely moves anything
  because a leaf's Hessian sum dwarfs `λ`.
- **Where GRN does matter is a vanishing Hessian.** On a Charbonnier custom
  objective at learning rate 1.0 (the paper's Figure 1 setting, pinned by
  `test_grn_per_row_rescues_charbonnier_newton_at_learning_rate_one`), plain
  Newton reaches a training loss around 3.8e5 while `grn_strength=1.0,
  grn_form="per_row"` converges to about 0.1. `grn_form="per_leaf"` only gets
  to 88 at strength 1.0.

## Full dev suite, default objectives

GRN at strength 1.0 against the default objective on all 68 dev tasks (same
setup, baseline `alloy` at defaults):

| Arm | Slice | Cells | Loss | W/T/L | Time |
|---|---|---:|---:|---:|---:|
| GRN 1.0 | all | 68 | 1.139 | 4/53/11 | 1.61x |
| GRN 1.0 | binary | 26 | 0.994 | 3/23/0 | 1.09x |
| GRN 1.0 | multiclass | 9 | 2.055 | 0/3/6 | 2.81x |
| GRN 1.0 | regression | 33 | 1.079 | 1/27/5 | 1.87x |
| GRN 1.0, leaf | all | 68 | 1.003 | 4/60/4 | 1.08x |
| GRN 1.0, leaf | binary | 26 | 0.999 | 1/25/0 | 1.06x |
| GRN 1.0, leaf | multiclass | 9 | 0.996 | 1/7/1 | 1.03x |
| GRN 1.0, leaf | regression | 33 | 1.007 | 2/28/3 | 1.11x |

- The per-row form hurts multiclass badly (shuttle 26x, KDDCup99 13x) and a
  few regression tasks (tecator 2.8x, house_prices_nominal 2.6x). On
  near-separable classification most row Hessians `p(1-p)` are tiny, so a
  `λ` set by the gradient norm dominates them and every step shrinks until
  the round budget runs out. Its best cells are small binary gains
  (Amazon employee access 0.92x, APSFailure 0.93x).
- The per-leaf form is a tie everywhere (worst cell house_prices_nominal
  1.32x, best Buzz-in-social-media 0.91x).
- So GRN is not a general default for any form. It belongs on losses whose
  Hessian can vanish for large residuals, like the Charbonnier case above.

## Recommendation

Take `glm_hessian="bounded"` to the held-out OpenML run (the CTR23 tasks with
non-negative targets) as a candidate default for the log-link objectives.
Keep GRN opt-in: it is for custom losses whose Hessian can vanish, and
`grn_form="per_row"` is the form that works there. Do not turn it on for
classification objectives, where the per-row form underfits.
