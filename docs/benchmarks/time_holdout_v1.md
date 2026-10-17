# Latest-periods holdout for automatic rounds (v1)

Date: 2026-10-17. Commit: 10198b1 on `claude/time-decay-83g1z4`. Machine:
4-core Linux container, default `n_jobs`.

## Question

With `n_estimators="auto"` and no `eval_set`, the round count is chosen on
a 10% holdout. When rows are ordered in time, a random holdout leaks: rows
next to each other in time look alike, so the holdout rewards rounds that
only memorise the neighbours of the held-out rows. Does holding out the
newest periods instead (`fit(X, y, time_index=period)`) predict the future
better, and does it cost anything on data with no time structure?

## Method

`benchmarks/time_holdout_benchmark.py`, defaults: rolling origin with 10
equal blocks of periods. Folds test on blocks 5 to 9, and each fold trains
on every earlier block. There are 5 seeds per dataset, and every fit uses
the defaults. The arms are `random` (`fit(X, y)`) and `time`
(`fit(X, y, time_index=period)`).

The unit of comparison is the test period. Each period's mean loss is
averaged over seeds, and the 95% interval for the mean paired difference
comes from a moving-block bootstrap over test periods (block length n^(1/3)).
That way serial correlation between neighbouring periods does not inflate
confidence. Delta is relative to the `random` arm, and negative favours
`time`.

The datasets are:

- the four public time-ordered sets from GitHub mirrors (see the script
  docstring);
- synthetic panels with a known relationship, regenerated per seed (80
  periods of 250 rows, 10 features, noise variance 1):
  - `synth_drift=s`: the coefficients take random-walk steps of size s each
    period;
  - `synth_jump`: one abrupt regime change at period 64.

## Results

| Dataset | Metric | Random | Time | Delta | 95% CI | Periods won | Folds won | Trees (random / time) | Fit s (random / time) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| weather | log loss | 0.43012 | 0.42988 | -0.06% | [-0.23%, +0.11%] | 48% | 12 / 25 | 245 / 201 | 1.2 / 1.0 |
| elec2 | log loss | 0.75916 | 0.57704 | -23.99% * | [-29.86%, -18.38%] | 65% | 25 / 25 | 1483 / 41 | 10.9 / 0.5 |
| beijing_pm25 | MSE | 0.49515 | 0.44966 | -9.19% * | [-11.58%, -6.83%] | 66% | 25 / 25 | 2175 / 234 | 10.6 / 1.3 |
| household_power | MSE | 0.23665 | 0.23599 | -0.28% | [-0.67%, +0.10%] | 50% | 15 / 25 | 246 / 159 | 1.2 / 0.8 |
| synth_drift=0 | MSE | 1.07274 | 1.07349 | +0.07% | [-0.02%, +0.18%] | 48% | 11 / 25 | 213 / 200 | 0.9 / 0.8 |
| synth_drift=0.03 | MSE | 1.11290 | 1.11307 | +0.02% | [-0.16%, +0.17%] | 50% | 14 / 25 | 220 / 249 | 1.0 / 1.1 |
| synth_drift=0.1 | MSE | 1.47100 | 1.46801 | -0.20% * | [-0.39%, -0.02%] | 60% | 14 / 25 | 272 / 296 | 1.1 / 1.2 |
| synth_jump | MSE | 2.47279 | 2.12766 | -13.96% * | [-28.58%, -1.48%] | 40% | 12 / 25 | 178 / 158 | 0.8 / 0.7 |

`*`: the 95% interval excludes zero. Raw per-fold rows are in
`/mnt/project-files/perf-program/results/time-holdout/`.

## Reading

- **Large wins where neighbouring rows are near-duplicates.**
  - On elec2 (half-hourly prices) the random holdout ran to 1,100 to 2,000
    trees, against about 20 to 40 with the time holdout. Log loss fell 24%
    and won all 25 folds.
  - On PM2.5 (hourly) the random holdout hit the 2,000-round cap on almost
    every fold. MSE fell 9.2% and won all 25 folds.
  - Both also fit 8 to 22 times faster, because the holdout pass stops early.
- **No cost when there is no time structure.**
  - On `synth_drift=0` the rows are i.i.d. across periods. The delta is
    +0.07%, with an interval of -0.02% to +0.18%.
  - On `synth_drift=0.03` it is +0.02%.
  - Weather (rain once per 30 days, little autocorrelation) and household
    power are within noise.
- **Drift helps a little.** `synth_drift=0.1` gains 0.2%, and the interval
  just excludes zero.
- **Regime change is the noisy case.**
  - In `synth_jump` fold 4, the holdout lands entirely after the change. The
    old regime then does not transfer, so the time holdout stops at 1 to 9
    trees: an honest "the past does not help". That beat the random arm's
    150 trees fitted to the old regime by a wide margin.
  - Elsewhere the two arms tie. The mean gain (-14%) comes from that one
    fold, which is why the interval is wide and only 40% of periods are won.

## Not tested

Numerai. Its targets span 4 weekly eras, so the last training eras' targets overlap the first
held-out eras; the latest-eras holdout has no embargo and leaks slightly
across the cut.
