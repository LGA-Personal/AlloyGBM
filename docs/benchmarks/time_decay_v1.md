# Time-decay weighting and the latest-periods holdout (v1)

Evidence for the opt-in `time_decay_half_life` parameter. Four public,
time-ordered datasets, rolling-origin evaluation, defaults everywhere else
(`n_estimators="auto"`), 2 seeds, 4 threads, container. Numerai has not been
run yet (its data is not reachable from the container), so none of this
changes a default.

Reproduce with `python benchmarks/time_decay_benchmark.py --seeds 2`. Raw
results: `perf-program/results/time-decay/time_decay_v1.{json,md}` in the
project's shared folder.

## Setup

| Dataset | Task | Rows | Period | Source |
|---|---|---:|---|---|
| elec2 | binary (price up) | 45,312 | day | NSW electricity, Harries 1999 |
| weather | binary (rain) | 18,159 | 30 days | NOAA daily, Elwell and Polikar 2011 |
| beijing_pm25 | regression (log1p PM2.5) | 41,757 | day | Liang et al. 2015 |
| household_power | regression (next-hour power) | 34,167 | week | Hebrail and Berard 2012 |

Each timeline is cut into 8 equal blocks of periods. Fold `k` (k = 0..3) trains
on every block before block `4 + k` and tests on that block, so every test set
is the future of its training data. Arms:

- `baseline`: no `time_index`; the auto-rounds holdout is a random 10% of rows.
- `time_holdout`: `time_decay_half_life=inf`: every weight is 1, but the
  auto-rounds holdout is the newest ~10% of periods.
- `hl=f`: half-life of `f` times the fold's training span (decay plus the
  latest-periods holdout).
- `hl=f/random`: the same decay with the random holdout
  (`ALLOYGBM_EXPERIMENT_TIME_DECAY_RANDOM_HOLDOUT=1`), to separate the two.

Deltas are paired by (fold, seed); "wins" counts the 8 pairs where the arm is
better.

## Results

Mean test loss (logloss for binary, RMSE for regression), delta against
`baseline`, and delta against `time_holdout`:

| Arm | elec2 | vs base | vs holdout | weather | vs base | vs holdout | PM2.5 | vs base | vs holdout | power | vs base | vs holdout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7929 | | | 0.4296 | | | 0.7045 | | | 0.4860 | | |
| time_holdout | **0.6000** | -0.193 (6/8) | | 0.4295 | -0.0002 (5/8) | | **0.6699** | -0.035 (8/8) | | **0.4851** | -0.0010 (8/8) | |
| hl=1 | 0.5906 | -0.202 (6/8) | -0.009 (6/8) | 0.4281 | -0.0015 (5/8) | -0.0013 (5/8) | 0.6715 | -0.033 (8/8) | +0.0016 (4/8) | 0.4856 | -0.0004 (4/8) | +0.0005 (4/8) |
| hl=0.5 | **0.5903** | -0.203 (6/8) | -0.010 (6/8) | **0.4278** | -0.0018 (6/8) | -0.0017 (6/8) | 0.6731 | -0.031 (8/8) | +0.0032 (3/8) | 0.4856 | -0.0005 (4/8) | +0.0005 (4/8) |
| hl=0.25 | 0.5933 | -0.200 (6/8) | -0.007 (4/8) | 0.4300 | +0.0004 (6/8) | +0.0006 (6/8) | 0.6790 | -0.026 (8/8) | +0.0090 (2/8) | 0.4859 | -0.0001 (5/8) | +0.0009 (3/8) |
| hl=0.125 | 0.5940 | -0.199 (6/8) | -0.006 (4/8) | 0.4393 | +0.0097 (5/8) | +0.0098 (5/8) | 0.7029 | -0.002 (4/8) | +0.0330 (0/8) | 0.4906 | +0.0045 (1/8) | +0.0055 (0/8) |
| hl=0.5/random | 0.8116 | +0.019 (3/8) | +0.212 (2/8) | 0.4279 | -0.0017 (6/8) | -0.0015 (6/8) | 0.7049 | +0.0003 (3/8) | +0.035 (0/8) | 0.4866 | +0.0006 (4/8) | +0.0016 (1/8) |

elec2 test AUC: baseline 0.811, time_holdout 0.822, hl=0.5 0.825.
The other `/random` arms are in the raw results; none beats its
latest-periods twin on elec2, PM2.5 or power.

Mean rounds chosen: elec2 1,454 (baseline) vs 53 (time_holdout); PM2.5 2,204
vs 233; power 254 vs 187; weather 227 vs 186.

## Reading it

- **The latest-periods holdout is the gain.** A random 10% holdout of
  autocorrelated rows sits right next to its training neighbours in time, so
  validation loss keeps falling long after the model stops generalising to the
  future, and the fit picks ten to forty times too many rounds. Holding out
  the newest periods fixes that: elec2 logloss -24% (AUC +0.010), PM2.5 RMSE
  -4.9% with 8/8 wins, household power -0.2% with 8/8 wins, weather unchanged.
- **Decay on top is small and data-dependent.** A half-life of half the
  training span adds another -0.010 logloss (+0.003 AUC) on elec2 and -0.4%
  logloss on weather (6/8), but costs +0.5% RMSE on PM2.5 and +0.1% on power.
  Short half-lives (1/8 of the span, effective sample size ~36% of rows) lose
  everywhere.
- **Decay without the time-ordered holdout does not help**: the `/random`
  arms keep the leaked round count and land at or above baseline on three of
  four datasets.
- Use `time_decay_half_life=inf` for the time-ordered holdout alone; try a
  half-life around half the training span only where the target's relation to
  the features drifts (elec2 is the textbook drift dataset).

## Not covered

- Numerai (the plan's test bed): eras as `time_index`, half-lives of none,
  208, 104 and 52 eras on the quick profile of `benchmarks/numerai_scale.py`.
- The holdout has no embargo between its training periods and the held-out
  newest periods; targets that overlap neighbouring periods (Numerai's 20-day
  targets) leak slightly across the cut.
