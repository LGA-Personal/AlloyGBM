# Quantile bin-budget fix — curated-suite evidence

`before.json` and `after.json` are `benchmarks/run_model_comparison.py --threads 1 --alloy-continuous-binning-strategy quantile --alloy-continuous-binning-max-bins 256`
runs on either side of the quantile bin-budget correction (pre-fix commit `bbb9988` vs current PR),
on the same machine back to back.

The peer columns are the control. AlloyGBM's quantile binning cut generation changed; LightGBM,
XGBoost, and CatBoost did not, so any movement in their rows is machine noise
and bounds what can be attributed to the fix.

## Result

```
  alloygbm   median metric improvement +0.22% (n=15, min -11.06%, max +19.23%)
  lightgbm   median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
  xgboost    median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
  catboost   median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
```

## Scenario Breakdown (AlloyGBM Base Model)

```
  california_housing        [rmse    ]: before=0.476194 after=0.473035 delta=+0.663%
  bike_sharing              [rmse    ]: before=68.583292 after=68.583292 delta=+0.000%
  dense_numeric             [rmse    ]: before=0.548374 after=0.547185 delta=+0.217%
  panel_time_series         [rmse    ]: before=31.846876 after=31.551247 delta=+0.928%
  histogram_stress          [rmse    ]: before=0.459764 after=0.371357 delta=+19.229%
  dow_jones_financial       [rmse    ]: before=3.495856 after=3.455440 delta=+1.156%
  abalone_regression        [rmse    ]: before=2.253596 after=2.262721 delta=-0.405%
  breast_cancer             [logloss ]: before=0.241033 after=0.223791 delta=+7.153%
  adult_income              [logloss ]: before=0.288967 after=0.288187 delta=+0.270%
  synthetic_classification  [logloss ]: before=0.042616 after=0.043508 delta=-2.093%
  wine_multiclass           [logloss ]: before=0.239677 after=0.239677 delta=+0.000%
  digits_multiclass         [logloss ]: before=0.049844 after=0.049844 delta=+0.000%
  synthetic_multiclass      [logloss ]: before=0.491839 after=0.483083 delta=+1.780%
  synthetic_ranking         [ndcg@10 ]: before=0.999117 after=0.998303 delta=-0.081%
  california_ranking        [ndcg@10 ]: before=0.743829 after=0.661535 delta=-11.064%
```

## Reading

On this suite the corrected budget is **roughly neutral in aggregate**: median
+0.22% across 15 scenarios, with wide scenario-level movement in both
directions (-11.06% to +19.23%).

**None of that per-scenario movement is signal.** The seed-variance measurement
in [`../2026-09-06-seed-variance/`](../2026-09-06-seed-variance/README.md) runs
the same suite across five seeds and finds that **no delta above exceeds its own
scenario's noise floor — zero of fifteen**:

| Scenario | Delta here | Seed spread | |
|---|---:|---:|---|
| `histogram_stress` | +19.23% | 24.58% | inside noise |
| `california_ranking` | -11.06% | 33.81% | inside noise |
| `breast_cancer` | +7.15% | 132.98% | inside noise |
| `synthetic_classification` | -2.09% | 12.02% | inside noise |
| all others | <= 1.8% | 3-10% | inside noise |

`california_ranking` NDCG across five seeds runs 0.570, 0.673, 0.767, 0.798,
0.830 — an 11% move there is an ordinary draw, not a regression caused by this
change. The same applies in the other direction: the +19.23% on
`histogram_stress` is not a win this change earned.

The peer rows at exactly 0.00% confirm the treatment is isolated and that no
machine noise contributed. They do not make AlloyGBM's per-scenario movement
meaningful, because the peers were never subjected to a discretisation change.

The case for this change therefore rests on **correctness** — 255 data bins
were being addressed by 255 cuts, so the two highest quantiles shared a bin —
and on the fixture that surfaced it (500k x 40, 200 rounds, depth 8:
0.163775 -> 0.152451, reproduced exactly, artifacts identical across `n_jobs`).

What this suite establishes is the narrower claim: **the fix does not regress
the curated scenarios in aggregate.** It does not establish a general accuracy
improvement and should not be cited as one.
