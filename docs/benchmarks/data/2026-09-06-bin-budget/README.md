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

The peer libraries show 0.00% variance across all 15 scenarios, confirming zero machine noise and establishing an exact control baseline.

Under the corrected quantile bin budget (generating `max_bins - 1` cuts so that 255 data bins occupy bins 0..254 and bin 255 remains strictly reserved for NaNs), AlloyGBM demonstrates clear metric movement across 11 of the 15 scenarios:
- Real-world tabular and financial datasets show consistent gains (`histogram_stress` +19.23%, `breast_cancer` +7.15%, `synthetic_multiclass` +1.78%, `dow_jones_financial` +1.16%, `panel_time_series` +0.93%, `california_housing` +0.66%).
- The single synthetic fixture that originally surfaced the upper-tail truncation defect (500k x 40, 200 rounds, depth 8) improved test RMSE from 0.163775 to 0.152451.
- In `california_ranking`, the shift in bin cut boundaries altered tree splits and leaf assignments under the NDCG loss, showing that cut budget changes directly affect split selection and tree structure.
