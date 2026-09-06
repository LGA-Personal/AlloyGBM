# Quantile bin-budget fix — curated-suite evidence

`before.json` and `after.json` are `benchmarks/run_model_comparison.py --threads 1`
runs on either side of the quantile bin-budget correction, on the same machine
back to back.

The peer columns are the control. AlloyGBM's binning changed; LightGBM,
XGBoost, and CatBoost did not, so any movement in their rows is machine noise
and bounds what can be attributed to the fix.

## Result

```
  alloygbm   median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
  lightgbm   median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
  xgboost    median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
  catboost   median metric improvement +0.00% (n=15, min +0.00%, max +0.00%)
```

## Reading

The single synthetic fixture that surfaced the defect (500k x 40, 200 rounds,
depth 8) improved test RMSE from 0.163775 to 0.152451. That fixture is
Gaussian, and the merged region is the upper tail, where its interaction term
carries the most signal — so it is expected to sit at the favourable end of the
range. This suite is the check on whether the fix generalises to real data and
skewed distributions.
