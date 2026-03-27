# Benchmarks

This page summarizes how AlloyGBM is benchmarked and what the current results
say.

## Methodology

The comparative benchmark runner lives in `benchmarks/run_model_comparison.py`.

The suite currently compares AlloyGBM against:

- XGBoost
- LightGBM
- CatBoost

The expanded regression set currently includes:

- `dense_numeric`
- `california_housing`
- `bike_sharing`
- `panel_time_series`
- `dow_jones_financial`

Profiles are evaluated across shallow, mid, and deep configurations to show how
each library behaves under different learning-rate / depth / round budgets.

## Current Results

From the current expanded public regression suite:

- AlloyGBM is strongest on `panel_time_series`.
- AlloyGBM is strong on `dow_jones_financial`, especially under the deeper low-learning-rate profile.
- AlloyGBM is competitive but not leading on `dense_numeric`.
- AlloyGBM currently trails on `california_housing` and `bike_sharing`.
- In the latest recorded benchmark refresh, AlloyGBM was also the fastest trainer on most scenario/profile rows.

The latest recorded benchmark refresh also verified that the new training
contract and native dense preprocessing path did not collapse AlloyGBM quality:

- RMSE stayed unchanged on most AlloyGBM benchmark rows.
- The few changed rows were small and did not indicate a broad regression.
- Fit time improved materially versus the prior stored benchmark artifact.

The concise public summary should be:

- strong on `panel_time_series`
- strong on `dow_jones_financial`
- weaker on `california_housing` and `bike_sharing`

Representative best-RMSE results from the latest recorded comparison:

| Scenario | Best Model | Profile | Notes |
| --- | --- | --- | --- |
| `panel_time_series` | AlloyGBM | `shallow_high_lr` | Best result in the suite for this dataset |
| `dow_jones_financial` | AlloyGBM | `deep_low_lr` | Strongest finance-style showing |
| `dense_numeric` | CatBoost / XGBoost | `deep_low_lr` | Alloy remains competitive but behind |
| `california_housing` | XGBoost | `deep_low_lr` | Alloy still has a visible regression gap |
| `bike_sharing` | CatBoost | `mid_balanced` | Alloy improves with depth but does not lead |

## Stage Timing Output

The benchmark runner now breaks AlloyGBM fit time into:

- `input_adaptation_seconds`
- `native_bridge_prepare_seconds`
- `native_train_seconds`
- `fit_seconds`
- `predict_seconds`

Use those columns to distinguish preprocessing-heavy regressions from actual
trainer regressions.

## How To Run Them

Basic run:

```bash
python3 benchmarks/run_model_comparison.py --force-prepare
```

Expanded public regression set:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --scenarios california_housing bike_sharing dense_numeric panel_time_series dow_jones_financial \
  --profile-grid default \
  --profile-seeds 7
```

See the full runner guide in [benchmarks/README.md](../../benchmarks/README.md).

To inspect the stage timing output from the current public suite:

```bash
python3 benchmarks/run_model_comparison.py \
  --scenarios california_housing bike_sharing dense_numeric panel_time_series dow_jones_financial \
  --profile-grid default \
  --profile-seeds 7
```

## How To Interpret The Results

Use the benchmark suite to answer two different questions:

- Where is AlloyGBM already clearly strong?
- Where does it still lag established libraries?

That second question matters. The current suite is intentionally honest about
weak spots, especially on broader real-world regression datasets.
