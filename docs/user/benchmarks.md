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
- LightGBM is usually the fastest trainer in the comparison set.

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

## How To Interpret The Results

Use the benchmark suite to answer two different questions:

- Where is AlloyGBM already clearly strong?
- Where does it still lag established libraries?

That second question matters. The current suite is intentionally honest about
weak spots, especially on broader real-world regression datasets.
