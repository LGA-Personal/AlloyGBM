# GBMRegressor Parameters

`GBMRegressor` is the main Python estimator in AlloyGBM.

## Core Training Parameters

- `learning_rate: float = 0.1`
  - Step size for additive boosting updates.
- `max_depth: int = 6`
  - Maximum tree depth.
- `n_estimators: int = 6`
  - Number of boosting rounds requested.
- `row_subsample: float = 1.0`
  - Fraction of rows sampled per round.
- `col_subsample: float = 1.0`
  - Fraction of features sampled per round.

## Stopping And Training Policy

- `early_stopping_rounds: int | None = None`
  - Stops training early when validation progress stalls.
- `min_validation_improvement: float = 0.0`
  - Minimum validation improvement treated as meaningful.
- `training_policy: str = "auto"`
  - `auto` applies dataset-aware training heuristics.
  - `manual` preserves the requested controls more directly.

`training_policy="auto"` is the recommended default unless you are doing a
tight parameter ablation and want fewer adaptive adjustments.

## Reproducibility

- `seed: int = 0`
  - Random seed for training-time sampling.
- `deterministic: bool = True`
  - Keeps training deterministic when possible.

## Continuous Feature Handling

- `continuous_binning_strategy: str = "linear"`
  - One of `linear`, `rank`, or `quantile`.
- `continuous_binning_max_bins: int = 256`
  - Upper bound on bins used for continuous quantization.

Use `quantile` when you want more robust handling of skewed continuous feature
distributions. Use `linear` when you want the simplest and usually fastest
default.

## Categorical Support

- `categorical_feature_index: int | None = None`
  - Optional single categorical feature column.
- `categorical_smoothing: float = 20.0`
  - Smoothing strength for categorical target encoding.
- `categorical_min_samples_leaf: int = 1`
  - Minimum support for categorical leaf statistics.
- `categorical_time_aware: bool = False`
  - Enables time-aware categorical behavior when `time_index` is provided.

Current limitation: AlloyGBM supports only one categorical feature column at a
time.

## Diagnostics

- `store_node_stats: bool = False`
  - Stores optional node-level training statistics in the model artifact for
    later analysis.

This is useful for future diagnostics and introspection, but not required for
normal prediction.

## Main Methods

- `fit(X, y, *, categorical_feature_values=None, time_index=None)`
- `predict(X)`
- `shap_values(X, *, include_expected_value=False)`
- `feature_importances(X, *, method="shap")`
- `predict_from_artifact(artifact_bytes, X)`

See [Quickstart](quickstart.md) for an end-to-end example.
