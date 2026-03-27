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
  - Stops training early when validation RMSE progress stalls.
- `min_validation_improvement: float = 0.0`
  - Minimum validation improvement treated as meaningful.
- `training_policy: str = "auto"`
  - `auto` applies dataset-aware training heuristics.
  - `manual` preserves the requested controls more directly.

`training_policy="auto"` is the recommended default unless you are doing a
tight parameter ablation and want fewer adaptive adjustments.

Early stopping is explicit-only. If `early_stopping_rounds` is set, you must
call `fit(..., eval_set=(X_valid, y_valid))`.

## Leaf And Split Controls

- `min_data_in_leaf: int = 1`
  - Minimum number of training rows allowed in a leaf. When
    `training_policy="auto"`, the engine may increase this value based on
    dataset size, but will never reduce it below the value you set.
- `lambda_l1: float = 0.0`
  - L1 regularization applied during split scoring.
- `lambda_l2: float = 0.0`
  - L2 regularization applied during split scoring.
- `min_child_hessian: float = 0.0`
  - Minimum child Hessian required for a split candidate.

These controls are part of the stable estimator surface. They are validated in
Python and plumbed through to the native trainer directly.

Current non-goal: AlloyGBM does not expose a `num_leaves` parameter yet. The
trainer remains depth-oriented rather than leaf-budget-oriented.

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

- `fit(X, y, *, eval_set=None, eval_time_index=None, categorical_feature_values=None, time_index=None)`
- `predict(X)`
- `shap_values(X, *, include_expected_value=False)`
- `feature_importances(X, *, method="shap")`
- `predict_from_artifact(artifact_bytes, X)`

Important `fit(...)` rules:

- If `early_stopping_rounds` is set, `eval_set` is required.
- If `eval_time_index` is passed, `eval_set` is required.
- If `categorical_time_aware=True`, training requires `time_index`, and
  validation also requires `eval_time_index` when `eval_set` is used.

## Post-Fit Attributes

After `fit(...)`, `GBMRegressor` may expose:

- `best_iteration_`
  - Best 0-based validation round, or `None` if no validation run happened.
- `best_score_`
  - Best validation RMSE, or `None` if no validation run happened.
- `n_estimators_`
  - Number of boosting rounds actually kept in the fitted model.
- `evals_result_`
  - Training summary shaped like `{"train": {"rmse": [...]}, "validation": {"rmse": [...]}}`.
- `fit_timing_`
  - Stage timing dictionary with:
    - `input_adaptation_seconds`
    - `native_bridge_prepare_seconds`
    - `native_train_seconds`
    - `total_fit_seconds`

See [Quickstart](quickstart.md) for an end-to-end example.
