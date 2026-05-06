# GBMRegressor Parameters

`GBMRegressor` is the main Python estimator for regression in AlloyGBM.

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
  - Stops training early when validation metric progress stalls.
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
- `min_split_gain: float = 0.0`
  - Minimum gain required for a split to be made. The auto training policy may
    set this adaptively; passing it explicitly overrides that.

## Tree Growth Strategy

- `tree_growth: str = "level"`
  - `level` grows trees level-by-level (depth-first). This is the default.
  - `leaf` grows trees leaf-by-leaf (best-first), selecting the leaf with the
    highest split gain at each step. This is similar to LightGBM's growth
    strategy and is often more efficient for the same leaf budget.
- `max_leaves: int | None = None`
  - Maximum number of leaves when using `tree_growth="leaf"`. If not set,
    defaults to `2^max_depth`.

## Constraints

- `monotone_constraints: list[int] | dict[int, int] | None = None`
  - Constrains features to be monotonically increasing (+1), decreasing (-1),
    or unconstrained (0). Pass a list with one entry per feature, or a dict
    mapping feature indices to constraints.
- `feature_weights: list[float] | dict[int, float] | None = None`
  - Per-feature importance weights that influence split selection. Higher
    weights make a feature more likely to be chosen as a split candidate.

## Reproducibility

- `seed: int = 0`
  - Random seed for training-time sampling.
- `deterministic: bool = True`
  - Keeps training deterministic when possible.

## Continuous Feature Handling

- `continuous_binning_strategy: str = "linear"`
  - One of `linear`, `rank`, or `quantile`.
- `continuous_binning_max_bins: int = 256`
  - Upper bound on bins used for continuous quantization. Supports up to 65,535
    bins. Higher bin counts may improve accuracy on high-cardinality features
    at the cost of additional memory.

Use `quantile` when you want more robust handling of skewed continuous feature
distributions. Use `linear` when you want the simplest and usually fastest
default.

## Categorical Support

- `categorical_feature_index: int | None = None`
  - Single categorical feature column (legacy interface, still supported).
- `categorical_feature_indices: list[int] | None = None`
  - Multiple categorical feature columns. Each listed column is treated as
    categorical during training.
- `categorical_smoothing: float = 20.0`
  - Smoothing strength for categorical target encoding.
- `categorical_min_samples_leaf: int = 1`
  - Minimum support for categorical leaf statistics.
- `categorical_time_aware: bool = False`
  - Enables time-aware categorical behavior when `time_index` is provided.
- `max_cat_threshold: int = 0`
  - Maximum number of categories for native categorical splits. When a
    categorical feature has at most this many unique values, AlloyGBM uses the
    Fisher-sort algorithm to find the optimal binary partition in O(K log K)
    time and encodes it as a compact bitset for O(1) prediction. Features
    exceeding this threshold fall back to target encoding. Default 0 disables
    native categorical splits entirely (all categoricals use target encoding).

When both `categorical_feature_index` and `categorical_feature_indices` are
provided, they are merged.

## MorphBoost (Adaptive Split Criterion)

`GBMRegressor` supports an opt-in MorphBoost training mode that augments the
standard gradient gain with an information-theoretic term, EMA-driven gain
shaping, and depth/iteration leaf shrinkage. See
[MorphBoost](morphboost.md) for the full parameter reference and the
[paper](https://arxiv.org/pdf/2511.13234) for the formulation.

- `training_mode: str = "auto"`
  - `auto` (default): standard training with dataset-aware policy heuristics.
  - `manual`: standard training, applies user-supplied controls verbatim.
  - `morph`: enable MorphBoost.
- `morph_rate: float = 0.1`
  - Per-iteration leaf shrinkage rate when `training_mode="morph"`.
- `evolution_pressure: float = 0.2`
  - Strength of EMA-driven gain shaping when `training_mode="morph"`.
- `morph_warmup_iters: int = 5`
  - Initial rounds for which the morph blend collapses to the pure
    gradient gain.
- `info_score_weight: float = 0.3`
  - Mixing weight for the information-theoretic term post-warmup. Set to
    `0.0` to disable the info-theoretic term entirely.
- `depth_penalty_base: float = 0.9`
  - Base of the leaf depth penalty applied as
    `depth_penalty_base ** (child_depth / 3.0)`.
- `balance_penalty: bool = True`
  - Penalize highly imbalanced splits.
- `lr_schedule: str = "constant"`
  - Per-iteration learning-rate schedule. One of `constant` or
    `warmup_cosine`. Independent of `training_mode` — usable on its own.
- `lr_warmup_frac: float = 0.1`
  - Fraction of `n_estimators` spent in the linear-warmup phase when
    `lr_schedule="warmup_cosine"`. Range `[0.0, 1.0]`.

## Warm-Starting

- `warm_start: bool = False`
  - When `True`, calling `fit()` continues training from the previously fitted
    model instead of starting from scratch. This enables incremental training.

## Diagnostics

- `store_node_stats: bool = False`
  - Stores optional node-level training statistics in the model artifact for
    later analysis.

## Main Methods

- `fit(X, y, *, sample_weight=None, eval_set=None, eval_sample_weight=None, group=None, eval_group=None, eval_time_index=None, categorical_feature_values=None, time_index=None)`
- `predict(X)`
- `shap_values(X, *, include_expected_value=False)`
- `feature_importances(X, *, method="shap")`
- `predict_from_artifact(artifact_bytes, X)`
- `save_model(path)`
- `load_model(path)` (classmethod)
- `artifact_bytes` -- property returning the raw artifact bytes
- `score(X, y)` -- returns R-squared (sklearn `RegressorMixin` convention)

Important `fit(...)` rules:

- If `early_stopping_rounds` is set, `eval_set` is required.
- If `eval_time_index` is passed, `eval_set` is required.
- If `categorical_time_aware=True`, training requires `time_index`, and
  validation also requires `eval_time_index` when `eval_set` is used.
- `sample_weight` applies per-sample weights to the training loss.

## Post-Fit Attributes

After `fit(...)`, `GBMRegressor` may expose:

- `best_iteration_`
  - Best 0-based validation round, or `None` if no validation run happened.
- `best_score_`
  - Best validation metric, or `None` if no validation run happened.
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
- `feature_names_`
  - Feature names captured from the training data (when available), or
    auto-generated as `f0`, `f1`, etc.

See [Quickstart](quickstart.md) for an end-to-end example.
