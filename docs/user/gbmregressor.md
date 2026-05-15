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
- `interaction_constraints: list[list[int]] | None = None`
  - LightGBM-compatible interaction constraints. Each inner list is a group
    of feature indices; any root-to-leaf path is restricted to splits on
    features from a single still-active group. Up to 64 groups per fit;
    enforced through both the level-wise and leaf-wise tree builders.
    Features that appear in no group are allowed everywhere.

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

## DRO Leaf Solver

- `leaf_solver: str = "standard"`
  - `"standard"` (default): the usual scalar Newton leaf update.
  - `"dro"`: a fast robust scalar update that penalizes weak leaf signal by
    within-leaf gradient dispersion before solving the leaf value.
- `dro_radius: float = 0.05`
  - Non-negative radius scaling the gradient-uncertainty penalty. `0.0`
    preserves standard-leaf predictions while recording DRO metadata.
- `dro_metric: str = "wasserstein"`
  - Accepted value for v0.7.1. It denotes the Wasserstein-inspired
    closed-form robust counterpart over leaf gradient uncertainty.

The v0.7.1 DRO solver is intentionally conservative: it is not a full
Wasserstein optimizer over raw feature/target distributions and does not claim
guaranteed live-market stability. It modifies split gain and final scalar leaf
values consistently using the same robust effective gradient. Inference speed is
unchanged because leaf values are baked into the artifact.

`leaf_solver="dro"` works on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`,
and composes with `training_mode="morph"`. It requires
`leaf_model="constant"` in v0.7.1; `leaf_model="linear"` continues to use the PL
leaf solver.

## Factor-Neutral Boosting

- `neutralization: str = "none"`
  - One of `"none"`, `"pre_target"`, `"per_round_gradient"`, or
    `"split_penalty"`.
- `factor_neutralization_lambda: float = 1e-6`
  - Finite, non-negative ridge term added to `F^T W F`.
- `factor_penalty: float = 0.0`
  - Finite, non-negative split exposure penalty scale. Only active for
    `neutralization="split_penalty"`.

Pass factors as fit-time data:

```python
model = GBMRegressor(neutralization="per_round_gradient", seed=7)
model.fit(X_train, y_train, factor_exposures=F_train)
```

`factor_exposures` must be dense, row-major, finite, and shaped
`(n_rows, n_factors)`. It is not stored as an estimator constructor parameter,
so sklearn cloning remains clean and large matrices are not embedded in
estimator params.

Mode semantics:

`neutralization="none"` preserves current behavior and ignores
`factor_exposures` unless a non-`None` matrix is provided with an inactive mode,
in which case Python raises a clear validation error to prevent silent user
mistakes.

`neutralization="pre_target"` residualizes the regression target once before
training:

```text
y_perp = y - F (F^T W F + lambda I)^-1 F^T W y
```

This mode is supported for `GBMRegressor` only. It is rejected for
classification and ranking because target residualization is not well-defined
for class labels or ranking relevance. `eval_set` is also rejected for
`pre_target` in this release because the public API does not yet accept
validation-set factor exposures to residualize validation targets consistently.

`neutralization="per_round_gradient"` projects objective gradients before each
boosting round:

```text
g_perp = g - F (F^T W F + lambda I)^-1 F^T W g
```

Hessians are unchanged. This mode is supported for regression, binary
classification, multiclass, and ranking. For multiclass, each class-gradient
column is projected independently against the same factor projector.

`neutralization="split_penalty"` includes per-round gradient projection and
subtracts a factor-load penalty from split gain:

```text
penalty = factor_penalty * || F_L^T update_L + F_R^T update_R ||^2 / max(row_count, 1)
gain_final = gain_after_existing_modes - penalty
```

For scalar leaves, `update_L` and `update_R` are the candidate scalar leaf
values before any final MorphBoost depth/iteration leaf scaling. For DRO
leaves, the scalar values use the DRO effective gradients. For MorphBoost, the
order is: project gradients, compute standard/DRO gradient gain, blend
MorphBoost information score, subtract factor penalty, then apply MorphBoost
leaf scaling when storing leaves. `split_penalty` performs additional
factor-exposure work during split search and should be treated as the slowest
neutralization mode until production-scale benchmarks justify stronger claims.

Compatibility:

| Feature | pre_target | per_round_gradient | split_penalty |
| --- | --- | --- | --- |
| `GBMRegressor` | supported | supported | supported |
| `GBMClassifier` | rejected | supported | supported |
| `GBMRanker` | rejected | supported | supported |
| `training_mode="morph"` | supported | supported | supported |
| `leaf_solver="dro"` | supported | supported | supported |
| `leaf_model="linear"` | supported | supported | rejected |
| warm start | supported | supported | supported |

This is a training-time regularization tool. It does not guarantee
prediction-time zero exposure unless predictions are neutralized against
evaluation-time factors outside the model.

Exposure matrices are not persisted in the estimator or artifact. As of
v0.7.1, neutralized warm-start and `init_model` continuation are supported
across all three modes: the caller must supply the same `factor_exposures`
matrix used for the initial fit, and `neutralization`,
`factor_neutralization_lambda`, and (for `split_penalty`) `factor_penalty`
must match the persisted contract. Mismatches raise a clear "does not
match" error. `pre_target` neutralization is idempotent under repeated
residualization against the same exposures, so warm-start continuation
residualizes the original targets again on the resumed fit and trains on
the same target stream as a fresh `N + M`-round fit.

## Piecewise-Linear Leaves

- `leaf_model: str = "constant"`
  - `"constant"` (default): standard scalar leaf value — identical to all prior
    AlloyGBM behaviour.
  - `"linear"`: each leaf stores a small linear model
    `f_s(x) = b_s + Σ α_j x_j` (up to 8 regressors per leaf, inherited from the
    split path's feature indices; the per-leaf cap is internal and not
    user-tunable in v0.7.1). Optimal weights are solved in closed form:
    `α* = -(XᵀHX + λI)⁻¹ Xᵀg`, regularised by the same `lambda_l2` you pass to
    the estimator.

  **When to use `"linear"`**: datasets where the residual signal within each tree
  node is approximately linear in the input features (e.g. smooth tabular
  regression, classification with well-separated linear decision boundaries).
  Benchmarks show ~10× faster convergence on linearly-structured data, +3.5%
  RMSE on California Housing, and +1.75pp accuracy on Breast Cancer, at a 2–8×
  training time overhead.

  **Recommended `lambda_l2`**: pass at least `lambda_l2=0.01` when using
  `leaf_model="linear"`. Without it, the closed-form ridge solve can produce
  noisy per-leaf weights that overfit at high round counts on non-linear data.

  **Multi-class softmax**: when `GBMClassifier` is fit with K > 2 classes, each
  per-class tree sequence independently uses linear leaves.

  **Limitations**:

  - Training time scales with the number of regressors per node (≤ 8×8
    Cholesky solve).
  - Native-bitset categorical features that use Fisher-sort splits
    (`max_cat_threshold > 0`) fall back to constant leaves for that split node;
    descendant leaves below such a split use linear leaves on all remaining
    numeric regressors.
  - SHAP (`shap_values`, `feature_importances`) supports
    `leaf_model="linear"` as of v0.7.1, returning a best-effort
    interventional decomposition (path-attributed leaf "constant part" +
    per-leaf row deviations against persisted global feature means). Strict
    additivity is relaxed for continuous-feature PL artifacts because SHAP's
    internal path walker still compares feature values against bin-index
    thresholds; tightening this is queued for v0.7.2. See
    [explanations.md](explanations.md) and
    [../limitations.md](../limitations.md) for details.

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

- `fit(X, y, *, sample_weight=None, eval_set=None, eval_sample_weight=None, group=None, eval_group=None, eval_time_index=None, categorical_feature_values=None, time_index=None, factor_exposures=None)`
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
- `diagnostics_per_round_`
  - List of per-round dicts containing `gradient_l2_norm`,
    `gradient_variance`, `hessian_l2_norm`, sampling counts
    (`n_active_rows`, `n_active_features`), and (when factor neutralization
    is active) `neutralization_effectiveness` in `[0, 1]`.
- `stop_reason_` / `rounds_completed_`
  - Engine's early-stop reason and actual committed round count.

See [Quickstart](quickstart.md) for an end-to-end example.
