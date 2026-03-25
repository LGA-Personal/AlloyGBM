GBMRegressor
============

``GBMRegressor`` is the main Python estimator in AlloyGBM.

Core parameters
---------------

- ``learning_rate: float = 0.1``
  - additive update step size
- ``max_depth: int = 6``
  - maximum tree depth
- ``n_estimators: int = 6``
  - requested boosting rounds
- ``row_subsample: float = 1.0``
  - per-round row sampling fraction
- ``col_subsample: float = 1.0``
  - per-round feature sampling fraction

Stopping and policy controls
----------------------------

- ``early_stopping_rounds: int | None = None``
- ``min_validation_improvement: float = 0.0``
- ``training_policy: str = "auto"``

``training_policy="auto"`` applies dataset-aware heuristics and is the
recommended default for practical use. ``manual`` is more appropriate for
controlled ablation work.

Reproducibility
---------------

- ``seed: int = 0``
- ``deterministic: bool = True``

Continuous-feature controls
---------------------------

- ``continuous_binning_strategy: str = "linear"``
- ``continuous_binning_max_bins: int = 256``

Use ``quantile`` when you need more robust handling of skewed continuous
features. Use ``linear`` when you want the simplest and usually fastest default.

Categorical support
-------------------

- ``categorical_feature_index: int | None = None``
- ``categorical_smoothing: float = 20.0``
- ``categorical_min_samples_leaf: int = 1``
- ``categorical_time_aware: bool = False``

Current limitation: AlloyGBM supports only a single categorical feature column
at a time.

Diagnostics
-----------

- ``store_node_stats: bool = False``

This stores optional node-level debug statistics inside the artifact for later
analysis. It is not required for ordinary prediction.

Main methods
------------

- ``fit(X, y, *, categorical_feature_values=None, time_index=None)``
- ``predict(X)``
- ``shap_values(X, *, include_expected_value=False)``
- ``feature_importances(X, *, method="shap")``
- ``predict_from_artifact(artifact_bytes, X)``

Recommended usage pattern
-------------------------

For most users:

- start with ``training_policy="auto"``
- keep ``deterministic=True`` during evaluation
- use time-aware validation for temporal or panel-like problems
- use the benchmark suite to compare profile shapes rather than trusting a
  single run
