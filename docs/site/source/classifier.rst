GBMClassifier
=============

``GBMClassifier`` is the binary classification estimator in AlloyGBM.

Overview
--------

``GBMClassifier`` extends ``GBMRegressor`` with a binary cross-entropy
(log-loss) objective. Predictions are probabilities obtained via sigmoid
transform. When sklearn is available, ``GBMClassifier`` inherits
``ClassifierMixin`` for full pipeline compatibility.

Quick example
-------------

.. code-block:: python

   from alloygbm import GBMClassifier, accuracy, log_loss

   model = GBMClassifier(
       learning_rate=0.05,
       max_depth=6,
       n_estimators=500,
       deterministic=True,
       seed=7,
   )
   model.fit(X_train, y_train)

   labels = model.predict(X_test)
   probas = model.predict_proba(X_test)

   print("accuracy:", accuracy(y_test, labels))
   print("log_loss:", log_loss(y_test, probas[:, 1]))

Parameters
----------

All parameters from ``GBMRegressor`` are accepted, including
``leaf_solver="dro"`` for robust scalar leaves, ``leaf_model="linear"`` for
piecewise-linear leaves (see :doc:`estimator`),
``neutralization="per_round_gradient"`` / ``"split_penalty"`` with
``factor_exposure_transform`` for factor-neutral training (active
``split_penalty`` defaults to effective ``"standardize"`` preprocessing), and
``training_mode="morph"`` and the MorphBoost / LR-schedule parameters
(``morph_rate``, ``evolution_pressure``, ``morph_warmup_iters``,
``info_score_weight``, ``depth_penalty_base``, ``balance_penalty``,
``lr_schedule``, ``lr_warmup_frac``). See :doc:`morphboost` for the full
reference. ``leaf_model="linear"`` and ``training_mode="morph"`` can be
combined. Multi-class softmax fits each per-class tree sequence with linear
leaves independently. The objective is always cross-entropy and is not
configurable.

``boosting_mode="goss"`` with ``goss_top_rate`` / ``goss_other_rate``
and ``boosting_mode="dart"`` with ``dart_drop_rate`` /
``dart_max_drop`` / ``dart_normalize_type`` / ``dart_sample_type`` are
both supported on **binary** classification (see :doc:`estimator`
"Boosting mode" for the full semantics).  Multi-class softmax
explicitly rejects non-``"standard"`` boosting modes pending
per-class gradient scoring (v0.10.x follow-up — applies to both GOSS
and DART).

Target requirements:

- ``y`` must contain only values in ``{0, 1}`` (or ``{0.0, 1.0}``)
- Both classes must be present in the training targets

Methods
-------

- ``fit(X, y, *, sample_weight=None, eval_set=None, ...)`` -- trains the
  classifier. Returns ``self``.
- ``predict(X)`` -- returns class labels (0 or 1) by thresholding at 0.5.
- ``predict_proba(X)`` -- returns array of shape ``(n_samples, 2)`` with
  columns ``[P(y=0), P(y=1)]``.
- ``predict_log_proba(X)`` -- returns log-probabilities.

Post-fit attributes
-------------------

In addition to the standard ``GBMRegressor`` post-fit attributes:

- ``classes_`` -- always ``[0, 1]``
- ``n_classes_`` -- always ``2``

sklearn compatibility
---------------------

When sklearn is installed, ``GBMClassifier``:

- inherits from ``ClassifierMixin``
- works with ``cross_val_score``, ``GridSearchCV``, ``Pipeline``
- implements ``__sklearn_tags__`` and ``_more_tags``
- ``score(X, y)`` returns accuracy

Early stopping
--------------

Early stopping monitors log-loss on the validation set:

.. code-block:: python

   model = GBMClassifier(
       n_estimators=2000,
       early_stopping_rounds=50,
       deterministic=True,
       seed=7,
   )
   model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
   print(model.best_iteration_)

Current scope
-------------

- Binary cross-entropy and multi-class softmax objectives are supported
- No ``scale_pos_weight`` parameter (use ``sample_weight`` for class imbalance)
