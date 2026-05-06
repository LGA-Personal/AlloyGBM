MorphBoost (Adaptive Split Criterion)
=====================================

MorphBoost is an opt-in training mode in AlloyGBM that augments the standard
gradient-gain split criterion with a normalized information-theoretic term,
plus several round-aware leaf adjustments. Implementation follows the
formulation in `Kriuk (2025), MorphBoost <https://arxiv.org/pdf/2511.13234>`_.

When To Use It
--------------

MorphBoost tends to help most on:

- Tabular problems with low signal-to-noise ratio (financial residuals,
  Numerai-style returns, etc.) where the standard gain criterion can
  overfit to spurious local-best splits.
- Workloads where you want the model to find structure that a
  pure-gradient-gain learner misses early in training.

It is not a strict upgrade — treat MorphBoost as a configuration to A/B
against ``training_mode="auto"`` rather than a default replacement.

How It Works
------------

For every candidate split, the gain is

.. code-block:: text

   gradient_score = standard XGBoost-style gradient gain
   info_score     = normalized information-gain term over the partition
   morph_weight   = tanh(iteration / 20)            # ramps in over training

   gain = (1 - info_score_weight) * gradient_score
        +  info_score_weight * info_score * morph_weight
        +  optional balance penalty

In addition:

- A per-class EMA over gradient statistics tracks recent training dynamics
  and shapes split selection during evaluation.
- Leaf values are scaled by a depth-based penalty
  (``depth_penalty_base ** (depth / 3)``) and a per-iteration shrinkage
  (``1 - morph_rate * progress``).
- An optional balance penalty discourages highly imbalanced splits.

Enabling It
-----------

Pass ``training_mode="morph"`` to any AlloyGBM estimator. The same
parameter exists on :class:`~alloygbm.GBMRegressor`,
:class:`~alloygbm.GBMClassifier`, and :class:`~alloygbm.GBMRanker`.

.. code-block:: python

   from alloygbm import GBMRegressor

   model = GBMRegressor(
       n_estimators=1200,
       max_depth=6,
       learning_rate=0.05,
       training_mode="morph",
       seed=7,
   )
   model.fit(X_train, y_train)

``training_mode`` accepts ``"auto"`` (default), ``"manual"``, or
``"morph"``.

Parameters
----------

All MorphBoost parameters are top-level keyword arguments on the estimator;
they only take effect when ``training_mode="morph"``.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Description
   * - ``morph_rate``
     - ``0.1``
     - Per-iteration leaf shrinkage rate. Range ``[0.0, 1.0]``.
   * - ``evolution_pressure``
     - ``0.2``
     - Strength of EMA-driven gain shaping. Range ``[0.0, 1.0]``.
   * - ``morph_warmup_iters``
     - ``5``
     - Initial rounds for which the morph blend collapses to the pure
       gradient gain.
   * - ``info_score_weight``
     - ``0.3``
     - Mixing weight for the information-theoretic term post-warmup.
       Range ``[0.0, 1.0]``. ``0.0`` disables the info-theoretic term.
   * - ``depth_penalty_base``
     - ``0.9``
     - Base of the leaf depth penalty. Range ``(0.0, 1.0]``. ``1.0``
       disables the penalty.
   * - ``balance_penalty``
     - ``True``
     - Whether to penalize highly imbalanced splits.
   * - ``lr_schedule``
     - ``"constant"``
     - Per-iteration LR schedule. ``"constant"`` or ``"warmup_cosine"``.
       Independent of ``training_mode`` — usable on its own.
   * - ``lr_warmup_frac``
     - ``0.1``
     - Fraction of ``n_estimators`` spent in the linear-warmup phase
       when ``lr_schedule="warmup_cosine"``. Range ``[0.0, 1.0]``. Must
       be left at the default ``0.1`` when ``lr_schedule="constant"``;
       non-default values with a constant schedule raise
       ``ValueError``.

Learning-Rate Schedules
-----------------------

``lr_schedule`` is independent of ``training_mode``. Two schedules are
supported:

- ``"constant"`` (default) — single fixed ``learning_rate`` for all rounds.
- ``"warmup_cosine"`` — linear warmup from a small fraction of
  ``learning_rate`` up to ``learning_rate`` over the first
  ``lr_warmup_frac * n_estimators`` rounds, then half-cosine decay to a
  floor of ``0.01 * learning_rate`` over the remainder.

The warmup-cosine schedule is most useful at very low ``learning_rate``
and high ``n_estimators`` (e.g. ``n_estimators=5000``,
``learning_rate=0.01``).

.. code-block:: python

   model = GBMRegressor(
       n_estimators=5000,
       learning_rate=0.01,
       training_mode="morph",
       lr_schedule="warmup_cosine",
       lr_warmup_frac=0.1,
   )

When a non-constant LR schedule is active, AlloyGBM's auto-stopping logic
becomes schedule-aware: the auto-tuned ``min_loss_improvement`` threshold
is scaled by ``current_lr / max_lr``, and empty / slightly-negative
rounds during the explicit warmup phase do not terminate training.

Persistence
-----------

Models trained with ``training_mode="morph"`` save and load identically
to auto-mode models — ``pickle``, ``save_model`` / ``load_model``, and
raw artifact export all work without extra steps. The morph configuration
is embedded as an optional artifact section so loaded models predict
consistently.
