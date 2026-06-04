AlloyGBM Documentation
======================

**AlloyGBM** is a Rust-first gradient boosting library supporting regression,
binary and multi-class classification, and learning-to-rank, with a Python API
oriented around native execution, deterministic training, explicit validation,
time-aware workflows, and zero-copy artifact-backed prediction.

The project is strongest on panel-style and finance-style workloads, with
competitive performance on general tabular benchmarks across all three task
types.

.. note::

   AlloyGBM ``0.12.8`` is a feature release on top of v0.12.7. The GLM
   (``"poisson"``, ``"gamma"``, ``"tweedie"``) and ``"quantile"`` objectives now
   work on ``GBMRanker`` and ``MultiLabelGBMRanker`` (both
   ``multi_label_mode="independent"`` and ``"joint"``), in addition to
   single-output ``GBMRegressor``. Only the Classifier / multiclass softmax
   paths still reject these objectives. No artifact format change — v0.12.7
   artifacts load and predict identically under v0.12.8. See :doc:`release`
   for full notes.


Getting started
---------------

If you are new to AlloyGBM, start in this order:

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   estimator
   classifier
   ranker
   morphboost
   validation
   explanations
   benchmarks

Technical reference
-------------------

.. toctree::
   :maxdepth: 2
   :caption: Technical Reference

   architecture
   api
   release
