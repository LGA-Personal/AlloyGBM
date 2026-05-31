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

   AlloyGBM ``0.12.5`` is a small feature release on top of v0.12.4.
   ``GBMRegressor.shap_interaction_values(X)`` now accepts artifacts
   trained with ``leaf_model="linear"`` — the row-dependent linear
   deviation ``w_j · (x_j − μ_j)`` is credited to the diagonal of the
   interaction matrix (the regressor feature's main effect), preserving
   both row-marginal and full additivity. This closes the
   ``leaf_model="linear"`` exception that was carved out in v0.11.0. No
   other user-facing API changes; model artifacts written by v0.12.4
   load and predict identically under v0.12.5. See :doc:`release` for
   full notes.


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
