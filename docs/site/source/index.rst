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

   AlloyGBM ``0.11.0`` is a minor feature release shipping two small,
   independent wins. First, pairwise **SHAP interaction values** on
   ``GBMRegressor`` via ``shap_interaction_values(X)`` -- Lundberg et al.
   (2020) Algorithm 2 in polynomial time ``O(T · L · D² · M)``, ported
   verbatim from the canonical ``slundberg/shap`` C++ reference. Second,
   three new **GLM regression objectives** -- ``"poisson"``,
   ``"gamma"``, ``"tweedie"`` (with ``tweedie_variance_power`` in
   ``(1, 2)``) -- all with log-link semantics (``predict()`` returns
   ``exp(raw)``), Newton gradients/hessians, and matching deviance
   metrics in ``alloygbm.evaluation``. Default behaviour for every
   existing user-facing API remains byte-identical to v0.10.6 when
   neither new feature is opted into. See :doc:`release` for full notes.

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
