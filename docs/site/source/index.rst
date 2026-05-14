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

   AlloyGBM ``0.7.1`` enables SHAP for piecewise-linear leaves, ships
   per-round training diagnostics, lifts the neutralized-warm-start
   rejection, adds LightGBM-compatible interaction constraints, and
   introduces ``MultiLabelGBMRanker`` for multi-output ranking.  The
   ``0.7.0`` factor-neutral boosting surface (``neutralization``,
   fit-time ``factor_exposures``), DRO scalar leaf solver, piecewise-linear
   leaves, MorphBoost adaptive split criterion, per-iteration
   learning-rate schedules, and SIMD-accelerated histogram and EMA kernels
   are all still available. See :doc:`estimator` and :doc:`morphboost` for
   parameter docs, and :doc:`release` for full notes.

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
