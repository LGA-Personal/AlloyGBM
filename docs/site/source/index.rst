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

   AlloyGBM ``0.5.0`` introduces piecewise-linear (PL) tree leaves via
   ``leaf_model="linear"`` on all three estimators, with closed-form
   ridge-solved per-leaf linear models. ``0.4.0`` introduced the opt-in
   MorphBoost adaptive split criterion (``training_mode="morph"``),
   per-iteration learning-rate schedules, and SIMD-accelerated histogram
   and EMA kernels. See :doc:`estimator` and :doc:`morphboost` for parameter
   docs, and :doc:`release` for full notes.

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
