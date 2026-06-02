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

   AlloyGBM ``0.12.7`` is a feature and compatibility release on top of v0.12.6.
   The ``"quantile"`` regression objective now fully composes with
   DART boosting, MorphBoost training, and piecewise-linear
   (``leaf_model="linear"``) leaves. Removed parameter rejections, integrated
   MorphBoost schedules in leaf refinement, and supported linear leaves during
   refinement by residualizing targets against build-time linear predictions
   without double-scaling linear coefficients. No artifact format
   change — v0.12.6 artifacts load and predict identically under
   v0.12.7. See :doc:`release` for full notes.


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
