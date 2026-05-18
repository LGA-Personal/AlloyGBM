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

   AlloyGBM ``0.8.0`` is a minor feature release.  It closes
   Limitation 4 (mixed linear-rank SHAP strict additivity) and adds
   LightGBM-style GOSS sampling as a new opt-in
   ``boosting_mode="goss"`` on ``GBMRegressor``, ``GBMClassifier``
   (binary), and ``GBMRanker``.  Default ``boosting_mode="standard"``
   is byte-identical to v0.7.5.  DART and joint multi-label ranking
   were scope-split out to v0.9.0 and v0.10.0 respectively to keep
   this release reviewable.  See :doc:`release` for full notes.

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
