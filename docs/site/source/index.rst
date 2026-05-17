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

   AlloyGBM ``0.7.4`` is a bug-fix release.  It closes the remaining
   v0.7.x SHAP-additivity carryover: strict additivity
   (``atol + rtol·|predict(x)|``) now holds for ``leaf_model="linear"``
   artifacts on the default predictor-aligned binning path.  The fix
   walks the row's full path and credits ``Σⱼ wⱼ·(xⱼ − μⱼ)`` at every
   visited node — matching how ``predict`` accumulates
   ``leaf.eval_row(row)`` at each visited node.  No user-visible API
   breakage.  See :doc:`release` for full notes.

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
