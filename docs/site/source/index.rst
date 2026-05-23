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

   AlloyGBM ``0.11.1`` is a feature release shipping quantile regression
   (``objective="quantile"``) with pinball loss semantics and parameter
   ``quantile_alpha`` (default ``0.5``, strictly in ``(0.0, 1.0)``) on
   ``GBMRegressor``, utilizing a proxy Hessian, empirical leaf refinement on the
   full dataset, and a fast unweighted quickselect path. See :doc:`release` for
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
