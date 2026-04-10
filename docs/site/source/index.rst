AlloyGBM Documentation
======================

**AlloyGBM** is a Rust-first gradient boosting library supporting regression,
binary classification, and learning-to-rank, with a Python API oriented around
native execution, deterministic training, explicit validation, time-aware
workflows, and zero-copy artifact-backed prediction.

The project is strongest on panel-style and finance-style workloads, with
competitive performance on general tabular benchmarks across all three task
types.

.. note::

   AlloyGBM ``0.2.0`` is a major capability expansion from the regression-only
   ``0.1.x`` series, adding classification, ranking, NaN support, model
   persistence, TreeSHAP, and many more features. See :doc:`release` for
   details.

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
