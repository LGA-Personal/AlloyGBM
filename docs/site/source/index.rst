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

   AlloyGBM ``0.12.9`` is a security and maintenance release on top of
   v0.12.8: it upgrades ``pyo3``/``numpy`` 0.24 → 0.29 to clear two RustSec
   advisories, moves CI off the deprecated Node 20 action runtime, and
   refreshes the dependency baseline. No user-facing API, behavior, or
   artifact-format changes — v0.12.8 artifacts load and predict identically
   under v0.12.9. The prior v0.12.8 release extended the GLM
   (``"poisson"``, ``"gamma"``, ``"tweedie"``) and ``"quantile"`` objectives
   to ``GBMRanker`` and ``MultiLabelGBMRanker``; only the Classifier /
   multiclass softmax paths still reject them. See :doc:`release` for full
   notes.


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
