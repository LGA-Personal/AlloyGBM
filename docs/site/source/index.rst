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

   AlloyGBM ``0.12.3`` completes the structural refactor: the PyO3 bridge
   (``bindings/python/src/lib.rs``, 6,619 lines) and the ``GBMRegressor``
   estimator (``bindings/python/alloygbm/regressor.py``, 4,909 lines) were
   each decomposed into focused single-responsibility modules — nine
   submodules for the bridge, a ``_regressor/`` mixin package for the
   estimator. **No user-facing API changes, no behavioral changes, no new
   features.** Model artifacts written by v0.12.2 load and predict
   identically under v0.12.3. See :doc:`release` for full notes.


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
