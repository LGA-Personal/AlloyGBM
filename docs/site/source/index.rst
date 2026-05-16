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

   AlloyGBM ``0.7.3`` is a bug-fix release.  It closes the four
   limitations queued in v0.7.2: SHAP additivity tolerance
   (``atol + rtol * |predicted|``), SHAP path-walker alignment with
   the predictor's float thresholds (new ``BinningContext``),
   MorphBoost warm-start EMA persistence (MorphMetadata artifact
   section v2), and the ``pyo3`` 0.23 → 0.24 upgrade
   (clears RUSTSEC-2025-0020).  No user-visible API breakage.  See
   :doc:`release` for full notes.

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
