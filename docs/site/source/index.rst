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

   AlloyGBM ``1.0.0`` is the first stable release. The public Python API,
   the binary artifact format, and determinism (byte-identical artifacts for
   a fixed seed, including across ``n_jobs`` thread counts) are now covered
   by semantic versioning. Four defaults changed at the 1.0 boundary --
   ``n_estimators`` 6 to 100, ``lambdarank_truncation_level`` ``None`` to
   30, the new ``dart_skip_drop=0.5``, and unrounded ``predict_proba`` --
   and three correctness defects were fixed. No artifact format change.
   See :doc:`release` for full notes.


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
