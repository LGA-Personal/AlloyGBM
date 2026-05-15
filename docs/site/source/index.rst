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

   AlloyGBM ``0.7.2`` is a documentation, supply-chain, and repo-hygiene
   release — no user-facing Python API changes.  It aligns the docs
   with the v0.7.1 surface that actually shipped (SHAP for
   piecewise-linear leaves, per-round training diagnostics, neutralized
   warm-start, LightGBM-compatible interaction constraints, and
   :class:`~alloygbm.MultiLabelGBMRanker`), hardens CI (full pytest
   suite + weekly ``cargo-audit`` / ``cargo-deny``), adds an
   ``examples/`` library, and rewrites the release operating manual.
   See :doc:`estimator` and :doc:`morphboost` for parameter docs, and
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
