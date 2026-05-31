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

   AlloyGBM ``0.12.4`` is a small bugfix release on top of the v0.12.3
   refactor completion. ``GBMRegressor.__module__`` now advertises its
   public ``alloygbm.regressor`` shim path instead of the private
   ``alloygbm._regressor._core`` implementation module, so pickle payloads
   and ``repr`` no longer leak internals. The joint trainer's module-level
   documentation in ``crates/engine/src/joint/mod.rs`` is refreshed to
   reflect the v0.10.x feature parity (DART, GOSS, MorphBoost, DRO, factor
   neutralization, warm-start, leaf-wise growth, native categorical
   splits, interaction constraints) that had landed since the original
   v0.10.0 minimal scope. **No user-facing API changes, no behavioral
   changes, no new features.** Model artifacts written by v0.12.3 load
   and predict identically under v0.12.4. See :doc:`release` for full
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
