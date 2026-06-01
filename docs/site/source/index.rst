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

   AlloyGBM ``0.12.6`` is a feature release on top of v0.12.5.
   ``GBMClassifier`` and ``MultiLabelGBMRanker`` now support
   ``shap_values(X)`` and ``shap_interaction_values(X)`` — returning a
   list of ``K`` matrices, one per class (classifier) or per output
   label (ranker). Joint multi-output rankers route through new
   per-output Rust entry points; independent-mode rankers fan out to
   per-label ``GBMRanker`` calls. ``global_importance_from_artifact_bytes``
   averages over outputs so importance magnitudes remain comparable
   across single-output and multi-output models. No artifact format
   change — v0.12.5 artifacts load and predict identically under
   v0.12.6. See :doc:`release` for full notes.


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
