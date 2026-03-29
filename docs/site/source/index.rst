AlloyGBM Documentation
======================

**AlloyGBM** is a Rust-first gradient boosting library for structured
regression, with a Python API oriented around native execution, deterministic
training, explicit validation, time-aware validation, and zero-copy, 
artifact-backed prediction.

The project is currently strongest on panel-style and finance-style regression
workloads, while remaining honest about weaker regimes such as broader
real-world tabular benchmarks.

.. note::

   AlloyGBM `0.1.x` is an intentionally narrow first public release series.
   Current release packaging focuses on macOS ``arm64``, Linux ``x86_64``
   manylinux wheels, and source distributions.

Getting started
---------------

If you are new to AlloyGBM, start in this order:

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   estimator
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
