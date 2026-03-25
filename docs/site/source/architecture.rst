Architecture
============

This page gives a technical overview of how AlloyGBM is organized internally.

High-level component layout
---------------------------

The repository is split into Rust workspace crates plus Python bindings:

- ``crates/core``
  - shared data contracts, matrices, gradients, artifacts
- ``crates/engine``
  - training logic and policy-driven iteration control
- ``crates/backend_cpu``
  - CPU histogram building and split evaluation
- ``crates/predictor``
  - artifact-backed prediction
- ``crates/shap``
  - SHAP explanation support
- ``crates/categorical``
  - categorical support helpers
- ``bindings/python``
  - Python extension module and public package

Training pipeline
-----------------

At a high level, Python training flows like this:

1. Python input validation and coercion
2. dense fast-path detection for array-like inputs
3. continuous-feature quantization when needed
4. Rust engine training
5. artifact serialization
6. native predictor handle creation for later inference

Suggested figure placeholder
----------------------------

.. note::

   Suggested diagram to add here:

   - filename: ``_static/training_pipeline.png``
   - placement: directly below this note
   - concept: a left-to-right flow diagram showing
     ``Python inputs -> quantization -> engine -> backend_cpu -> artifact -> predictor handle``

   This should be the primary architecture diagram for the docs site.

Artifact design
---------------

AlloyGBM keeps a binary artifact format rather than a JSON-first model
serialization scheme. That is important because:

- predictor compatibility matters
- artifact-backed inference is part of the public Python story
- optional diagnostics can be added without changing the core predictor path

Recent design choices
---------------------

The current codebase includes several design decisions inspired by Perpetual but
adapted to AlloyGBM's narrower scope:

- dense native ingestion paths to avoid unnecessary Python row materialization
- flat histogram storage for better cache behavior
- dataset-aware training policy in ``auto`` mode
- optional node statistics for later introspection
- leaf refinement kept opt-in rather than default-on

Why the public API stays small
------------------------------

AlloyGBM is intentionally not trying to expose every booster feature through a
wide Python surface yet. The current approach is:

- keep the estimator small
- move hot paths into native code
- stay explicit about current limitations
- let benchmark evidence drive capability expansion

Suggested figure placeholder
----------------------------

.. note::

   Suggested second diagram to add here:

   - filename: ``_static/tree_node_structure.png``
   - placement: after the "Artifact design" section
   - concept: a split node with threshold bin, left/right child stats, gain,
     and optional debug metrics

   This would support both the architecture explanation and the diagnostics
   story.
