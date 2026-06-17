Installation
============

Requirements
------------

- Python ``3.11+``
- macOS or Linux
- a supported prebuilt wheel, or a local Rust toolchain when building from source

Install from PyPI
-----------------

.. code-block:: console

   pip install alloygbm

For ``0.12.9``, this is expected to work best on:

- macOS ``arm64``
- Linux ``x86_64`` environments compatible with the published manylinux wheel

Build from source
-----------------

.. code-block:: console

   python -m pip install --upgrade maturin
   maturin develop --manifest-path bindings/python/Cargo.toml --release

This path is the recommended fallback when no compatible prebuilt wheel is
available.

Verify the install
------------------

.. code-block:: python

   import alloygbm

   print(alloygbm.native_runtime_info())

The native runtime info object confirms that the extension module loaded
correctly.

Platform policy for ``0.12.9``
-----------------------------

- officially targeted wheel platforms:
  - macOS ``arm64``
  - Linux ``x86_64`` manylinux
- source distribution provided as a fallback
- Windows support deferred until a later release
