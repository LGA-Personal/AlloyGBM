# Installation

## Requirements

- Python `3.11+`
- macOS or Linux
- a supported native Rust wheel, or a local Rust toolchain when building from source

## Install From PyPI

```bash
pip install alloygbm
```

This is the preferred path once public wheels are available for your platform.

## Install From Source

```bash
python -m pip install --upgrade maturin
maturin develop --manifest-path bindings/python/Cargo.toml --release
```

This builds the native Rust extension in-place against your current Python
environment.

## Verify The Install

```python
import alloygbm

print(alloygbm.native_runtime_info())
```

You should see runtime information for the loaded native extension module.

## Notes

- AlloyGBM uses a native extension module, so installation is not pure Python.
- The public Python package name is `alloygbm`.
- If native import fails, rebuild or reinstall the package rather than mixing
  source-tree imports with an older installed wheel.
- `0.12.7` wheel support:
  - macOS `arm64`
  - Linux `x86_64` via a manylinux-oriented build
  - Windows is deferred until a later release
