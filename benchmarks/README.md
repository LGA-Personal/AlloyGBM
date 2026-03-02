# Benchmark Dataset Workspace

This directory organizes benchmark dataset preparation for `v0.8.3` and later hardening slices.

## Layout

- `dense_numeric/`
  - `manifest.yaml`
  - `prepare.py`
- `panel_time_series/`
  - `manifest.yaml`
  - `prepare.py`
- `histogram_stress/`
  - `manifest.yaml`
  - `prepare.py`

Generated data is written under `benchmarks/data/` (ignored by git).

## Usage

Examples:

```bash
python3 benchmarks/dense_numeric/prepare.py
python3 benchmarks/panel_time_series/prepare.py --max-rows 150000
python3 benchmarks/histogram_stress/prepare.py --rows 100000 --features 48
```

Cross-package model comparison (speed + accuracy):

```bash
python3 benchmarks/run_model_comparison.py --force-prepare
```

Outputs are written to `benchmarks/results/`:

- `model_comparison_latest.csv`
- `model_comparison_latest.json`
- `model_comparison_latest.md`

Each `prepare.py` script is self-contained and uses:

- UCI direct URLs where applicable.
- Python stdlib download flow with fallback to `curl`/`wget` when available.
- Deterministic output conventions suitable for repeatable benchmark runs.
