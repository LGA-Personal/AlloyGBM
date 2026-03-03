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
- `dow_jones_financial/`
  - `manifest.yaml`
  - `prepare.py`

Generated data is written under `benchmarks/data/` (ignored by git).

## Usage

Examples:

```bash
python3 benchmarks/dense_numeric/prepare.py
python3 benchmarks/panel_time_series/prepare.py --max-rows 150000
python3 benchmarks/histogram_stress/prepare.py --rows 100000 --features 48
python3 benchmarks/dow_jones_financial/prepare.py --force-download
```

Cross-package model comparison (speed + accuracy):

```bash
python3 benchmarks/run_model_comparison.py --force-prepare
```

Profile matrix comparisons (shallow/mid/deep):

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default \
  --profile-seeds 7,17,29
```

Alloy continuous-feature binning strategy A/B:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default \
  --profile-seeds 7,17,29 \
  --alloy-continuous-binning-strategy quantile \
  --alloy-continuous-binning-max-bins 256
```

Supported values: `linear` (default), `rank`, `quantile`.

The benchmark runner now validates the loaded `alloygbm` runtime contract before running:

- `GBMRegressor` must expose benchmark training controls (`n_estimators`, subsampling knobs).
- native extension must expose `train_regression_artifact`.

If the runtime check fails, benchmarks stop early with an actionable error instead of silently benchmarking a stale baseline package.

Optional ultra profile (`10000` rounds) on constrained scenarios:

```bash
python3 benchmarks/run_model_comparison.py \
  --force-prepare \
  --profile-grid default_ultra \
  --profile-seeds 7 \
  --scenarios dense_numeric dow_jones_financial
```

Outputs are written to `benchmarks/results/`:

- `model_comparison_latest.csv`
- `model_comparison_latest.json`
- `model_comparison_latest.md`
- `model_comparison_profile_summary_latest.csv`
- `model_comparison_profile_summary_latest.json`

Each `prepare.py` script is self-contained and uses:

- UCI direct URLs where applicable.
- Python stdlib download flow with fallback to `curl`/`wget` when available.
- Deterministic output conventions suitable for repeatable benchmark runs.

Temporal leakage safeguards:

- `panel_time_series` uses a next-timestep target (`target_co_gt`) rather than same-timestep target duplication.
- `dow_jones_financial` excludes forward-looking `next_weeks_*` fields from model features and keeps only the target as future information.
- `run_model_comparison.py` performs timestamp-boundary splits so a timestamp cannot appear in both train and test partitions.

Continuous-feature training caveats (`v0.9.6` context):

- Alloy native training now accepts continuous float features via deterministic bridge quantization.
- Capacity/profile diagnostics should be interpreted from repeated profile runs (`--profile-grid default --profile-seeds 7,17,29`), not single-seed snapshots.
- Low-SNR financial scenarios (for example `dow_jones_financial`) can show small RMSE spread across profiles even when predictions/artifact capacity differ materially.
