# Fit Speed v1: Split-Scan Pruning and Relative Early Stopping

- Harness: `benchmarks/default_quality.py` (19 datasets, 5 seeds, each
  library at its own defaults) and `benchmarks/fit_speed_scale.py` (fixed
  100 rounds on larger synthetic data).
- Base: `main` at `9a25aa6` (after PR #148). Candidate: this branch.
- Environment: Linux x86_64 container, 4 threads, Python 3.13,
  scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
- Raw records: [`data/fit_speed_v1/`](data/fit_speed_v1/).

## Result

Every model is bit-identical to `main`: all 1,140 test losses (19 datasets
× 5 seeds × 2 arms × depths 4, 6 and 8) match exactly. Only fit time
changes.

Total fit seconds over 95 fits per arm:

| Depth | Auto rounds: main | Auto rounds: branch | Change | 100 rounds at 0.1: main | branch | Change |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 193.4 | 156.0 | −19% | 15.9 | 12.0 | −24% |
| 6 (default) | 228.6 | 161.4 | **−29%** | 30.3 | 19.2 | **−36%** |
| 8 | 263.9 | 175.9 | −33% | 48.8 | 27.7 | −43% |

Peers at their defaults on the same machine (depth 6): LightGBM 11.5 s,
XGBoost 12.6 s, CatBoost 157.6 s. At the same 100 rounds AlloyGBM was 2.6×
LightGBM's time and is now 1.7×. With auto rounds AlloyGBM was 1.45× CatBoost
and is now on par (1.02×), with unchanged accuracy (still ahead of LightGBM
and XGBoost at every depth, per `auto_rounds_v1.md`).

Per dataset, auto rounds, depth 6 (seconds over 5 seeds):

| Dataset | main | branch | Ratio |
|---|---:|---:|---:|
| `digits` | 56.11 | 42.86 | 0.76 |
| `linear_interaction` | 26.25 | 19.27 | 0.73 |
| `make_multiclass` | 20.46 | 13.01 | 0.64 |
| `hastie_10_2` | 20.01 | 15.63 | 0.78 |
| `zero_inflated` | 15.40 | 12.34 | 0.80 |
| `wine` | 14.46 | 6.03 | 0.42 |
| `skewed_discrete` | 10.17 | 7.59 | 0.75 |
| `make_classification` | 6.50 | 4.33 | 0.67 |
| `friedman1` | 5.89 | 4.18 | 0.71 |
| `small_wide_noisy` | 4.77 | 2.03 | 0.43 |
| `breast_cancer` | 2.70 | 1.47 | 0.55 |
| `diabetes` | 1.94 | 0.42 | 0.22 |

Larger data, fixed 100 rounds, best of 3 (no regression where bins are
dense):

| Case | main | branch | Ratio |
|---|---:|---:|---:|
| regression 200k × 30, depth 6 | 2.51 | 2.36 | 0.94 |
| regression 200k × 30, depth 10 | 5.95 | 4.98 | 0.84 |
| 12-value features 100k × 40, depth 8 | 2.01 | 1.81 | 0.90 |
| 5-class 50k × 20, depth 6 | 3.69 | 3.24 | 0.88 |

## What changed

Profiling a 600-row, 64-feature fit showed about 80% of training in
`best_split_for_feature`, the per-feature threshold scan. It evaluated every
bin, twice (once per NaN direction), at every node, although a node deep in
the tree, or a feature with few distinct values, leaves most bins empty.

1. **Empty thresholds are skipped.** A threshold whose cumulative left
   statistics equal the previous threshold's bit for bit has the same gain,
   and the argmax only moves on a material improvement, so it can never be
   chosen. Only changed thresholds are evaluated, and runs of eight
   exactly-zero bins are skipped outright. Bin 0 (only missing rows go left)
   always stays a candidate.
2. **The second NaN-direction pass is skipped when a node has no missing
   rows.** With all missing statistics exactly zero it repeats the first
   pass bit for bit.
3. **The column-major histogram kernel dispatches on the bin storage width
   once per column** instead of once per row.
4. **The auto-rounds holdout stops under its own rule.** The holdout picks
   its round count with a 0.1% relative tolerance, but the engine stopped
   the pass on an absolute rule and kept training a long flat tail (on
   `digits`, about 1,560 rounds to choose 773). A new
   `TrainParams::early_stopping_relative_tolerance` lets the engine stop at
   the same round the Python rule does; it compares in `f64` on the reported
   `f32` losses, so the decision matches round for round. The chosen and
   refit round counts are unchanged (pinned by tests).

## Where the time goes now

After these changes the histogram build (the per-row accumulation) is the
largest cost on small data, followed by per-node O(bins) work: zeroing and
copying 256-bin histograms and subtracting them even for features with a
dozen distinct values. Per-feature bin counts (as LightGBM stores them) would
cut that, and the refit pass is still about half of an auto-rounds fit.

## Reproduce

```bash
# Base build installed or on PYTHONPATH, then the branch build:
python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --output base_depth6.json
python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --output cand_depth6.json
python benchmarks/default_quality.py compare base_depth6.json cand_depth6.json --baseline alloy --candidate alloy
for d in 4 8; do  # likewise with --max-depth $d
  python benchmarks/default_quality.py run --arms alloy alloy-fixed --seeds 5 --threads 4 --max-depth $d --output cand_depth$d.json
done
python benchmarks/default_quality.py run --arms lightgbm xgboost catboost --seeds 5 --threads 4 --output peers_depth6.json
python benchmarks/fit_speed_scale.py scale_cand.json
```
