# Perpetual Inspiration For AlloyGBM

## Purpose

This note captures concrete implementation ideas from the vendored `perpetual/` clone that appear relevant to AlloyGBM. The goal is not to copy Perpetual wholesale. The goal is to identify:

- mechanisms that look directly portable into AlloyGBM,
- ideas that are promising but depend on broader Alloy capabilities,
- areas where Perpetual is more complex than Alloy should currently become.

The review focused primarily on `perpetual/src/`, with supporting checks in its Python bindings and packaging.

## High-Value Fits

### 1. Zero-copy and columnar data ingestion

Perpetual has a proper split between a contiguous column-major matrix and a borrowed columnar matrix with validity masks:

- `perpetual/src/data.rs`
  - `Matrix<'a, T>` at lines 112-193
  - `ColumnarMatrix<'a, T>` at lines 195-240
- `perpetual/src/booster/core.rs`
  - `fit_columnar` at lines 1378-1426

Why this matters for Alloy:

- Alloy currently trains through the Python bridge using `Vec<Vec<f32>>` rows and then copies into dense buffers in `bindings/python/src/lib.rs:148-249`.
- That is simple, but it leaves performance on the table and makes future Arrow/Polars integration awkward.

Suggested Alloy direction:

- add a native borrowed columnar matrix type beside `DatasetMatrix`,
- add Python conversion paths for pandas/polars/Arrow-like inputs,
- keep the current dense row path as the fallback,
- thread the new input representation through training and prediction without changing the public artifact format.

Best Alloy landing points:

- `crates/core/src/lib.rs`
- `bindings/python/src/lib.rs`
- `bindings/python/alloygbm/regressor.py`

### 2. Histogram accumulation optimized for cache locality

Perpetual’s histogram builder is one of the best concrete low-level references in the codebase:

- `perpetual/src/histogram.rs`
  - direct accumulation fallback at lines 96-138
  - flat-buffer accumulation at lines 140-220
  - x86 prefetching inside the flat-buffer path at lines 173-199 and 224-240

Why this matters for Alloy:

- Alloy already has a thoughtful CPU backend with scalar vs row-first vs AVX2 kernel selection in `crates/backend_cpu/src/lib.rs:12-21`, `:60-70`, and `:138-174`.
- Perpetual’s additional idea is to accumulate into compact flat arrays first and only scatter back into larger bin structs afterward. That is a plausible next step for Alloy’s backend hot loop.

Suggested Alloy direction:

- prototype a flat-buffer histogram path for the row-first kernel,
- keep the existing kernel selection and architecture detection,
- gate the new path behind workload and bin-count thresholds just as Perpetual does.

Best Alloy landing point:

- `crates/backend_cpu/src/lib.rs`

### 3. Adaptive training policy based on dataset shape

Perpetual encodes a lot of operational tuning directly into training policy:

- `perpetual/src/booster/core.rs`
  - adaptive eta schedule at lines 404-480
  - automatic leaf regularization at lines 483-515
  - leaf refinement iteration count at lines 518-560
  - training loop and policy application at lines 1441-1715

This includes:

- budget-to-eta softening by task shape,
- automatic row subsampling,
- dynamic feature sampling,
- adaptive iteration limits,
- memory-aware node allocation,
- generalization-aware tree damping/rejection.

Why this matters for Alloy:

- Alloy already has explicit iteration controls and a compact regression contract:
  - `bindings/python/alloygbm/regressor.py:107-239`
  - `crates/engine/src/lib.rs:248-260`
- What Alloy lacks is a policy layer that turns dataset shape and task characteristics into better defaults.

Suggested Alloy direction:

- do not copy Perpetual’s `budget` abstraction directly,
- do copy the pattern of dataset-aware defaults for:
  - learning rate presets,
  - row/column subsampling defaults,
  - max rounds ceilings,
  - early stopping defaults,
  - regularization defaults.

Best Alloy landing points:

- `bindings/python/alloygbm/regressor.py`
- `crates/engine/src/lib.rs`

### 4. Post-structure leaf refinement

Perpetual refines leaf outputs after tree structure is already fixed:

- `perpetual/src/booster/core.rs`
  - refinement support setup at lines 518-620
  - refinement use inside the training loop at lines 1694-1705

Why this matters for Alloy:

- Alloy currently has a much simpler training regime centered on stump fitting in `crates/engine/src/lib.rs`.
- A lightweight post-fit refinement pass for stump leaf values is much more achievable than introducing a large new tree-growing system.

Suggested Alloy direction:

- add an optional leaf-value refinement pass after stump selection,
- start with squared-error only,
- use backtracking or bounded update steps like Perpetual does,
- verify especially on low-SNR benchmark tasks where small leaf-value corrections may matter.

Best Alloy landing point:

- `crates/engine/src/lib.rs`

### 5. Optional node statistics and richer training diagnostics

Perpetual carries optional node stats and training metadata through configuration and model state:

- `perpetual/src/booster/config.rs`
  - `save_node_stats` and training/runtime options at lines 136-199
- `perpetual/src/booster/core.rs`
  - model metadata field at lines 214-216

Why this matters for Alloy:

- Alloy already has a solid binary artifact contract and metadata support:
  - `crates/core/src/lib.rs:88-127`
  - `crates/core/src/lib.rs:383-395`
- But it does not currently preserve the kind of node traffic/generalization/debug stats that later enable calibration, drift monitoring, and deeper diagnostics.

Suggested Alloy direction:

- add an optional debug/statistics section to the artifact format,
- keep predictor compatibility intact by making the section optional,
- record only what will unlock real workflows:
  - node counts,
  - maybe per-leaf training coverage,
  - maybe validation/generalization summaries.

Best Alloy landing points:

- `crates/core/src/lib.rs`
- `crates/engine/src/lib.rs`
- `crates/predictor/src/lib.rs`

### 6. Better Python-side categorical ingestion policy

Perpetual’s Python layer has stronger dataframe and category handling than Alloy’s current single-column categorical interface:

- `perpetual/package-python/python/perpetual/utils.py`
  - categorical support thresholds at lines 10-41
  - dataframe conversion at lines 130-229
  - columnar dataframe conversion starts at line 232

Notable ideas:

- auto-detect categorical columns from dataframe types,
- preserve category mappings,
- downgrade fragile categorical columns to numeric when category support is too low,
- flatten in column-major order for native consumption.

Why this matters for Alloy:

- Alloy’s current Python API exposes only `categorical_feature_index` plus explicit category values:
  - `bindings/python/alloygbm/regressor.py:124-187`
- That is useful for tests and controlled benchmarks but not a mature dataframe-facing user experience.

Suggested Alloy direction:

- keep the core native categorical mechanism simple,
- improve the Python input policy first,
- add dataframe-aware category detection and conversion,
- add category support heuristics before claiming broad native categorical support.

Best Alloy landing points:

- `bindings/python/alloygbm/regressor.py`
- `bindings/python/src/lib.rs`

## Stronger Longer-Term Ideas

## Experiment Notes

### 2026-03-24 follow-up benchmark probes

Two additional trainer-policy ideas were evaluated after the initial Perpetual-inspired commits:

- reactive stopping from observed loss-improvement decay,
- stronger split regularization for noisy small-wide datasets.

#### Reactive stopping

An engine-only experiment added a reactive stop based on the recent loss-improvement trace. The focused benchmark slice showed no measurable change:

- `dense_numeric / shallow_high_lr`
  - baseline: `0.0333s`, `rmse 0.58306`
  - reactive stop: `0.0339s`, `rmse 0.58306`
- `dense_numeric / mid_balanced`
  - baseline: `0.1263s`, `rmse 0.55514`
  - reactive stop: `0.1256s`, `rmse 0.55514`
- `dow_jones_financial / mid_balanced`
  - baseline: `0.2897s`, `rmse 3.45227`
  - reactive stop: `0.2887s`, `rmse 3.45227`

Conclusion:

- this did not buy anything on the current benchmark slice,
- the existing auto round-cap heuristic already appears to be doing the useful work for these datasets,
- reactive stopping was reverted instead of being carried forward as dead complexity.

#### Split regularization probes

Existing experiment hooks were used to probe stronger regularization:

- `min_child_hessian`
- `split_l2`
- `min_leaf_magnitude`

Findings:

- `min_child_hessian=4` helped one financial timing case in an earlier probe but was not robust. On the committed focused rerun it was worse than baseline on both runtime and RMSE, and it catastrophically hurt `dense_numeric`.
- `split_l2=2` was the most promising remaining lever:
  - `dense_numeric / mid_balanced`
    - baseline: `0.1265s`, `rmse 0.55514`
    - `split_l2=2`: `0.2239s`, `rmse 0.55226`
  - `dow_jones_financial / mid_balanced`
    - baseline: `0.2920s`, `rmse 3.45227`
    - `split_l2=2`: `0.3206s`, `rmse 3.43777`

Conclusion:

- global stronger regularization is not a safe default,
- dataset-aware `split_l2` remains a credible next step for auto policy,
- the right shape is auto-only and targeted to noisy small-wide datasets rather than a new global default.

### 7. Calibration and uncertainty estimation

Perpetual has a substantial calibration layer:

- `perpetual/src/calibration/classification.rs:8-180`
- `perpetual/src/calibration/regression.rs:6-280`

It supports:

- isotonic probability calibration,
- conformal-style set or interval calibration,
- fold-variance and min-max derived uncertainty scores,
- both dense and columnar data paths.

Why this matters for Alloy:

- Alloy’s current public surface is still centered on regression plus evaluation helpers.
- Calibration becomes much more attractive after Alloy has classification or interval workflows, and after it preserves richer training statistics.

Suggested Alloy direction:

- defer until after the current core trainer is more mature,
- keep this in mind as a future use for optional node stats and richer prediction APIs.

### 8. Drift monitoring through node traffic

Perpetual’s drift code is conceptually elegant:

- `perpetual/src/drift/calculation.rs:14-117`

The basic idea:

- score new data by traversed nodes,
- compare child traffic against training-time child traffic,
- aggregate simple statistical divergence.

Why this matters for Alloy:

- this is a relatively lightweight production feature once node stats exist,
- it does not require retraining or labels for basic data-drift detection.

Suggested Alloy direction:

- defer until optional node stats exist,
- revisit after a serving or monitoring story becomes important.

### 9. Causal wrappers as composition, not core algorithm changes

Perpetual’s causal package wraps the base booster into standard meta-learners:

- `perpetual/src/causal/metalearners.rs:14-220`

This is notable because:

- it is mostly orchestration around the core learner,
- it does not require a bespoke causal tree algorithm to provide value,
- it reuses the main booster contract consistently.

Why this matters for Alloy:

- once Alloy has a stable regressor API and better input handling, similar wrappers could be added in Python first.

Suggested Alloy direction:

- keep this as a later Python-layer expansion,
- do not let it distract from core trainer quality and benchmarking right now.

### 10. Broader explainability surface

Perpetual offers more explanation modes than Alloy:

- contribution and importance modes in `perpetual/src/booster/config.rs:13-72`
- tree prediction contribution variants in `perpetual/src/tree/predict.rs`
- partial dependence in `perpetual/src/partial_dependence.rs`

Alloy already has a serious SHAP implementation:

- `crates/shap/src/lib.rs`

So the main inspiration here is not exact SHAP, but:

- approximate contribution variants for speed,
- partial dependence,
- cleaner end-user exposure of multiple explanation modes.

Suggested Alloy direction:

- keep exact SHAP as the correctness anchor,
- consider adding PDP and one lightweight approximate contribution mode later.

## Cautions

### Do not copy the `budget` abstraction wholesale

Perpetual’s training heuristics are tightly coupled to its user-facing `budget` concept. Alloy does not currently share that product model, and forcing it in now would probably blur the API and complicate benchmark interpretation.

The useful takeaway is:

- learn from Perpetual’s adaptive policy design,
- do not inherit its public abstraction just because the internal heuristics are interesting.

### Do not replace Alloy’s artifact format with Perpetual-style JSON model IO

Perpetual’s serialized model surface is convenient for ecosystem-facing tooling, but Alloy’s sectioned binary artifact design is better aligned with:

- compatibility enforcement,
- strict predictor loading,
- optional extension sections.

The right move for Alloy is to extend its current artifact format carefully, not replace it.

### Be selective about scope

Perpetual is a much broader product:

- classification,
- ranking,
- multi-output,
- calibration,
- drift,
- causal ML,
- Python and R packaging.

Alloy currently has a much narrower and cleaner core. The value here is in borrowing high-leverage engineering ideas, not in reproducing Perpetual’s full surface area.

## Recommended Priority Order For Alloy

### Near-term

1. Zero-copy / columnar ingestion
2. Flat-buffer histogram optimization
3. Dataset-aware training defaults
4. Post-fit leaf refinement
5. Optional node statistics / debug artifact sections
6. Better Python categorical ingestion policy

### Later

1. Calibration and uncertainty
2. Drift monitoring
3. Causal wrappers
4. Broader explainability APIs

## Concrete Alloy Touchpoints

If this note is turned into execution work, the most likely Alloy files to change are:

- `crates/backend_cpu/src/lib.rs`
- `crates/core/src/lib.rs`
- `crates/engine/src/lib.rs`
- `crates/predictor/src/lib.rs`
- `bindings/python/src/lib.rs`
- `bindings/python/alloygbm/regressor.py`

## Summary

The strongest direct inspiration from Perpetual is not any single feature. It is the combination of:

- a better data-ingestion architecture,
- deeper histogram optimization,
- smarter automatic training policy,
- richer optional diagnostics.

Those are the ideas most worth porting into AlloyGBM first. Features like calibration, drift, and causal wrappers are credible future work, but they depend on Alloy first becoming stronger at the systems and training-policy layers.
