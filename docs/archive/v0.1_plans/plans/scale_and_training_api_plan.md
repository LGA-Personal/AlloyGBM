# AlloyGBM Scale And Training API Plan

## Purpose

Capture a clean plan for making AlloyGBM more practical on large tabular datasets, especially Numerai-scale workloads with millions of rows and tens to thousands of columns.

This note is planning only. It is not an implementation commitment.

## Motivation

The current `Kintsugi` comparison script at `/Users/lashby/Projects/Numerai/Kintsugi/research/experiments/alloygbm_comparison_ab_test.py` reports AlloyGBM as materially slower than CatBoost, LightGBM, and XGBoost on large Numerai-style data.

That script also highlights a product gap:

- users want a simple, explicit parameter surface
- users expect early stopping to work from the public estimator
- users expect scale behavior to be respectable on wide and tall datasets

## What Exists Today

### Public knobs already exposed

`GBMRegressor` already exposes:

- `learning_rate`
- `max_depth`
- `n_estimators`
- `row_subsample`
- `col_subsample`
- `early_stopping_rounds`
- `min_validation_improvement`
- `seed`
- `training_policy`
- `continuous_binning_strategy`
- `continuous_binning_max_bins`

Evidence:

- `bindings/python/alloygbm/regressor.py`
- `docs/user/gbmregressor.md`

### Engine knobs that exist but are not cleanly public

The Rust engine already has internal support for:

- `min_rows_per_leaf`
- split `l1` and `l2` regularization
- `min_child_hessian`
- minimum leaf magnitude thresholds
- validation-aware early stopping

But today these are mostly hidden behind policy code or environment variables instead of stable public parameters.

Evidence:

- `crates/engine/src/lib.rs`
- `crates/backend_cpu/src/lib.rs`

### Important contract gap

`GBMRegressor` exposes `early_stopping_rounds`, but the engine requires a validation dataset for early stopping and the current Python `fit()` API does not accept `eval_set`, `validation_fraction`, or a holdout dataset.

That means the public parameter exists, but the public training contract does not actually make it usable.

Evidence:

- `bindings/python/alloygbm/regressor.py`
- `bindings/python/src/lib.rs`
- `crates/engine/src/lib.rs`

## Main Findings On Current Scale Cost

### 1. Python preprocessing is still in the hot path

For dense numeric data, the Python layer still does substantial work before Rust trains:

- converts buffers into Python `list[float]`
- derives min and max bounds in Python
- sorts full feature columns in Python for `rank` and `quantile`
- quantizes dense values in Python

This is likely a major contributor to poor behavior on very large datasets.

Evidence:

- `bindings/python/alloygbm/regressor.py`

### 2. Dense bridge paths still rescan and copy full matrices

Even after the Python side produces a dense payload, the Rust bridge still rescans values, validates finiteness, checks whether data is pre-binned, and rebuilds dense bin buffers before training.

This is simpler than a true zero-copy or borrowed-matrix path, but it compounds cost at scale.

Evidence:

- `bindings/python/src/lib.rs`

### 3. Sampling work is still full-scan, per-round

Row and feature subsampling currently builds sampled index sets by hashing all candidate rows or features each round, then selecting and sorting the retained subset.

On millions of rows and many rounds, this can be an expensive source of allocation and CPU time.

Evidence:

- `crates/engine/src/lib.rs`

### 4. Training loop copies large prediction buffers every round

The engine keeps `predictions` and `candidate_predictions` buffers and copies full prediction arrays each round before deciding whether to commit the tree.

This is understandable for correctness and rollback, but on very large row counts it becomes a noticeable bandwidth cost.

Evidence:

- `crates/engine/src/lib.rs`

### 5. Some requested knobs are not simple surface additions

`min_data_in_leaf`, `lambda_l1`, and `lambda_l2` map naturally onto existing engine concepts.

`num_leaves` is harder:

- the current trainer is depth-constrained and depthwise
- a true `num_leaves` contract needs explicit semantics
- exposing it too early risks a misleading compatibility knob

We should not add a fake `num_leaves` parameter just because other libraries have one.

## Planning Principles

1. Keep the public API small and explicit.
2. Do not add compatibility theater.
3. Move scale-sensitive preprocessing out of Python.
4. Benchmark every change on both existing Alloy benchmarks and the Kintsugi Numerai case.
5. Prefer a staged plan where API cleanup and engine work can land independently.

## Recommended Workstreams

### Workstream 1: Measurement First

Before changing behavior, add measurement that separates:

- Python preprocessing time
- bridge conversion time
- native training time
- prediction time

Recommended outputs:

- per-stage wall clock timings
- row count and feature count
- binning strategy and bin count
- whether dense fast path was used
- rounds completed vs requested

This should be added to benchmark tooling before deeper optimization work so we stop guessing where the time is going.

### Workstream 2: Public Training Surface Cleanup

### Recommended public parameters

Promote these to first-class stable estimator parameters:

- `learning_rate`
- `n_estimators`
- `max_depth`
- `row_subsample`
- `column_subsample`
- `seed`
- `min_data_in_leaf`
- `lambda_l1`
- `lambda_l2`
- `min_child_hessian`
- `early_stopping_rounds`
- `min_validation_improvement`

Implementation note:

- keep `col_subsample` as a supported alias for compatibility
- prefer `column_subsample` in docs if we want the parameter name to read more clearly

### Parameters that should wait for explicit semantics

- `num_leaves`

Recommendation:

- do not expose `num_leaves` until the engine supports a real leaf budget contract
- if we want a near-term control, introduce `max_leaves` or `max_nodes` only after the trainer genuinely enforces it

### Fit API changes needed for early stopping

Add one of these:

- `fit(..., eval_set=(X_val, y_val))`
- `fit(..., eval_set=(X_val, y_val), eval_time_index=...)`
- `fit(..., validation_fraction=0.1, validation_strategy="random|tail")`

Recommendation:

- support explicit `eval_set` first
- optionally add `validation_fraction` later for convenience

### User-facing training outputs

Add lightweight estimator attributes after fit:

- `best_iteration_`
- `best_score_`
- `n_estimators_`
- `evals_result_`

This keeps early stopping inspectable and avoids opaque behavior.

### Workstream 3: Move Ingestion And Quantization Into Native Code

This is the highest-leverage scale improvement path.

### Phase 3A: Dense numeric native ingest

Add a native path that accepts a borrowed dense matrix view and performs:

- validation
- continuous bin derivation
- quantization

inside Rust instead of Python.

Target outcome:

- no Python `list[float]` flattening for numpy-like dense inputs
- no Python-side full-column sorting for quantile or rank binning

### Phase 3B: Columnar ingest

Add a columnar native path for dataframe-like inputs.

Why this matters:

- Numerai and research pipelines often originate in pandas, polars, or Arrow-style data
- columnar layout is a better base for feature-wise quantile and histogram work

This direction is consistent with the existing note in `docs/plans/perpetual_inspiration_for_alloygbm.md`.

### Workstream 4: Expose Real Leaf And Split Regularization

The engine already has the core ingredients. The clean job is to make them stable and coherent.

Recommended internal alignment:

- `min_data_in_leaf` -> engine `min_rows_per_leaf`
- `lambda_l1` -> split `l1` shrinkage
- `lambda_l2` -> split `l2` shrinkage
- `min_child_hessian` -> existing split threshold

Goal:

- remove reliance on environment variables for core training behavior
- put all common training controls in `TrainParams` or a close companion config object

### Workstream 5: Training-Loop Scale Optimizations

After ingestion is fixed, address the native hot loops most likely to matter on Numerai-size data.

### Priority candidates

1. Cheaper row and feature sampling
2. Reduce per-round allocation in sampling paths
3. Revisit full-buffer prediction copying each round
4. Reuse workspace buffers across rounds
5. Add larger-scale histogram memory policy and cache reuse work

Important note:

This should come after measurement and native-ingest work. Otherwise we risk tuning the wrong layer first.

### Workstream 6: Benchmark And Verification Policy

Use two benchmark families for every stage:

### Existing Alloy suite

- `benchmarks/run_model_comparison.py`
- public regression benchmarks already in repo

### Numerai-scale application benchmark

- `/Users/lashby/Projects/Numerai/Kintsugi/research/experiments/alloygbm_comparison_ab_test.py`

Add at least three tracked outcomes:

- train wall time
- peak memory or rough memory proxy
- prediction quality

This should become the release gate for any “scale improvement” claim.

## Proposed Execution Order

1. Add measurement instrumentation to benchmark and training paths.
2. Fix the estimator contract so early stopping is actually usable.
3. Promote existing regularization and leaf controls from hidden/internal to public/stable.
4. Move dense numeric preprocessing and quantization into Rust.
5. Add columnar ingest.
6. Optimize native sampling and per-round buffer behavior.
7. Revisit `num_leaves` only after leaf-budget semantics are real.

## Recommended Short-Term Scope

If we want the cleanest next increment without taking on a full trainer rewrite, the best package is:

- make early stopping real through `eval_set`
- expose `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `min_child_hessian`
- add benchmark instrumentation
- move dense numeric preprocessing out of Python

That package improves both usability and scale without forcing a growth-policy redesign.

## Risks To Avoid

- adding a fake `num_leaves` parameter with unclear behavior
- keeping important training controls behind env vars
- optimizing engine internals before removing Python preprocessing bottlenecks
- claiming early stopping support without a usable validation API
- measuring only synthetic benchmarks and not the Numerai workload

## Open Questions

1. Should Alloy keep `training_policy="auto"` as the default once more knobs become public?
2. Do we want `column_subsample` to replace `col_subsample` in the public docs, or only exist as an alias?
3. Should validation convenience be `eval_set`, `validation_fraction`, or both?
4. Is `num_leaves` actually a roadmap target, or is `max_depth + min_data_in_leaf + max_leaves` the cleaner Alloy contract?

## Immediate Next Step

Turn this plan into two concrete follow-ups:

1. a profiling task for the current Kintsugi Numerai case
2. an implementation task for estimator contract cleanup (`eval_set`, early stopping, public regularization knobs)
