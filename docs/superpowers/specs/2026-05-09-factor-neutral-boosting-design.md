# Factor-Neutral Boosting Design

## Goal

Add a configurable factor-neutral training path that lets users provide a dense row-aligned factor exposure matrix and ask AlloyGBM to train on factor-orthogonal signal rather than raw factor-loaded gradients.

## User-Facing API

Constructor parameters on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`:

```python
GBMRegressor(
    neutralization="none",                 # "none" | "pre_target" | "per_round_gradient" | "split_penalty"
    factor_neutralization_lambda=1e-6,      # finite, >= 0 ridge added to F^T W F
    factor_penalty=0.0,                     # finite, >= 0; only active for neutralization="split_penalty"
)
```

Fit-time parameter:

```python
model.fit(X, y, factor_exposures=F)
```

`factor_exposures` is dense, row-major, finite, and shaped `(n_rows, n_factors)`. It is fit data, not constructor state, so sklearn cloning remains clean and large matrices are not embedded in estimator params.

## Mode Semantics

`neutralization="none"` preserves current behavior and ignores `factor_exposures` unless a non-`None` matrix is provided with an inactive mode, in which case Python raises a clear validation error to prevent silent user mistakes.

`neutralization="pre_target"` residualizes the regression target once before training:

```text
y_perp = y - F (F^T W F + lambda I)^-1 F^T W y
```

This mode is supported for `GBMRegressor` only. It is rejected for classification and ranking because target residualization is not well-defined for class labels or ranking relevance.

`neutralization="per_round_gradient"` projects objective gradients before each boosting round:

```text
g_perp = g - F (F^T W F + lambda I)^-1 F^T W g
```

Hessians are unchanged. This mode is supported for regression, binary classification, multiclass, and ranking. For multiclass, each class-gradient column is projected independently against the same factor projector.

`neutralization="split_penalty"` includes per-round gradient projection and subtracts a factor-load penalty from split gain:

```text
penalty = factor_penalty * || F_L^T update_L + F_R^T update_R ||^2 / max(row_count, 1)
gain_final = gain_after_existing_modes - penalty
```

For scalar leaves, `update_L` and `update_R` are the candidate scalar leaf values before any final MorphBoost depth/iteration leaf scaling. For DRO leaves, the scalar values use the DRO effective gradients. For MorphBoost, the order is: project gradients, compute standard/DRO gradient gain, blend MorphBoost information score, subtract factor penalty, then apply MorphBoost leaf scaling when storing leaves.

## Compatibility Matrix

| Feature | pre_target | per_round_gradient | split_penalty |
| --- | --- | --- | --- |
| `GBMRegressor` | supported | supported | supported |
| `GBMClassifier` | rejected | supported | supported |
| `GBMRanker` | rejected | supported | supported |
| `training_mode="morph"` | supported | supported | supported |
| `leaf_solver="dro"` | supported | supported | supported |
| `leaf_model="linear"` | supported | supported | rejected |
| warm start | rejected for non-`none` in first release | rejected for non-`none` in first release | rejected for non-`none` in first release |

Prediction-time neutralization is not included. Inference remains unchanged because the trained leaf values are baked into artifacts. The model may store neutralization metadata for introspection, but it does not store the training factor matrix.

## Rust Architecture

Core additions in `crates/core/src/lib.rs`:

```rust
pub enum NeutralizationKind {
    None,
    PreTarget,
    PerRoundGradient,
    SplitPenalty,
}

pub struct FactorNeutralizationConfig {
    pub kind: NeutralizationKind,
    pub ridge_lambda: f32,
    pub split_penalty: f32,
}

pub struct FactorExposureMatrix {
    pub row_count: usize,
    pub factor_count: usize,
    pub values: Vec<f32>,
}
```

`TrainParams` gets `neutralization_config: Option<FactorNeutralizationConfig>`. `TrainingDataset` gets `factor_exposures: Option<FactorExposureMatrix>`.

Engine additions in `crates/engine/src/lib.rs`:

- `FactorProjector` validates and precomputes the weighted Gram matrix Cholesky factor once per fit.
- `project_in_place(&mut [GradientPair])` projects gradient sums while leaving hessians unchanged.
- `residualize_targets(&mut [f32])` implements `pre_target`.
- Training validation ensures active modes have exposures with matching row count and at least one factor.

Backend additions in `crates/backend_cpu/src/lib.rs`:

- `FactorSplitContext` carries exposure matrix, factor count, split penalty, and active row mapping.
- Numeric and native categorical split scans accumulate factor sums per candidate side only when `neutralization="split_penalty"`.
- Standard split paths remain unchanged when split penalty is inactive, including SIMD standard path eligibility.

## Python Architecture

Python parameter work follows the existing convention in `CLAUDE.md`: update constructor, validation, assignment, `__repr__`, `get_params`, `set_params`, and `_params_order` together.

`factor_exposures` is accepted in `fit()` on all estimators. Ranker sorting must apply the same ordering to `factor_exposures` as it applies to `X`, `y`, and `group`.

The PyO3 bridge gets optional `factor_exposure_values`, `factor_exposure_row_count`, and `factor_exposure_factor_count` arguments for each training entry point. Python flattens NumPy inputs to row-major `float32`.

## Mathematical Claims

Public docs must say this feature projects training targets or gradients against user-provided nuisance exposures. It does not guarantee live-market neutrality and does not guarantee prediction-time zero exposure unless users also neutralize predictions externally or future work adds a prediction API that accepts factor exposures.

The split penalty reduces factor-loaded candidate updates, but it is a penalty, not a hard equality constraint. The wording should be "factor-neutralized" or "factor-orthogonalized" training, not "mathematically barred from learning beta."

## Testing Strategy

Rust unit tests:

- Projection returns input unchanged for `NeutralizationKind::None`.
- Projection makes `F^T g_perp` approximately zero for full-rank exposures.
- Ridge handles duplicate/collinear factors without failing.
- Pre-target residualization makes `F^T y_perp` approximately zero.
- Per-round regression training reduces factor correlation in predictions versus standard training on a synthetic factor-dominated dataset.
- Multiclass projects each class gradient independently.
- Split penalty lowers gain for a candidate with large factor exposure and preserves standard gain when `factor_penalty=0.0`.
- Split penalty is rejected with `leaf_model="linear"`.

Python tests:

- Constructor validation and sklearn `get_params` / `set_params` / clone compatibility.
- `fit(..., factor_exposures=F)` validates shape, finiteness, and active mode.
- `pre_target` works for regressor and is rejected for classifier/ranker.
- `per_round_gradient` trains regressor, classifier, multiclass classifier, and ranker.
- `split_penalty` trains with constant leaves and is rejected with PL leaves.
- Save/load and pickle preserve neutralization params and predictions.

Benchmark hooks:

- Add `alloygbm_factor_neutral`, `alloygbm_factor_neutral_dro`, and `alloygbm_factor_neutral_morph` arms to `benchmarks/run_model_comparison.py`.
- Add reporting for prediction-factor correlation and residual score after neutralizing predictions against evaluation factors.

## Self-Review

No placeholders remain. The design intentionally separates target/gradient projection from split penalties because split penalties require extra per-candidate factor statistics. The only deliberate first-release rejection is `split_penalty + leaf_model="linear"`, because PL leaf exposure load depends on row-level linear predictions rather than scalar leaf values.
